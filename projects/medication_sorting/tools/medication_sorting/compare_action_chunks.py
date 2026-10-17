"""对照现有π0.5的1/2/4步闭环执行；不采集、不训练，不覆盖旧评估。"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

from tools.medication_sorting.evaluate_policy import active_controllers
from tools.medication_sorting.run_randomized_experiment import (
    ROOT, PYTHON, WRAPPER, LAYOUT, EXPERT_PROFILE, RANDOMIZATION, Experiment,
    evaluation_specs, port_open, require_complete_collection, require_complete_evaluation,
)


SOURCE = ROOT/'data/medication_sorting/multi_object/randomized160-20261014/production'
CHECKPOINT = ROOT/'data/medication_sorting/checkpoints/pi05_g1_medication_sorting_lora/g1_four_medication_160_20261014_20k/19999'
CHUNKS = (1, 2, 4)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_specs():
    return [{**spec, 'execute_chunk': chunk} for chunk in CHUNKS for spec in evaluation_specs()]


def evaluation_command(spec, output):
    return [PYTHON, '-m', 'tools.medication_sorting.evaluate_policy',
            '--checkpoint-reference', str(CHECKPOINT),
            '--training-manifest', str(SOURCE/'split/split_manifest.json'),
            '--target', spec['target'], '--profile', EXPERT_PROFILE, '--layout', str(LAYOUT),
            '--randomization-profile', RANDOMIZATION, '--episodes', str(spec['episodes']),
            '--seed-start', str(spec['seed_start']), '--steps', '150',
            '--execute-chunk', str(spec['execute_chunk']), '--control-hz', '5', '--output', str(output)]


def configuration():
    require_complete_collection(json.loads((SOURCE/'raw/progress.json').read_text()))
    if json.loads((SOURCE/'status.json').read_text()).get('training_completed') is not True:
        raise ValueError('原实验未确认训练完成，拒绝加载不明检查点')
    committed = json.loads((CHECKPOINT/'_CHECKPOINT_METADATA').read_text()).get('commit_timestamp_nsecs')
    stats = list((CHECKPOINT/'assets').rglob('norm_stats.json'))
    if type(committed) is not int or committed <= 0 or len(stats) != 1:
        raise ValueError('检查点未完整提交或归一化统计量不唯一')
    return {'checkpoint_provenance': {
                'path': str(CHECKPOINT.resolve()), 'commit_timestamp_nsecs': committed,
                'params_manifest_sha256': sha256(CHECKPOINT/'params/manifest.ocdbt'),
                'params_metadata_sha256': sha256(CHECKPOINT/'params/_METADATA'),
                'norm_stats_sha256': sha256(stats[0])},
            'layout_sha256': sha256(LAYOUT),
            'training_manifest_sha256': sha256(SOURCE/'split/split_manifest.json'),
            'implementation_sha256': {name: sha256(Path(__file__).with_name(name)) for name in (
                'compare_action_chunks.py', 'infer_policy.py', 'evaluate_policy.py',
                'pi05_adapter.py', 'rendered_rgb.py', 'robot_control.py', 'task_specs.py',
                'randomization.py')},
            'control_hz': 5, 'steps': 150, 'action_horizon': 10, 'chunks': list(CHUNKS),
            'specs': comparison_specs(), 'evaluation_only': True,
            'seed_role': 'previous_test_reused_as_paired_tuning_set',
            'independent_test_completed': False, 'continuous_sorting_validated': False}


def require_block(summary, spec, config):
    require_complete_evaluation(summary, spec)
    controls = summary.get('controls', {})
    expected = {'execute_chunk': spec['execute_chunk'], 'steps': 150,
                'control_hz': 5, 'camera_backend': 'viewport'}
    if any(controls.get(k) != v for k, v in expected.items()):
        raise ValueError('动作段对照的执行参数不一致')
    for row in summary['episodes']:
        report = json.loads(Path(row['report_path']).read_text())
        provenance = (report.get('policy_metadata') or {}).get('checkpoint_provenance', {})
        expected_success = (row.get('returncode') == 0 and report.get('success') is True
                            and report.get('grasp_verified') is True and not report.get('error'))
        valid = (all(provenance.get(k) == v for k, v in config['checkpoint_provenance'].items())
                 and report.get('experiment_layout_sha256') == config['layout_sha256']
                 and report.get('seed') == row['seed']
                 and report.get('object_name') == spec['target']
                 and report.get('expert_profile') == EXPERT_PROFILE
                 and report.get('randomization_profile') == RANDOMIZATION
                 and report.get('controller') == 'pi05_checkpoint_policy'
                 and report.get('expert_actions_used') is False and not report.get('error')
                 and row.get('returncode') in (0, 1)
                 and all(row.get(key) is True for key in (
                     'target_matches', 'checkpoint_path_matches', 'reported_reset_matches_seed'))
                 and type(row.get('success')) is bool and type(row.get('grasp_verified')) is bool
                 and row['success'] == expected_success
                 and row['grasp_verified'] == (report.get('grasp_verified') is True))
        if not valid:
            raise ValueError(f'回合证据或模型来源不符，保留现场而不宣称对照完成：{row["report_path"]}')
    if summary.get('success_count') != sum(row['success'] for row in summary['episodes']):
        raise ValueError('对照成功计数与逐回合记录不符')


def summarize_blocks(blocks):
    actual = [block['spec'] for block in blocks]
    if actual != comparison_specs():
        raise ValueError('动作段对照缺少或重复预定批次，不能汇总完成')
    by_chunk = []
    for chunk in CHUNKS:
        rows = []
        for block in blocks:
            if block['spec']['execute_chunk'] != chunk:
                continue
            require_complete_evaluation(block['summary'], block['spec'])
            episodes = block['summary']['episodes']
            rows.append({'target': block['spec']['target'], 'tasks': len(episodes),
                         'successes': sum(row['success'] for row in episodes),
                         'grasp_verified': sum(row['grasp_verified'] for row in episodes),
                         'no_grasp': sum(not row['grasp_verified'] for row in episodes),
                         'post_grasp_failure': sum(row['grasp_verified'] and not row['success'] for row in episodes)})
        totals = {key: sum(row[key] for row in rows) for key in (
            'tasks', 'successes', 'grasp_verified', 'no_grasp', 'post_grasp_failure')}
        by_chunk.append({'execute_chunk': chunk, **totals,
                         'success_rate': totals['successes']/totals['tasks'], 'by_object': rows})
    selected = max(by_chunk, key=lambda row: (row['successes'], row['execute_chunk']))
    return {'complete': True, 'total_tasks': sum(row['tasks'] for row in by_chunk),
            'by_chunk': by_chunk, 'selected_chunk': selected['execute_chunk'],
            'selection_rule': 'highest_success_count_then_larger_chunk_on_tie',
            'seed_role': 'previous_test_reused_as_paired_tuning_set',
            'independent_test_completed': False, 'continuous_sorting_validated': False}


class ChunkComparison(Experiment):
    def __init__(self, output):
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f'对照输出目录已存在，拒绝覆盖：{output}')
        config = configuration()
        output.mkdir(parents=True, exist_ok=False)
        super().__init__(output)
        self.config = config
        write_json(self.output/'configuration.json', config)
        self.status = {'state': 'starting', 'evaluation_only': True,
                       'checkpoint': str(CHECKPOINT), 'completed_blocks': 0, 'planned_blocks': 12,
                       'planned_tasks': 120, 'output': str(self.output),
                       'independent_test_completed': False, 'continuous_sorting_validated': False}

    def start_policy(self, chunk):
        if port_open(8000):
            raise RuntimeError('8000端口已有策略服务，拒绝接管')
        self.counter += 1
        log = self.logs/f'{self.counter:04d}-policy_chunk_{chunk}.log'
        command = [WRAPPER, '-m', 'tools.medication_sorting.pi05_adapter', '--repo-id', self.repo,
                   '--action-horizon', '10', 'serve', '--', f'--policy.dir={CHECKPOINT}', '--port=8000']
        with log.open('w') as stream:
            self.policy = subprocess.Popen(command, cwd=ROOT,
                env=dict(self.environment, XLA_PYTHON_CLIENT_MEM_FRACTION='0.65'),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.publish('starting_policy', active_chunk=chunk, policy_pid=self.policy.pid, policy_log=str(log))
        deadline = time.monotonic()+600
        while not port_open(8000):
            if self.policy.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'对照策略服务启动失败：{log}')
            time.sleep(2)

    def stop_policy(self):
        self.stop_owned(self.policy)
        self.policy = None
        deadline = time.monotonic()+30
        while port_open(8000):
            if time.monotonic() > deadline:
                raise RuntimeError('对照策略退出后8000端口仍被占用，不接管未知进程')
            time.sleep(.5)

    def execute(self):
        if any(port_open(port) for port in (12345, 50051, 8000)):
            raise RuntimeError('评估端口已被其他进程占用，拒绝启动')
        if active_controllers():
            raise RuntimeError('已有采集或推理控制进程，拒绝并发对照')
        blocks = []
        try:
            for chunk in CHUNKS:
                self.start_policy(chunk)
                try:
                    for spec in (row for row in comparison_specs() if row['execute_chunk'] == chunk):
                        if shutil.disk_usage(ROOT).free < 100*1024**3:
                            raise RuntimeError('剩余空间不足100GiB，保留结果并停止对照')
                        if configuration() != self.config:
                            raise ValueError('模型、场景、划分或实现文件在对照期间变化，停止混用结果')
                        output = self.output/f'chunk_{chunk}'/spec['target']
                        self.status.update(active_chunk=chunk, active_target=spec['target'],
                                           active_summary=str(output/'summary.json'))
                        self.start_app_with_retry()
                        try:
                            self.run(evaluation_command(spec, output),
                                     f'evaluate_chunk_{chunk}_{spec["target"]}', allowed=(0, 1), timeout=14400)
                        finally:
                            self.stop_app()
                        if configuration() != self.config:
                            raise ValueError('模型、场景、划分或实现文件在对照期间变化，停止混用结果')
                        summary = json.loads((output/'summary.json').read_text())
                        require_block(summary, spec, self.config)
                        blocks.append({'spec': spec, 'summary': summary})
                        progress = [{'spec': row['spec'], 'tasks': row['summary']['task_count'],
                                     'successes': row['summary']['success_count']} for row in blocks]
                        write_json(self.output/'progress.json', {'complete': False, 'blocks': progress,
                                   'completed_tasks': sum(row['tasks'] for row in progress), 'planned_tasks': 120})
                        self.publish('block_completed', completed_blocks=len(blocks),
                                     completed_tasks=sum(row['tasks'] for row in progress))
                finally:
                    self.stop_policy()
            if configuration() != self.config:
                raise ValueError('模型、场景、划分或实现文件在对照期间变化，停止混用结果')
            result = summarize_blocks(blocks)
            write_json(self.output/'comparison.json', result)
            self.publish('complete', completed_blocks=12, completed_tasks=120,
                         selected_chunk=result['selected_chunk'], by_chunk=result['by_chunk'])
        except BaseException as error:
            self.publish('stopped_with_error', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            try:
                self.stop_app()
            finally:
                self.stop_owned(self.policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = ROOT/'data/medication_sorting/policy_evaluations'
    parent.mkdir(parents=True, exist_ok=True)
    with (parent/'.chunk_comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            raise KeyboardInterrupt(f'动作段对照收到终止信号{signum}')
        signal.signal(signal.SIGTERM, stop)
        ChunkComparison(args.output).execute()


if __name__ == '__main__':
    main()
