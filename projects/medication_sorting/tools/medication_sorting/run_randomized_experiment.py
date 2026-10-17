"""四药品160条采集、两万步微调和40次纯策略评估的受控长任务。"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from tools.medication_sorting.task_specs import TARGETS


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT/'.venv/bin/python')
WRAPPER = str(ROOT/'tools/medication_sorting/pi05_python.sh')
LAYOUT = ROOT/'data/medication_sorting/scenes/drag_classfication_bottles70.json'
EXPERT_PROFILE = 'randomized_tabletop_70'
RANDOMIZATION = 'tabletop_v1'


def require_complete_collection(report):
    if (report.get('complete') is not True or report.get('accepted') != 160
        or set(report.get('by_object', {})) != set(TARGETS)
        or any(type(row.get('accepted')) is not int or row['accepted'] != 40
               for row in report['by_object'].values())):
        raise ValueError('只有四种药品各40条合格回合才能进入训练')


def training_command(repo_id, experiment):
    return [WRAPPER, '-m', 'tools.medication_sorting.pi05_adapter', '--repo-id', repo_id,
            '--batch-size', '1', '--action-horizon', '10', 'train', '--',
            f'--exp-name={experiment}',
            f'--weight-loader.params-path={ROOT}/data/medication_sorting/pretrained/pi05_base/params',
            '--num-train-steps=20000', '--save-interval=2000', '--log-interval=200',
            '--lr-schedule.warmup-steps=500', '--lr-schedule.peak-lr=2e-5',
            '--lr-schedule.decay-steps=20000', '--lr-schedule.decay-lr=2e-6']


def evaluation_specs():
    return [{'target': target, 'episodes': 10, 'seed_start': 4000000000+index*10000}
            for index, target in enumerate(TARGETS)]


def require_complete_evaluation(summary, spec):
    expected = list(range(spec['seed_start'], spec['seed_start']+spec['episodes']))
    actual = [row.get('seed') for row in summary.get('episodes', [])]
    if summary.get('task_count') != 10 or actual != expected:
        raise ValueError('评估未完成该物品全部10个预定seed，不能标记完成')


def require_complete_conversion(output, expected, repo_id):
    manifest = json.loads((Path(output)/'g1_conversion_manifest.json').read_text())
    sources = [str(Path(row['path']).resolve()) for row in manifest.get('accepted', [])]
    wanted = [str(Path(row['source']).resolve()) for row in expected]
    if (manifest.get('repo_id') != repo_id or manifest.get('fps') != 5
        or manifest.get('converted_episodes') != len(wanted) or manifest.get('rejected') != []
        or sorted(sources) != sorted(wanted) or len(set(sources)) != len(wanted)):
        raise ValueError('转换数据不完整或来源与实际划分不同，拒绝训练')
    info = json.loads((Path(output)/'meta/info.json').read_text())
    if info.get('total_episodes') != len(wanted) or info.get('fps') != 5:
        raise ValueError('LeRobot落盘回合数量或频率不一致')
    return manifest


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            return True
    except OSError:
        return False


class Experiment:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.output/'pipeline_logs'
        self.logs.mkdir(exist_ok=True)
        self.app = self.policy = None
        self.counter = max((int(path.name.split('-', 1)[0]) for path in self.logs.iterdir()
                            if path.name.split('-', 1)[0].isdigit()), default=0)
        if (self.output/'status.json').exists():
            history = self.output/'status_history'
            history.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copyfile(self.output/'status.json', history/f'{stamp}-{os.getpid()}.json')
        self.status = {'state': 'starting', 'training_steps_requested': 20000,
                       'training_completed': False, 'inference_completed': False,
                       'expert_profile': EXPERT_PROFILE, 'randomization_profile': RANDOMIZATION,
                       'output': str(self.output), 'continuous_sorting_validated': False}
        self.environment = dict(os.environ, PYTHONUNBUFFERED='1', SY_ROOT=str(ROOT),
                                HF_LEROBOT_HOME=str(ROOT/'data/medication_sorting/lerobot'),
                                OPENPI_DATA_HOME=str(ROOT/'runtime/cache/openpi'))
        self.repo = 'local/g1_four_medication_160_20261014_train'
        self.validation_repo = 'local/g1_four_medication_160_20261014_validation'
        self.experiment = 'g1_four_medication_160_20261014_20k'

    def publish(self, state, **values):
        self.status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        temporary = self.output/'status.json.tmp'
        temporary.write_text(json.dumps(self.status, ensure_ascii=False, indent=2))
        temporary.replace(self.output/'status.json')
        print(json.dumps(self.status, ensure_ascii=False), flush=True)

    def run(self, command, name, *, allowed=(0,), timeout=7200, environment=None):
        self.counter += 1
        log = self.logs/f'{self.counter:04d}-{name}.log'
        self.publish(name, current_command=command, current_log=str(log))
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment or self.environment,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            finally:
                self.stop_owned(child)
        if code not in allowed:
            raise RuntimeError(f'{name}退出码{code}，日志：{log}')
        return code

    def start_app(self):
        if port_open(12345) or port_open(50051):
            raise RuntimeError('已有OrcaLab占用端口，拒绝操作未归本长任务管理的窗口')
        self.counter += 1
        log = self.logs/f'{self.counter:04d}-orcalab.log'
        with log.open('w') as stream:
            self.app = subprocess.Popen(['bash', str(ROOT/'tools/medication_sorting/start_orcalab.sh')],
                                        cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.publish('starting_orcalab', app_pid=self.app.pid, app_log=str(log))
        deadline = time.monotonic()+180
        while not port_open(12345):
            if self.app.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'OrcaLab启动失败，日志：{log}')
            time.sleep(1)
        self.run([PYTHON, str(ROOT/'tools/medication_sorting/prepare_scene.py'),
                  '--layout', str(LAYOUT), '--record-seconds', '0'], 'load_scene', timeout=300)

    def start_app_with_retry(self):
        for attempt in range(1, 4):
            try:
                self.start_app()
                return
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                if self.app is None:
                    raise
                self.publish('scene_start_retry', startup_attempt=attempt, last_start_error=str(error))
                try:
                    self.stop_app()
                except (RuntimeError, subprocess.TimeoutExpired) as cleanup_error:
                    self.publish('scene_start_cleanup_failed', startup_cleanup_error=str(cleanup_error))
                if attempt == 3:
                    raise RuntimeError('场景连续三次启动失败，保留日志并停止') from error

    def stop_app(self):
        if self.app is None:
            return
        try:
            self.backup_stop_scene()
        finally:
            self.stop_owned(self.app)
            self.status['last_app_exit_code'] = self.app.returncode
            self.app = None
            deadline = time.monotonic()+30
            while port_open(12345) or port_open(50051):
                if time.monotonic() > deadline:
                    raise RuntimeError('本任务OrcaLab退出后端口仍未释放')
                time.sleep(.5)

    def backup_stop_scene(self):
        if self.app.poll() is None and port_open(12345):
            backup = self.output/f'editor_backup_{self.counter:04d}.json'
            code = (
                'import asyncio; from prepare_scene import create_orcalab_client, decode_result\n'
                'async def save():\n'
                ' async with create_orcalab_client(12345,timeout=30) as client:\n'
                f'  decode_result(await client.call_tool_mcp(name="save_layout",arguments={{"layout_path":{str(backup)!r}}},timeout=30))\n'
                '  state=decode_result(await client.call_tool_mcp(name="get_simulation_state",arguments={},timeout=30))\n'
                '  if state.get("state")=="Running":\n'
                '   decode_result(await client.call_tool_mcp(name="stop_simulation",arguments={},timeout=30))\n'
                'asyncio.run(save())\n')
            environment = dict(self.environment, PYTHONPATH=str(ROOT/'tools/medication_sorting'))
            self.run([PYTHON, '-c', code], 'backup_stop_scene', timeout=120, environment=environment)
            json.loads(backup.read_text())

    @staticmethod
    def stop_owned(process):
        if process is None:
            return
        # 已回收的PID若重新出现，不能再根据旧句柄发送信号。
        if process.poll() is not None and Path(f'/proc/{process.pid}').exists():
            raise RuntimeError('启动器PID已被复用，拒绝清理未知进程组')
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=10)
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # 启动器可能先退出，同组GUI/渲染子进程仍需清理。
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)

    def collect(self):
        raw = self.output/'raw'
        no_progress = 0
        for batch in range(200):
            before = json.loads((raw/'progress.json').read_text()) if (raw/'progress.json').exists() else {'accepted': 0}
            if before.get('complete'):
                require_complete_collection(before)
                return
            self.start_app_with_retry()
            try:
                self.run([PYTHON, '-m', 'tools.medication_sorting.collect_balanced', '--layout', str(LAYOUT),
                          '--profile', EXPERT_PROFILE, '--randomization-profile', RANDOMIZATION,
                          '--per-target', '40', '--max-attempts', '200', '--max-new-attempts', '4',
                          '--output', str(raw)], f'collect_batch_{batch:03d}', allowed=(0, 1))
            finally:
                self.stop_app()
            report = json.loads((raw/'progress.json').read_text())
            self.publish('collection_batch_reviewed', accepted=report['accepted'], attempts=report['attempts'],
                         by_object=report['by_object'])
            if report['complete']:
                require_complete_collection(report)
                return
            no_progress = no_progress+1 if report['accepted'] <= before['accepted'] else 0
            if no_progress >= 3 or any(row['attempts'] >= 200 and row['accepted'] < 40 for row in report['by_object'].values()):
                raise RuntimeError('连续三批无新增合格数据或触及尝试上限，已停止；禁止用不合格数据开始训练')
        raise RuntimeError('采集批次超过上限，未满足160条验收')

    def train(self):
        require_complete_collection(json.loads((self.output/'raw/progress.json').read_text()))
        split = self.output/'split'
        self.run([PYTHON, '-m', 'tools.medication_sorting.split_episodes', '--input', str(self.output/'raw'),
                  '--output', str(split), '--per-object-train', '32', '--per-object-validation', '8',
                  '--train-repo-id', self.repo, '--validation-repo-id', self.validation_repo], 'split')
        manifest = json.loads((split/'split_manifest.json').read_text())
        if len(manifest['train']) != 128 or len(manifest['validation']) != 32:
            raise ValueError('划分没有满足128条训练和32条独立验证')
        for partition, repo in [('train', self.repo), ('validation', self.validation_repo)]:
            self.run([WRAPPER, '--cpu', '-m', 'tools.medication_sorting.convert_episodes',
                      '--input', str(split/partition), '--output', str(Path(self.environment['HF_LEROBOT_HOME'])/repo),
                      '--repo-id', repo, '--fps', '5', '--terminal-phase', 'settle_final'], f'convert_{partition}')
            require_complete_conversion(Path(self.environment['HF_LEROBOT_HOME'])/repo, manifest[partition], repo)
        self.run([WRAPPER, '--cpu', '-m', 'tools.medication_sorting.pi05_adapter', '--repo-id', self.repo,
                  'norm-stats'], 'training_only_norm_stats')
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 14*1024**3:
            raise RuntimeError('训练前可用主机内存不足14GiB，保留数据，不冒险启动训练')
        environment = dict(self.environment, XLA_PYTHON_CLIENT_MEM_FRACTION='0.80')
        self.run(training_command(self.repo, self.experiment), 'train_20000', timeout=21600, environment=environment)
        checkpoint = ROOT/'data/medication_sorting/checkpoints/pi05_g1_medication_sorting_lora'/self.experiment/'19999'
        metadata = json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
        if not metadata.get('commit_timestamp_nsecs') or not (checkpoint/'params/manifest.ocdbt').is_file():
            raise RuntimeError('最终检查点未完整提交，不启动推理服务')
        self.checkpoint = checkpoint
        self.publish('training_completed', training_completed=True, checkpoint=str(checkpoint),
                     split_manifest=str(split/'split_manifest.json'))

    def evaluate(self):
        if port_open(8000):
            raise RuntimeError('8000端口已有非本任务策略服务，拒绝覆盖')
        command = [WRAPPER, '-m', 'tools.medication_sorting.pi05_adapter', '--repo-id', self.repo,
                   '--action-horizon', '10', 'serve', '--', f'--policy.dir={self.checkpoint}', '--port=8000']
        with (self.logs/'policy_server.log').open('w') as stream:
            self.policy = subprocess.Popen(command, cwd=ROOT,
                            env=dict(self.environment, XLA_PYTHON_CLIENT_MEM_FRACTION='0.65'),
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+600
        while not port_open(8000):
            if self.policy.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('真实检查点策略服务启动失败，请查看policy_server.log')
            time.sleep(2)
        summaries = []
        for spec in evaluation_specs():
            self.start_app_with_retry()
            try:
                output = self.output/'evaluation'/spec['target']
                self.run([PYTHON, '-m', 'tools.medication_sorting.evaluate_policy',
                          '--checkpoint-reference', str(self.checkpoint),
                          '--training-manifest', str(self.output/'split/split_manifest.json'),
                          '--target', spec['target'], '--profile', EXPERT_PROFILE, '--layout', str(LAYOUT),
                          '--randomization-profile', RANDOMIZATION, '--episodes', str(spec['episodes']),
                          '--seed-start', str(spec['seed_start']), '--steps', '150', '--execute-chunk', '4',
                          '--control-hz', '5', '--output', str(output)], f'evaluate_{spec["target"]}', allowed=(0, 1))
                summary = json.loads((output/'summary.json').read_text())
                require_complete_evaluation(summary, spec)
                summaries.append({'target': spec['target'], **summary})
            finally:
                self.stop_app()
        result = {'tasks': sum(row['task_count'] for row in summaries),
                  'successes': sum(row['success_count'] for row in summaries), 'by_object': summaries,
                  'checkpoint': str(self.checkpoint), 'continuous_sorting_validated': False}
        result['success_rate'] = result['successes']/result['tasks']
        (self.output/'inference_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        self.publish('complete', inference_completed=True, inference_successes=result['successes'],
                     inference_tasks=result['tasks'], inference_success_rate=result['success_rate'])

    def execute(self):
        if shutil.disk_usage(ROOT).free < 100*1024**3:
            raise RuntimeError('剩余空间不足100GiB，停止实验')
        try:
            self.collect()
            self.train()
            self.evaluate()
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
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            raise KeyboardInterrupt(f'收到终止信号{signum}')
        signal.signal(signal.SIGTERM, stop)
        Experiment(args.output).execute()


if __name__ == '__main__':
    main()
