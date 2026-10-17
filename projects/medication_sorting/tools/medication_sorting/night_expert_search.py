"""限时夜间专家参数对照与30分钟守护；不采RGB、不训练、不修改源码。"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess

from tools.medication_sorting.continuous_expert import execute_round
from tools.medication_sorting.robot_control import FixedBaseRobot
from tools.medication_sorting.run_randomized_experiment import Experiment, LAYOUT, PYTHON, ROOT, port_open

SERVICE = 'sy-medication-expert-night-20261016.service'
TIMER = 'sy-medication-expert-night-watch-20261016.timer'
SCREEN_SEEDS = [81000, 81001, 81002]
REVIEW_SEEDS = list(range(91000, 91010))
TERMINAL = {'complete', 'blocked', 'expired', 'stopped'}


def candidates():
    result = [{'name': 'baseline', 'parameters': dict(park_left_arm=False, herb_y_correction=0.,
                upright_herb=False, melatonin_extra_tilt=0., herb_extra_tilt=0.)}]
    for park, y, upright, tilt in itertools.product((False, True), (0., -.015, -.025),
                                                   (False, True), (0., -10.)):
        result.append({'name': f'p{int(park)}_y{round(-1000*y)}_u{int(upright)}_h{round(-tilt)}',
                       'parameters': dict(park_left_arm=park, herb_y_correction=y, upright_herb=upright,
                                          melatonin_extra_tilt=-10., herb_extra_tilt=tilt)})
    return result


def score(reports):
    return (sum(bool(r.get('success')) for r in reports),
            sum(bool(obj.get('success')) for r in reports for obj in r.get('objects', [])))


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def existing_round(folder):
    if not folder.exists():
        return None
    path = folder/'report.json'
    try:
        if path.is_file():
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                return value
    except json.JSONDecodeError:
        pass
    record = {'success': False, 'interrupted': True, 'objects': [],
              'error': '上次运行被中断或报告不完整；计失败，保留原目录，不重跑覆盖'}
    write_json(folder/'interrupted_record.json', record)
    return record


def fingerprints():
    names = ('night_expert_search.py', 'continuous_expert.py', 'run_expert.py', 'robot_control.py',
             'task_specs.py', 'randomization.py', 'run_four_objects.py', 'infer_policy.py',
             'physics_diagnostics.py', 'prepare_scene.py', 'start_orcalab.sh', 'run_randomized_experiment.py')
    paths = [Path(__file__).with_name(name) for name in names] + [LAYOUT]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def publish(output, state, **fields):
    status = read_json(output/'status.json', {})
    status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    write_json(output/'status.json', status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


def may_continue(output, until):
    if (output/'STOP').exists():
        publish(output, 'stopped', reason='收到STOP文件，停止发起下一轮')
        return False
    if datetime.now(timezone.utc) >= until:
        publish(output, 'expired', reason='到达夜间截止时间，停止发起下一轮')
        return False
    if shutil.disk_usage(ROOT).free < 100*1024**3:
        raise RuntimeError('磁盘可用空间不足100GiB，停止实验')
    return True


def run_job(output, job, seeds, until):
    folder = output/job['name']
    folder.mkdir(exist_ok=True)
    reports = []
    experiment = None
    robot = None
    try:
        for i, seed in enumerate(seeds):
            round_folder = folder/f'round_{i:04d}'
            old = existing_round(round_folder)
            if old is not None:
                reports.append(old)
                continue
            if not may_continue(output, until):
                return None
            publish(output, 'running', phase=job['phase'], current_candidate=job['name'],
                    current_round=i, current_seed=seed, completed_rounds_in_job=i,
                    successes_in_job=score(reports)[0])
            if robot is None:
                experiment = Experiment(folder/'app')
                experiment.status.update(role='night_continuous_expert_search', training_steps_requested=0,
                                         training_started=False, inference_started=False)
                experiment.start_app_with_retry()
                robot = FixedBaseRobot(frame_skip=200)
                path = Path(robot.xml_path)
                write_json(folder/'scene_source.json', {'source_xml': str(path),
                           'source_xml_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'layout': str(LAYOUT), 'layout_sha256': hashlib.sha256(LAYOUT.read_bytes()).hexdigest()})
            if not may_continue(output, until):
                return None
            row = execute_round(robot, round_folder, seed=seed, **job['parameters'])
            reports.append(row)
            publish(output, 'running', phase=job['phase'], current_candidate=job['name'],
                    completed_rounds_in_job=len(reports), successes_in_job=score(reports)[0])
    finally:
        try:
            if robot is not None:
                robot.disconnect()
        finally:
            if experiment is not None:
                experiment.stop_app()
    result = {'candidate': job, 'reports': reports, 'rounds': len(reports),
              'successes': score(reports)[0], 'completed_object_stages': score(reports)[1]}
    write_json(folder/'summary.json', result)
    return result


def render_report(output, summary):
    lines = ['# 四药品夜间专家对照结果', '',
             '这是自动专家的真实物理试验，不是π0.5推理成绩。未采集新的RGB示范，未启动训练。', '',
             '整轮成功条件：四件在同一场景连续抓放，末尾20个5Hz步同时释放、完整在正确筐内且稳定；'
             '各件均有连续3步双指接触并抬升超过6厘米的证据。', '',
             '| 候选 | 成功整轮/尝试 | 完成物品阶段数 |', '| --- | --- | --- |']
    for row in summary['screening']:
        lines.append(f'| {row["candidate"]["name"]} | {row["successes"]}/{row["rounds"]} '
                     f'| {row["completed_object_stages"]} |')
    review = summary.get('review')
    if review:
        lines += ['', f'新初态复核：{review["successes"]}/{review["rounds"]}整轮成功。',
                  '候选及全部参数见summary.json和plan.json。',
                  '连续专家达到8/10门槛，下一步建设连续RGB采集；本任务没有自动开始训练。'
                  if summary.get('expert_gate_passed') else
                  '尚未达到8/10门槛，不进行批量示范采集或微调，先分析失败轨迹。']
    if summary.get('error'):
        lines += ['', '运行异常：'+summary['error']]
    (output/'结果汇总.md').write_text('\n'.join(lines)+'\n')


def run_search(output, until):
    output.mkdir(parents=True, exist_ok=True)
    plan = {'candidates': candidates(), 'screen_seeds': SCREEN_SEEDS, 'review_seeds': REVIEW_SEEDS,
            'until': until.isoformat(), 'source_sha256': fingerprints(),
            'max_rounds': 85, 'training_enabled': False, 'new_rgb_collection_enabled': False,
            'acceptance_unchanged': True, 'original_data_and_checkpoint_unchanged': True}
    saved = read_json(output/'plan.json')
    if saved is not None and saved != plan:
        publish(output, 'blocked', reason='夜间计划或源码指纹改变，拒绝混用已有试验目录')
        raise RuntimeError('夜间计划或源码指纹改变，拒绝混用已有试验目录')
    write_json(output/'plan.json', plan)
    summary = {'screening': [], 'review': None, 'training_started': False, 'complete': False,
               'vla_success_rate_measured': False, 'expert_gate_passed': False}
    try:
        for candidate in plan['candidates']:
            if fingerprints() != plan['source_sha256']:
                raise RuntimeError('运行期间源码或场景发生变化，停止混合不同配置结果')
            row = run_job(output, dict(candidate, phase='screening'), SCREEN_SEEDS, until)
            if row is None:
                return
            summary['screening'].append(row)
            write_json(output/'summary.json', summary)
            render_report(output, summary)
        best = max(summary['screening'], key=lambda row: score(row['reports']))
        summary['selected_candidate'] = best['candidate']
        review = dict(best['candidate'], name='review_'+best['candidate']['name'], phase='review')
        row = run_job(output, review, REVIEW_SEEDS, until)
        if row is None:
            return
        summary.update(review=row, expert_gate_passed=row['successes'] >= 8, complete=True)
        publish(output, 'complete', reason='预定对照和新初态复核已结束，不再重复消耗资源',
                expert_gate_passed=summary['expert_gate_passed'],
                review_successes=row['successes'], review_rounds=row['rounds'],
                training_started=False, next_stage='continuous_rgb_collection_requires_followup')
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        publish(output, 'blocked', reason=summary['error'])
        raise
    finally:
        write_json(output/'summary.json', summary)
        render_report(output, summary)


def watch_decision(*, now, until, state, active, owned, ports_occupied, age_seconds, restarts):
    if now >= until:
        return 'expired'
    if state in TERMINAL:
        return 'finished'
    if not owned:
        return 'blocked'
    if active == 'active' and age_seconds <= 1800:
        return 'wait'
    if active not in ('active', 'failed', 'inactive'):
        return 'wait'
    if restarts >= 2 or (active != 'active' and ports_occupied):
        return 'blocked'
    return 'restart'


def systemctl(*args):
    return subprocess.run(['systemctl', '--user', *args], check=True, capture_output=True,
                          text=True, timeout=180)


def watch(output, until):
    now = datetime.now(timezone.utc)
    status = read_json(output/'status.json', {})
    previous = read_json(output/'watchdog.json', {})
    properties = dict(line.split('=', 1) for line in systemctl('show', SERVICE,
                     '-p', 'ExecStart', '-p', 'WorkingDirectory', '-p', 'KillMode',
                     '-p', 'ActiveState').stdout.splitlines() if '=' in line)
    expected = f'argv[]={PYTHON} -m tools.medication_sorting.night_expert_search --output {output} --until {until.isoformat()} ;'
    owned = expected in properties.get('ExecStart', '') and properties.get('WorkingDirectory') == str(ROOT) \
        and properties.get('KillMode') == 'control-group'
    age = (now-datetime.fromisoformat(status['updated_at'])).total_seconds() if status.get('updated_at') else 1900
    action = watch_decision(now=now, until=until, state=status.get('state'),
            active=properties.get('ActiveState'), owned=owned, ports_occupied=port_open(12345) or port_open(50051),
            age_seconds=age, restarts=previous.get('restarts', 0))
    restarts = previous.get('restarts', 0)
    if action == 'restart':
        systemctl('restart', SERVICE)
        restarts += 1
    elif action in ('expired', 'finished', 'blocked'):
        systemctl('stop', TIMER)
    record = dict(action=action, checked_at=now.isoformat(), restarts=restarts,
                  owned=owned, service_state=properties.get('ActiveState'), status_age_seconds=age)
    write_json(output/'watchdog.json', record)
    if action != 'wait':
        print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--until', type=datetime.fromisoformat, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.until.tzinfo is None:
        parser.error('截止时间必须包含时区')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data/medication_sorting/overnight'):
        parser.error('夜间结果必须写入本任务独立的overnight目录')
    output.mkdir(parents=True, exist_ok=True)
    with (output/('.watch.lock' if args.watch else '.worker.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.watch:
            watch(output, args.until)
        else:
            run_search(output, args.until)


if __name__ == '__main__':
    main()
