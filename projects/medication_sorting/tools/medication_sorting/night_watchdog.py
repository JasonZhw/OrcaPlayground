"""定时检查药品采集；仅在可续采阶段恢复本项目systemd服务。"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess

from tools.medication_sorting.run_randomized_experiment import (
    ROOT, PYTHON, port_open, require_complete_collection,
)


SERVICE = 'sy-medication-160-20k.service'
TIMER = 'sy-medication-night-watch.timer'
OUTPUT = ROOT/'data/medication_sorting/multi_object/randomized160-20261014/production'
COLLECTION_STATES = {'starting', 'starting_orcalab', 'load_scene', 'backup_stop_scene',
                     'collection_batch_reviewed', 'stopped_with_error'}


def completion_decision(status, progress):
    finished = status.get('state') == 'complete' and status.get('inference_completed') is True
    if progress.get('complete') is not True and not finished:
        return None
    try:
        require_complete_collection(progress)
    except (ValueError, TypeError, AttributeError):
        return {'action': 'blocked', 'reason': '采集完成标记与四种各40条配额不符，需检查数据',
                'missing_since': None}
    return {'action': 'finished' if finished else 'collection_finished',
            'reason': '实验已完成，不再恢复软件' if finished else '四种药品各40条已采满，结束采集守护，不重启下游阶段',
            'missing_since': None}


def decide_action(*, now, until, active_state, status, progress, downstream_started,
                  ports_ready, last_activity, previous, free_bytes):
    def result(action, reason, **values):
        return {'action': action, 'reason': reason, 'missing_since': None, **values}

    if until is not None and now >= until:
        return result('expired', '夜间守护到期，仅停止守护定时器，不停止主任务')
    if (completion := completion_decision(status, progress)) is not None:
        return completion
    state = status.get('state', '')
    collecting = state.startswith('collect_batch_') or state in COLLECTION_STATES
    if downstream_started or not collecting or status.get('training_completed'):
        action = 'wait' if active_state in {'active', 'activating', 'deactivating'} else 'blocked'
        return result(action, '已进入下游阶段或阶段未知，不重放采集训练流程')
    if free_bytes < 100*1024**3:
        return result('blocked', '磁盘不足100GiB，不循环重启')
    for row in progress.get('by_object', {}).values():
        if row.get('attempts', 0) >= 200 and row.get('accepted', 0) < 40:
            return result('blocked', '目标尝试次数耗尽，保留失败证据，不降低质量')
    error = status.get('error', '')
    if any(text in error for text in ('ValueError', '不一致', '拒绝覆盖', '缺少批次配置', '尝试上限')):
        return result('blocked', '需要先检查数据或配置错误，不能盲目重启')
    if active_state in {'inactive', 'failed'}:
        if ports_ready:
            return result('blocked', '服务已停止但端口仍被其他进程占用，不接管外部窗口')
        return result('restart', '采集服务已退出，按原配置从已有回合继续')
    if active_state != 'active' or not state.startswith('collect_batch_'):
        return result('wait', '服务正处于启动或清理阶段，等待自身超时处理')
    if not ports_ready:
        since = previous.get('missing_since')
        since = now.timestamp() if since is None else since
        if now.timestamp()-since >= 120:
            return result('restart', '采集时通信端口连续消失至少两分钟')
        return result('wait', '通信端口暂不可用，等待两分钟宽限期', missing_since=since)
    if now.timestamp()-last_activity > 900:
        return result('restart', '采集日志和进度连续十五分钟没有更新')
    return result('wait', '采集正常运行')


def owned_service(properties):
    expected = f'argv[]={PYTHON} -m tools.medication_sorting.run_randomized_experiment --output {OUTPUT} ;'
    group = properties.get('ControlGroup', '')
    group_ok = group.endswith('/'+SERVICE) or (
        not group and properties.get('ActiveState') in {'inactive', 'failed'})
    return bool(expected in properties.get('ExecStart', '')
                and properties.get('WorkingDirectory') == str(ROOT)
                and properties.get('KillMode') == 'control-group' and group_ok)


def systemctl(*arguments):
    return subprocess.run(['systemctl', '--user', *arguments], check=True,
                          text=True, capture_output=True, timeout=180)


def can_recreate(properties, status):
    return (properties.get('LoadState') == 'not-found'
            and status.get('output') == str(OUTPUT)
            and status.get('expert_profile') == 'randomized_tabletop_70'
            and status.get('randomization_profile') == 'tabletop_v1')


def restart_service(properties):
    if properties.get('LoadState') != 'not-found':
        return systemctl('restart', SERVICE)
    return subprocess.run([
        'systemd-run', '--user', '--unit='+SERVICE,
        '--description=四药品160条采集及pi05两万步训练评估',
        '--property=WorkingDirectory='+str(ROOT), '--property=KillMode=control-group',
        '--property=TimeoutStopSec=120', '--property=Restart=no',
        PYTHON, '-m', 'tools.medication_sorting.run_randomized_experiment', '--output', str(OUTPUT),
    ], check=True, text=True, capture_output=True, timeout=180)


def read_json(path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{path}不是JSON对象')
    return value


def latest_activity(output, status):
    paths = [output/'status.json', output/'raw/progress.json', output/'raw/review_status.json']
    log = status.get('current_log')
    if log:
        path = Path(log).resolve()
        if path.is_relative_to(output/'pipeline_logs'):
            paths.append(path)
    return max((path.stat().st_mtime for path in paths if path.is_file()), default=0)


def downstream_exists(output):
    paths = [output/'split', output/'evaluation',
             ROOT/'data/medication_sorting/lerobot/local/g1_four_medication_160_20261014_train',
             ROOT/'data/medication_sorting/checkpoints/pi05_g1_medication_sorting_lora/g1_four_medication_160_20261014_20k']
    return any(path.exists() for path in paths)


def check(output, until):
    now = datetime.now(timezone.utc)
    previous = read_json(output/'night_watchdog.json')
    status = read_json(output/'status.json')
    progress = read_json(output/'raw/progress.json')
    properties = dict(line.split('=', 1) for line in systemctl(
        'show', SERVICE, '-p', 'ExecStart', '-p', 'WorkingDirectory', '-p', 'KillMode',
        '-p', 'ControlGroup', '-p', 'ActiveState', '-p', 'LoadState').stdout.splitlines() if '=' in line)
    ports = {str(port): port_open(port) for port in (12345, 50051)}
    if until is not None and now >= until:
        decision = {'action': 'expired', 'reason': '夜间守护到期，主任务保持不变', 'missing_since': None}
    elif (completion := completion_decision(status, progress)) is not None:
        decision = completion
    elif not (owned_service(properties) or can_recreate(properties, status)):
        decision = {'action': 'blocked', 'reason': '服务命令或归属不匹配，不操作', 'missing_since': None}
    else:
        # 停止状态任一端口占用即拒绝接管；采集状态必须两端口均可用。
        ready = all(ports.values()) if properties['ActiveState'] == 'active' else any(ports.values())
        decision = decide_action(now=now, until=until, active_state=properties['ActiveState'],
                                 status=status, progress=progress, downstream_started=downstream_exists(output),
                                 ports_ready=ready, last_activity=latest_activity(output, status),
                                 previous=previous, free_bytes=shutil.disk_usage(ROOT).free)
    restarts = previous.get('restarts', 0)
    if decision['action'] == 'restart':
        restart_service(properties)
        restarts += 1
    elif decision['action'] in {'expired', 'finished', 'collection_finished'}:
        systemctl('stop', TIMER)
    record = {**decision, 'checked_at': now.isoformat(), 'until': until.isoformat() if until else None,
              'service': SERVICE, 'service_state': properties.get('ActiveState'),
              'pipeline_state': status.get('state'), 'ports': ports,
              'accepted': progress.get('accepted'), 'attempts': progress.get('attempts'),
              'restarts': restarts}
    temporary = output/'night_watchdog.json.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temporary.replace(output/'night_watchdog.json')
    if (record['action'], record['reason']) != (previous.get('action'), previous.get('reason')) or record['action'] == 'restart':
        with (output/'night_watchdog_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', type=datetime.fromisoformat, help='可选截止时间；默认守护至四种药品各40条')
    args = parser.parse_args()
    if args.until is not None and args.until.tzinfo is None:
        parser.error('--until必须携带时区')
    if not OUTPUT.is_dir():
        parser.error('正式数据目录不存在，拒绝新建不同实验')
    with (OUTPUT/'.night_watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(OUTPUT, args.until)


if __name__ == '__main__':
    main()
