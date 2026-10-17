"""长实验的阶段门禁离线测试，不加载模型或连接仿真。"""

import importlib
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time

import pytest

from tools.medication_sorting.task_specs import TARGETS


def module():
    return importlib.import_module('tools.medication_sorting.run_randomized_experiment')


def test_training_requires_exactly_forty_accepted_per_object():
    report = {'complete': True, 'accepted': 160,
              'by_object': {target: {'accepted': 40} for target in TARGETS}}
    module().require_complete_collection(report)
    report['by_object'][next(iter(TARGETS))]['accepted'] = 39
    with pytest.raises(ValueError):
        module().require_complete_collection(report)


def test_training_command_is_20k_from_official_base_and_new_experiment():
    command = module().training_command('local/new_train', 'new_160_trial')
    assert '--num-train-steps=20000' in command
    assert '--exp-name=new_160_trial' in command
    assert '--resume' not in command
    assert any('pi05_base/params' in part for part in command)
    assert '--save-interval=2000' in command


def test_evaluation_has_ten_new_seeds_per_target_without_overlap():
    specs = module().evaluation_specs()
    assert len(specs) == 4
    assert {row['target'] for row in specs} == set(TARGETS)
    seeds = [seed for row in specs for seed in range(row['seed_start'], row['seed_start']+row['episodes'])]
    assert len(seeds) == len(set(seeds)) == 40
    assert all(seed > 3000000000 for seed in seeds)


def test_evaluation_completion_requires_all_ten_exact_seeds():
    spec = module().evaluation_specs()[0]
    incomplete = {'task_count': 9, 'episodes': [{'seed': spec['seed_start']+i} for i in range(9)]}
    with pytest.raises(ValueError):
        module().require_complete_evaluation(incomplete, spec)


def test_conversion_requires_all_expected_sources_and_no_rejections(tmp_path):
    expected = [{'source': str(tmp_path/f'episode_{i:04d}')} for i in range(128)]
    report = {'repo_id': 'local/test', 'fps': 5, 'converted_episodes': 127,
              'accepted': [{'path': row['source']} for row in expected[:-1]], 'rejected': [{'reason': 'bad'}]}
    (tmp_path/'g1_conversion_manifest.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module().require_complete_conversion(tmp_path, expected, 'local/test')


def test_backup_failure_still_stops_owned_app(tmp_path, monkeypatch):
    experiment = module().Experiment(tmp_path)
    calls = []
    class Process:
        returncode = None
        def poll(self):return self.returncode
    process = Process()
    experiment.app = process
    monkeypatch.setattr(module(), 'port_open', lambda port: process.returncode is None)
    def fail(*a, **k):raise RuntimeError('backup failed')
    monkeypatch.setattr(experiment, 'run', fail)
    def stop(child):
        calls.append(child)
        child.returncode = -15
    monkeypatch.setattr(experiment, 'stop_owned', stop)
    with pytest.raises(RuntimeError, match='backup failed'):
        experiment.stop_app()
    assert calls == [process]
    assert experiment.app is None


@pytest.mark.parametrize('backup_timeout', [False, True])
def test_backup_failure_still_checks_port_release(tmp_path, monkeypatch, backup_timeout):
    experiment = module().Experiment(tmp_path)
    class Process:
        returncode = None
    process = Process()
    experiment.app = process
    checked = []
    def fail():
        if backup_timeout:
            raise subprocess.TimeoutExpired('backup', 120)
        raise RuntimeError('backup failed')
    monkeypatch.setattr(experiment, 'backup_stop_scene', fail)
    monkeypatch.setattr(experiment, 'stop_owned', lambda child: setattr(child, 'returncode', -15))
    monkeypatch.setattr(module(), 'port_open', lambda port: checked.append(port) or False)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        experiment.stop_app()
    assert checked == [12345, 50051]
    assert experiment.app is None


@pytest.mark.parametrize('launcher_already_exited', [False, True])
def test_stop_owned_closes_child_listener_when_launcher_exits_first(launcher_already_exited):
    child_code = (
        'import json,os,signal,socket,time\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        'sock=socket.socket(); sock.bind(("127.0.0.1",0)); sock.listen()\n'
        'print(json.dumps({"pid":os.getpid(),"port":sock.getsockname()[1]}),flush=True)\n'
        'time.sleep(120)\n'
    )
    launcher_code = f'import subprocess,sys; child=subprocess.Popen([sys.executable,"-c",{child_code!r}])'
    if not launcher_already_exited:
        launcher_code += '; child.wait()'
    launcher = subprocess.Popen([sys.executable, '-c', launcher_code], start_new_session=True,
                                stdout=subprocess.PIPE, text=True)
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
    try:
        assert select.select([launcher.stdout], [], [], 5)[0], '测试监听进程未及时就绪'
        info = json.loads(launcher.stdout.readline())
        assert os.getpgid(info['pid']) == launcher.pid
        if launcher_already_exited:
            assert launcher.wait(timeout=5) == 0
        module().Experiment.stop_owned(launcher)
        deadline = time.monotonic()+2
        while True:
            try:
                with socket.create_connection(('127.0.0.1', info['port']), timeout=.2):
                    pass
            except OSError:
                break
            if time.monotonic() >= deadline:
                pytest.fail('启动器退出后，同组忽略SIGTERM的子进程仍占用监听端口')
            time.sleep(.02)
        assert foreign.poll() is None, '不得停止其他进程组'
    finally:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        launcher.wait(timeout=5)
        launcher.stdout.close()
        foreign.terminate()
        foreign.wait(timeout=5)


def test_stop_owned_refuses_a_reused_launcher_pid():
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
    class ReapedProcess:
        pid = foreign.pid
        def poll(self):return 0
    try:
        with pytest.raises(RuntimeError, match='PID.*复用'):
            module().Experiment.stop_owned(ReapedProcess())
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_restart_continues_log_numbers_and_preserves_previous_status(tmp_path):
    logs = tmp_path/'pipeline_logs'
    logs.mkdir()
    (logs/'0015-backup_stop_scene.log').write_text('original failure')
    saved = {'state': 'stopped_with_error', 'error': 'timeout'}
    (tmp_path/'status.json').write_text(json.dumps(saved))
    experiment = module().Experiment(tmp_path)
    assert experiment.counter >= 15
    histories = list((tmp_path/'status_history').glob('*.json'))
    assert len(histories) == 1
    assert json.loads(histories[0].read_text()) == saved


@pytest.mark.parametrize('start_timeout', [False, True])
@pytest.mark.parametrize('cleanup_timeout', [False, True])
def test_scene_start_retry_cleans_owned_app_before_retry(tmp_path, monkeypatch, start_timeout, cleanup_timeout):
    experiment = module().Experiment(tmp_path)
    helper = getattr(experiment, 'start_app_with_retry', None)
    assert callable(helper), '场景启动超时应有限重试本任务自己的窗口'
    calls = []
    def start():
        calls.append('start')
        experiment.app = object()
        if calls.count('start') == 1:
            if start_timeout:
                raise subprocess.TimeoutExpired('load_scene', 300)
            raise RuntimeError('load_scene退出码1')
    def stop():
        calls.append('stop')
        experiment.app = None
        if cleanup_timeout:
            raise subprocess.TimeoutExpired('backup_stop_scene', 120)
        raise RuntimeError('backup timeout; owned app already stopped')
    monkeypatch.setattr(experiment, 'start_app', start)
    monkeypatch.setattr(experiment, 'stop_app', stop)
    helper()
    assert calls == ['start', 'stop', 'start']


def test_scene_start_retry_never_controls_foreign_window(tmp_path, monkeypatch):
    experiment = module().Experiment(tmp_path)
    helper = getattr(experiment, 'start_app_with_retry', None)
    assert callable(helper)
    calls = []
    def start():
        calls.append('start')
        raise RuntimeError('已有OrcaLab占用端口')
    monkeypatch.setattr(experiment, 'start_app', start)
    monkeypatch.setattr(experiment, 'stop_app', lambda: calls.append('stop'))
    with pytest.raises(RuntimeError, match='占用端口'):
        helper()
    assert calls == ['start']


def test_scene_start_retry_stops_after_three_failures(tmp_path, monkeypatch):
    experiment = module().Experiment(tmp_path)
    helper = getattr(experiment, 'start_app_with_retry', None)
    assert callable(helper)
    calls = []
    def start():
        calls.append('start')
        experiment.app = object()
        raise RuntimeError('load_scene退出码1')
    def stop():
        calls.append('stop')
        experiment.app = None
    monkeypatch.setattr(experiment, 'start_app', start)
    monkeypatch.setattr(experiment, 'stop_app', stop)
    with pytest.raises(RuntimeError, match='三次'):
        helper()
    assert calls == ['start', 'stop'] * 3
