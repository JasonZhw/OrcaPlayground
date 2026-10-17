"""夜间守护的离线决策测试，不连接或停止真实仿真。"""

from datetime import datetime, timedelta, timezone
import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from tools.medication_sorting.task_specs import TARGETS


NOW = datetime(2026, 10, 14, 14, 20, tzinfo=timezone.utc)
UNTIL = datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc)


def module():
    return importlib.import_module('tools.medication_sorting.night_watchdog')


def decide(**changes):
    arguments = dict(now=NOW, until=UNTIL, active_state='active',
                     status={'state': 'collect_batch_002'}, progress={},
                     downstream_started=False, ports_ready=True,
                     last_activity=NOW.timestamp(), previous={}, free_bytes=150*1024**3)
    arguments.update(changes)
    return module().decide_action(**arguments)


def test_active_collection_keeps_running():
    assert decide()['action'] == 'wait'


def test_watchdog_can_run_without_a_clock_expiry():
    assert decide(until=None)['action'] == 'wait'


def completed_progress():
    return {'complete': True, 'accepted': 160,
            'by_object': {target: {'accepted': 40} for target in TARGETS}}


def test_completed_collection_ends_guard_without_restarting_training():
    assert decide(until=None, active_state='failed', ports_ready=False,
                  progress=completed_progress())['action'] == 'collection_finished'


def test_invalid_complete_flag_does_not_end_guard_or_restart():
    progress = {**completed_progress(), 'accepted': 159}
    assert decide(progress=progress)['action'] == 'blocked'


def test_review_heartbeat_prevents_false_stall_restart(tmp_path):
    raw = tmp_path/'raw'
    raw.mkdir()
    for path in (tmp_path/'status.json', raw/'progress.json'):
        path.write_text('{}')
        os.utime(path, (NOW.timestamp()-1800, NOW.timestamp()-1800))
    heartbeat = raw/'review_status.json'
    heartbeat.write_text('{}')
    os.utime(heartbeat, (NOW.timestamp(), NOW.timestamp()))
    latest = module().latest_activity(tmp_path, {})
    assert latest == NOW.timestamp()
    assert decide(last_activity=latest)['action'] == 'wait'


def test_cli_defaults_to_guarding_until_collection_finishes(tmp_path, monkeypatch):
    mod = module()
    calls = []
    monkeypatch.setattr(mod, 'OUTPUT', tmp_path)
    monkeypatch.setattr(mod, 'check', lambda output, until: calls.append((output, until)))
    monkeypatch.setattr(sys, 'argv', ['watchdog'])
    mod.main()
    assert calls == [(tmp_path, None)]


def test_completed_collection_stops_timer_with_no_clock_expiry(tmp_path, monkeypatch):
    mod = module()
    (tmp_path/'status.json').write_text(json.dumps({'state': 'collection_batch_reviewed'}))
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/progress.json').write_text(json.dumps(completed_progress()))
    calls = []
    def control(*arguments):
        calls.append(arguments)
        return SimpleNamespace(stdout='LoadState=loaded\nActiveState=active\n')
    monkeypatch.setattr(mod, 'systemctl', control)
    monkeypatch.setattr(mod, 'owned_service', lambda _: True)
    monkeypatch.setattr(mod, 'port_open', lambda _: True)
    monkeypatch.setattr(mod, 'downstream_exists', lambda _: False)
    record = mod.check(tmp_path, None)
    assert record['action'] == 'collection_finished'
    assert record['until'] is None
    assert ('stop', mod.TIMER) in calls
    assert not any('restart' in call for call in calls)


def test_completed_collection_stops_only_own_timer_with_foreign_main_service(tmp_path, monkeypatch):
    mod = module()
    (tmp_path/'status.json').write_text(json.dumps({'state': 'collection_batch_reviewed'}))
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/progress.json').write_text(json.dumps(completed_progress()))
    calls = []
    def control(*arguments):
        calls.append(arguments)
        return SimpleNamespace(stdout='LoadState=loaded\nActiveState=active\nExecStart=unrelated\n')
    monkeypatch.setattr(mod, 'systemctl', control)
    monkeypatch.setattr(mod, 'port_open', lambda _: True)
    record = mod.check(tmp_path, None)
    assert record['action'] == 'collection_finished'
    assert ('stop', mod.TIMER) in calls
    assert not any(call[0] in {'restart', 'stop'} and mod.SERVICE in call for call in calls)


@pytest.mark.parametrize('progress', [{}, {**completed_progress(), 'accepted': 159}])
def test_finished_experiment_still_requires_valid_collection(progress):
    assert decide(status={'state': 'complete', 'inference_completed': True},
                  progress=progress)['action'] == 'blocked'


def test_failed_collection_restarts_without_erasing_data():
    result = decide(active_state='failed', ports_ready=False,
                    status={'state': 'backup_stop_scene', 'error': 'timeout'})
    assert result['action'] == 'restart'


@pytest.mark.parametrize('state', ['train_20000', 'convert_train', 'split', 'training_completed',
                                   'evaluate_target', 'complete'])
def test_later_stage_never_restarts_whole_pipeline(state):
    assert decide(active_state='failed', status={'state': state})['action'] != 'restart'


def test_no_window_during_training_is_normal():
    assert decide(status={'state': 'train_20000'}, ports_ready=False)['action'] == 'wait'


def test_existing_downstream_artifacts_block_collection_replay():
    assert decide(active_state='failed', downstream_started=True)['action'] == 'blocked'


def test_missing_ports_must_persist_for_two_minutes():
    first = decide(ports_ready=False)
    assert first['action'] == 'wait'
    assert first['missing_since'] == NOW.timestamp()
    assert decide(now=NOW+timedelta(seconds=119), ports_ready=False,
                  previous=first)['action'] == 'wait'
    assert decide(now=NOW+timedelta(seconds=120), ports_ready=False,
                  previous=first)['action'] == 'restart'


def test_reappearing_ports_reset_missing_grace_period():
    result = decide(previous={'missing_since': NOW.timestamp()-300})
    assert result['action'] == 'wait'
    assert result.get('missing_since') is None


def test_stalled_collector_restarts_but_slow_progress_does_not():
    assert decide(last_activity=NOW.timestamp()-901)['action'] == 'restart'
    assert decide(last_activity=NOW.timestamp()-899)['action'] == 'wait'


def test_scene_startup_has_its_own_timeout_not_collector_watchdog():
    assert decide(status={'state': 'load_scene'}, ports_ready=False,
                  last_activity=NOW.timestamp()-180)['action'] == 'wait'


def test_expiry_only_ends_watchdog():
    result = decide(now=UNTIL)
    assert result['action'] == 'expired'


def test_failed_service_with_open_external_ports_is_not_taken_over():
    assert decide(active_state='failed', ports_ready=True)['action'] == 'blocked'


def test_failed_service_restarts_once_ports_are_released():
    assert decide(active_state='failed', ports_ready=False)['action'] == 'restart'


def test_low_disk_does_not_loop_restart():
    assert decide(active_state='failed', ports_ready=False,
                  free_bytes=99*1024**3)['action'] == 'blocked'


def test_attempt_exhaustion_does_not_loop_restart():
    progress = {'by_object': {'target': {'attempts': 200, 'accepted': 39}}}
    assert decide(active_state='failed', ports_ready=False, progress=progress)['action'] == 'blocked'


def test_inconsistent_collection_configuration_is_not_retried():
    assert decide(active_state='failed', ports_ready=False,
                  status={'state': 'collect_batch_000', 'error': '断点续跑配置与已有批次不一致'})['action'] == 'blocked'


def test_completed_experiment_ends_watchdog_without_restart():
    assert decide(status={'state': 'complete', 'inference_completed': True},
                  progress=completed_progress())['action'] == 'finished'


def test_service_ownership_requires_exact_executable_workdir_and_cgroup():
    root = str(module().ROOT)
    props = {'ExecStart': f'{{ path={root}/.venv/bin/python ; argv[]={root}/.venv/bin/python -m tools.medication_sorting.run_randomized_experiment --output {root}/data/medication_sorting/multi_object/randomized160-20261014/production ; }}',
             'WorkingDirectory': root, 'KillMode': 'control-group',
             'ControlGroup': '/user.slice/app.slice/sy-medication-160-20k.service'}
    assert module().owned_service(props)
    for key, value in [('WorkingDirectory', '/tmp'), ('ExecStart', 'python unrelated.py'),
                       ('KillMode', 'process'), ('ControlGroup', '/unrelated.service')]:
        assert not module().owned_service({**props, key: value})


def test_missing_transient_service_can_be_recreated_only_for_saved_experiment():
    mod = module()
    status = {'output': str(mod.OUTPUT), 'expert_profile': 'randomized_tabletop_70',
              'randomization_profile': 'tabletop_v1', 'state': 'collect_batch_002'}
    props = {'LoadState': 'not-found', 'ActiveState': 'inactive', 'ExecStart': ''}
    assert mod.can_recreate(props, status)
    assert not mod.can_recreate({**props, 'LoadState': 'loaded'}, status)
    assert not mod.can_recreate(props, {**status, 'output': '/tmp/other'})
    assert not mod.can_recreate(props, {})


def test_recreate_uses_same_command_and_cgroup_without_automatic_training_restart(monkeypatch):
    mod = module()
    commands = []
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    mod.restart_service({'LoadState': 'not-found'})
    command = commands[0]
    assert command[:2] == ['systemd-run', '--user']
    assert '--property=KillMode=control-group' in command
    assert '--property=Restart=no' in command
    assert command[-5:] == [mod.PYTHON, '-m', 'tools.medication_sorting.run_randomized_experiment',
                             '--output', str(mod.OUTPUT)]


def test_expiry_stops_timer_even_if_main_unit_definition_disappeared(tmp_path, monkeypatch):
    mod = module()
    (tmp_path/'status.json').write_text(json.dumps({'state': 'starting'}))
    calls = []
    def control(*arguments):
        calls.append(arguments)
        return SimpleNamespace(stdout='LoadState=not-found\nActiveState=inactive\n')
    monkeypatch.setattr(mod, 'systemctl', control)
    monkeypatch.setattr(mod, 'port_open', lambda _: False)
    record = mod.check(tmp_path, NOW-timedelta(days=1))
    assert record['action'] == 'expired'
    assert ('stop', mod.TIMER) in calls
