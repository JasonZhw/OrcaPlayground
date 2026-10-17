"""多目标评估驱动离线测试；假服务不代表模型成功率。"""

import json
from pathlib import Path
import subprocess

import numpy as np

from tools.medication_sorting import evaluate_policy as module
from tools.medication_sorting.randomization import sample_reset


def test_expected_reset_supports_shared_tabletop_distribution():
    offset, yaw = sample_reset(42, profile='tabletop_v1')
    expected = module._expected_reset(42, profile='tabletop_v1')
    np.testing.assert_array_equal(expected['offset'], offset)
    assert expected['yaw'] == yaw


def test_evaluator_forwards_target_and_rejects_wrong_task_report(tmp_path, monkeypatch):
    target = 'melatonin_tablet_bottle_1'
    monkeypatch.setattr(module, 'EXECUTION_LOCK', tmp_path/'lock')
    monkeypatch.setattr(module, 'active_controllers', lambda: [])
    def run(command, **kwargs):
        for name, value in [('--target', target), ('--profile', 'randomized_tabletop_70'),
                            ('--randomization-profile', 'tabletop_v1'), ('--layout', str(tmp_path/'layout.json'))]:
            assert command[command.index(name)+1] == value
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        (output/'report.json').write_text(json.dumps({
            'success': True, 'grasp_verified': True, 'expert_actions_used': False,
            'controller': 'pi05_checkpoint_policy', 'object_name': 'wrong_object',
            'expert_profile': 'randomized_tabletop_70', 'randomization_profile': 'tabletop_v1'}))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    result = module.evaluate_policy(tmp_path/'evaluation', checkpoint_reference='/new/checkpoint', episodes=1,
                                    target=target, profile='randomized_tabletop_70',
                                    layout=tmp_path/'layout.json', randomization_profile='tabletop_v1')
    assert result['success_count'] == 0
    assert result['failed_count'] == 1


def test_missing_server_metadata_counts_as_failure_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'EXECUTION_LOCK', tmp_path/'lock')
    monkeypatch.setattr(module, 'active_controllers', lambda: [])
    def run(command, **kwargs):
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        (output/'report.json').write_text(json.dumps({'success': False, 'policy_metadata': None,
            'controller': 'pi05_checkpoint_policy', 'expert_actions_used': False,
            'error': 'synthetic connection failure'}))
        return subprocess.CompletedProcess(command, 1, '', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    report = module.evaluate_policy(tmp_path/'evaluation', checkpoint_reference='/checkpoint', episodes=2,
                                   target='melatonin_tablet_bottle_1')
    assert report['task_count'] == report['failed_count'] == 2
