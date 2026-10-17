"""动作段对照的离线测试；不连接真实模型或仿真。"""

import copy
import importlib
import json
from pathlib import Path

import pytest

from tools.medication_sorting.task_specs import TARGETS


def module():
    return importlib.import_module('tools.medication_sorting.compare_action_chunks')


def test_specs_pair_all_three_chunks_on_same_forty_seeds():
    specs = module().comparison_specs()
    assert len(specs) == 12
    sets = []
    for chunk in (1, 2, 4):
        rows = [row for row in specs if row['execute_chunk'] == chunk]
        assert [row['target'] for row in rows] == list(TARGETS)
        seeds = [(row['target'], seed) for row in rows
                 for seed in range(row['seed_start'], row['seed_start']+row['episodes'])]
        assert len(seeds) == len(set(seeds)) == 40
        sets.append(seeds)
    assert sets[0] == sets[1] == sets[2]


def test_commands_only_change_chunk_target_seed_and_output(tmp_path):
    mod = module()
    for spec in mod.comparison_specs():
        command = mod.evaluation_command(spec, tmp_path)
        assert command[:3] == [mod.PYTHON, '-m', 'tools.medication_sorting.evaluate_policy']
        assert command[command.index('--checkpoint-reference')+1] == str(mod.CHECKPOINT)
        assert command[command.index('--control-hz')+1] == '5'
        assert command[command.index('--steps')+1] == '150'
        assert command[command.index('--episodes')+1] == '10'
        assert command[command.index('--execute-chunk')+1] == str(spec['execute_chunk'])
        assert command[command.index('--training-manifest')+1] == str(mod.SOURCE/'split/split_manifest.json')
        assert 'train' not in command and 'collect_balanced' not in ' '.join(command)


def reports(tmp_path, spec, *, successes=3):
    mod = module()
    provenance = {'path': str(mod.CHECKPOINT), 'commit_timestamp_nsecs': 123,
                  'params_manifest_sha256': 'params', 'params_metadata_sha256': 'meta',
                  'norm_stats_sha256': 'norm'}
    config = {'checkpoint_provenance': provenance, 'layout_sha256': 'layout'}
    summary = {'target': spec['target'], 'task_count': 10, 'success_count': successes,
               'controls': {'execute_chunk': spec['execute_chunk'], 'steps': 150,
                            'control_hz': 5, 'camera_backend': 'viewport'}, 'episodes': []}
    for i in range(10):
        folder = tmp_path/f'seed_{spec["seed_start"]+i}'
        folder.mkdir(parents=True)
        report = {'success': i < successes, 'grasp_verified': i < 6,
                  'controller': 'pi05_checkpoint_policy', 'expert_actions_used': False,
                  'object_name': spec['target'], 'expert_profile': mod.EXPERT_PROFILE,
                  'randomization_profile': mod.RANDOMIZATION, 'seed': spec['seed_start']+i,
                  'experiment_layout_sha256': 'layout',
                  'policy_metadata': {'checkpoint_provenance': provenance}}
        path = folder/'report.json'
        path.write_text(json.dumps(report))
        summary['episodes'].append({'seed': spec['seed_start']+i, 'success': i < successes,
                                    'grasp_verified': i < 6, 'report_path': str(path),
                                    'target_matches': True, 'checkpoint_path_matches': True,
                                    'reported_reset_matches_seed': True, 'returncode': 0 if i < successes else 1})
    return summary, config


def test_block_accepts_physical_failures_in_denominator(tmp_path):
    spec = module().comparison_specs()[0]
    summary, config = reports(tmp_path, spec)
    module().require_block(summary, spec, config)
    assert summary['task_count'] == 10 and summary['success_count'] == 3


def test_block_matches_evaluator_summary_without_top_level_target(tmp_path):
    spec = module().comparison_specs()[0]
    summary, config = reports(tmp_path, spec)
    summary.pop('target')
    module().require_block(summary, spec, config)


@pytest.mark.parametrize('returncode', [0, 1])
def test_raw_success_without_grasp_stays_a_failed_attempt(tmp_path, returncode):
    mod = module()
    spec = mod.comparison_specs()[0]
    summary, config = reports(tmp_path, spec)
    row = summary['episodes'][7]
    path = Path(row['report_path'])
    report = json.loads(path.read_text())
    report['success'] = True
    path.write_text(json.dumps(report))
    row['returncode'] = returncode
    mod.require_block(summary, spec, config)
    assert row['success'] is False and summary['success_count'] == 3


def test_configuration_fingerprints_randomization_dependency(tmp_path, monkeypatch):
    mod = module()
    source, checkpoint = tmp_path/'source', tmp_path/'checkpoint'
    (source/'raw').mkdir(parents=True)
    (source/'raw/progress.json').write_text(json.dumps({
        'complete': True, 'accepted': 160,
        'by_object': {target: {'accepted': 40} for target in TARGETS}}))
    (source/'status.json').write_text(json.dumps({'training_completed': True}))
    (checkpoint/'assets/test').mkdir(parents=True)
    (checkpoint/'assets/test/norm_stats.json').write_text('{}')
    (checkpoint/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs': 123}))
    monkeypatch.setattr(mod, 'SOURCE', source)
    monkeypatch.setattr(mod, 'CHECKPOINT', checkpoint)
    monkeypatch.setattr(mod, 'sha256', lambda path: 'hash')
    config = mod.configuration()
    assert 'randomization.py' in config['implementation_sha256']


@pytest.mark.parametrize('mutation', ['missing_seed', 'duplicate_seed', 'wrong_chunk', 'wrong_checkpoint', 'wrong_layout', 'expert_used'])
def test_incomplete_or_mismatched_block_cannot_finish(tmp_path, mutation):
    mod = module()
    spec = mod.comparison_specs()[0]
    summary, config = reports(tmp_path, spec)
    path = Path(summary['episodes'][0]['report_path'])
    report = json.loads(path.read_text())
    if mutation == 'missing_seed':
        summary['episodes'].pop()
    elif mutation == 'duplicate_seed':
        summary['episodes'][1]['seed'] = summary['episodes'][0]['seed']
    elif mutation == 'wrong_chunk':
        summary['controls']['execute_chunk'] = 8
    elif mutation == 'wrong_checkpoint':
        report['policy_metadata']['checkpoint_provenance']['params_manifest_sha256'] = 'other'
    elif mutation == 'wrong_layout':
        report['experiment_layout_sha256'] = 'other'
    else:
        report['expert_actions_used'] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        mod.require_block(summary, spec, config)


def test_summary_requires_all_blocks_and_keeps_failures(tmp_path):
    mod = module()
    blocks = []
    for i, spec in enumerate(mod.comparison_specs()):
        summary, _ = reports(tmp_path/str(i), spec)
        blocks.append({'spec': spec, 'summary': summary})
    result = mod.summarize_blocks(blocks)
    assert result['complete'] is True
    assert result['total_tasks'] == 120
    assert result['selected_chunk'] == 4
    assert result['independent_test_completed'] is False
    assert all(row['tasks'] == 40 and row['successes'] == 12 for row in result['by_chunk'])
    assert all(row['no_grasp'] == 16 and row['post_grasp_failure'] == 12 for row in result['by_chunk'])
    with pytest.raises(ValueError):
        mod.summarize_blocks(blocks[:-1])
    duplicate = copy.deepcopy(blocks)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError):
        mod.summarize_blocks(duplicate)


def test_runner_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        module().ChunkComparison(tmp_path)


def test_runner_stops_at_foreign_port_before_starting_process(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'configuration', lambda: {'test': True})
    experiment = mod.ChunkComparison(tmp_path/'new')
    monkeypatch.setattr(mod, 'port_open', lambda _: True)
    monkeypatch.setattr(mod, 'active_controllers', lambda: [])
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('不得启动或接管进程'))
    with pytest.raises(RuntimeError, match='端口'):
        experiment.execute()


@pytest.mark.parametrize('change_final', [False, True])
def test_runner_runs_all_blocks_restarts_policy_per_chunk_and_never_trains(tmp_path, monkeypatch, change_final):
    mod = module()
    _, config = reports(tmp_path/'fixture', mod.comparison_specs()[0])
    state = {'changed': False}
    monkeypatch.setattr(mod, 'configuration', lambda: {**config, 'changed': True} if state['changed'] else config)
    monkeypatch.setattr(mod, 'port_open', lambda _: False)
    monkeypatch.setattr(mod, 'active_controllers', lambda: [])
    experiment = mod.ChunkComparison(tmp_path/'run')
    policy_starts, calls = [], []
    monkeypatch.setattr(experiment, 'start_policy', policy_starts.append)
    monkeypatch.setattr(experiment, 'stop_policy', lambda: calls.append('stop_policy'))
    monkeypatch.setattr(experiment, 'start_app_with_retry', lambda: calls.append('start_app'))
    monkeypatch.setattr(experiment, 'stop_app', lambda: calls.append('stop_app'))
    def run(command, name, **kwargs):
        assert command[2] == 'tools.medication_sorting.evaluate_policy'
        spec = mod.comparison_specs()[calls.count('evaluate')]
        output = Path(command[command.index('--output')+1])
        summary, _ = reports(output, spec)
        summary.pop('target')
        (output/'summary.json').write_text(json.dumps(summary))
        calls.append('evaluate')
        if change_final and calls.count('evaluate') == 12:
            state['changed'] = True
    monkeypatch.setattr(experiment, 'run', run)
    if change_final:
        with pytest.raises(ValueError, match='变化'):
            experiment.execute()
        assert not (experiment.output/'comparison.json').exists()
        assert experiment.status['state'] == 'stopped_with_error'
        return
    experiment.execute()
    assert policy_starts == [1, 2, 4]
    assert calls.count('stop_policy') == 3
    assert calls.count('evaluate') == calls.count('start_app') == 12
    result = json.loads((experiment.output/'comparison.json').read_text())
    assert result['total_tasks'] == 120
    assert experiment.status['state'] == 'complete'


def test_failed_rollout_still_cleans_owned_processes_and_preserves_error(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'configuration', lambda: {'test': True})
    monkeypatch.setattr(mod, 'port_open', lambda _: False)
    monkeypatch.setattr(mod, 'active_controllers', lambda: [])
    experiment = mod.ChunkComparison(tmp_path/'run')
    calls = []
    monkeypatch.setattr(experiment, 'start_policy', lambda chunk: calls.append('start_policy'))
    monkeypatch.setattr(experiment, 'stop_policy', lambda: calls.append('stop_policy'))
    monkeypatch.setattr(experiment, 'start_app_with_retry', lambda: calls.append('start_app'))
    monkeypatch.setattr(experiment, 'stop_app', lambda: calls.append('stop_app'))
    def run(*args, **kwargs):
        raise RuntimeError('test rollout failure')
    monkeypatch.setattr(experiment, 'run', run)
    with pytest.raises(RuntimeError, match='rollout failure'):
        experiment.execute()
    assert 'stop_app' in calls and 'stop_policy' in calls
    assert experiment.status['state'] == 'stopped_with_error'
    assert not (experiment.output/'comparison.json').exists()
