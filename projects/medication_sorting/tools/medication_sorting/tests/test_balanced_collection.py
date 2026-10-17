"""批次计数与断点续跑测试；质量审核替身不是仿真实测。"""

import importlib
import json

import pytest


def runner():
    return importlib.import_module('tools.medication_sorting.collect_balanced')


def test_balanced_retries_rejections_without_counting_them(tmp_path, monkeypatch):
    module = runner()
    calls = []
    def attempt(target, path, seed):
        path.mkdir()
        calls.append((target, seed))
        (path/'test_result.json').write_text(json.dumps({'accepted': len(calls) != 1}))
    def review(path):
        return {'accepted': json.loads((path/'test_result.json').read_text())['accepted'], 'errors': []}
    monkeypatch.setattr(module, 'verify_episode', review)
    result = module.collect_balanced(tmp_path/'raw', attempt=attempt, targets=['a', 'b'],
                                     per_target=2, max_attempts=4, seed_start=100,
                                     configuration={'profile': 'test'})
    assert result['complete'] is True
    assert result['accepted'] == 4
    assert result['attempts'] == 5
    assert result['by_object']['a']['accepted'] == 2
    assert len({seed for _, seed in calls}) == 5
    assert list((tmp_path/'raw'/'a').glob('episode_*'))


def test_resume_does_not_repeat_seed_or_overwrite_and_rechecks_quality(tmp_path, monkeypatch):
    module = runner()
    calls = []
    def attempt(target, path, seed):
        path.mkdir()
        calls.append(seed)
    monkeypatch.setattr(module, 'verify_episode', lambda path: {'accepted': True, 'errors': []})
    first = module.collect_balanced(tmp_path/'raw', attempt=attempt, targets=['a'],
                                    per_target=2, max_attempts=4, seed_start=100,
                                    configuration={}, max_new_attempts=1)
    assert first['complete'] is False
    second = module.collect_balanced(tmp_path/'raw', attempt=attempt, targets=['a'],
                                     per_target=2, max_attempts=4, seed_start=100,
                                     configuration={})
    assert second['complete'] is True
    assert calls == [100, 101]
    with pytest.raises(ValueError, match='配置'):
        module.collect_balanced(tmp_path/'raw', attempt=attempt, targets=['a'],
                                per_target=2, max_attempts=4, seed_start=100,
                                configuration={'different': True})


def test_attempt_cap_reports_missing_quota(tmp_path, monkeypatch):
    module = runner()
    def attempt(target, path, seed):
        path.mkdir()
        raise RuntimeError('synthetic failed grasp')
    monkeypatch.setattr(module, 'verify_episode', lambda path: {'accepted': False, 'errors': ['test']})
    result = module.collect_balanced(tmp_path/'raw', attempt=attempt, targets=['a'],
                                     per_target=2, max_attempts=3, seed_start=100,
                                     configuration={})
    assert result['complete'] is False
    assert result['accepted'] == 0
    assert result['attempts'] == 3


def test_history_review_publishes_progress_before_verifying_each_episode(tmp_path, monkeypatch):
    module = runner()
    raw = tmp_path/'raw'
    monkeypatch.setattr(module, 'verify_episode', lambda path: {'accepted': True, 'errors': []})
    arguments = dict(targets=['a'], per_target=2, max_attempts=4, seed_start=100, configuration={})
    module.collect_balanced(raw, attempt=lambda target, path, seed: path.mkdir(), **arguments)
    seen = []
    def review(path):
        status = json.loads((raw/'review_status.json').read_text())
        seen.append((status['state'], status['reviewed_attempts']))
        return {'accepted': True, 'errors': []}
    monkeypatch.setattr(module, 'verify_episode', review)
    module.collect_balanced(raw, attempt=lambda *args: pytest.fail('配额已满，不得重复采集'), **arguments)
    assert seen == [('reviewing_history', 0), ('reviewing_history', 1)]
    final = json.loads((raw/'review_status.json').read_text())
    assert final['state'] == 'history_review_complete'
    assert final['reviewed_attempts'] == 2
    assert final['last_episode'] == str((raw/'a/episode_0001').resolve())
    assert final['updated_at']


def test_resume_quality_cannot_override_wrong_experiment_metadata(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, 'verify_episode', lambda path: {'accepted': True, 'errors': []})
    (tmp_path/'metadata.json').write_text(json.dumps({
        'seed': 99, 'object_name': 'a', 'expert_profile': 'wrong',
        'randomization_profile': 'tabletop_v1', 'experiment_layout_sha256': 'abc',
        'fps': 5, 'randomized_reset': True}))
    quality = module.review_attempt(tmp_path, target='a', seed=100,
                                    configuration={'expert_profile': 'expected',
                                      'randomization_profile': 'tabletop_v1',
                                      'layout_sha256': 'abc', 'control_hz': 5})
    assert quality['accepted'] is False


def test_resume_rejects_changed_numeric_distribution_with_same_profile(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, 'verify_episode', lambda path: {'accepted': True, 'errors': []})
    config = {'expert_profile': 'test', 'randomization_profile': 'tabletop_v1',
              'randomization_config': {'xy_half_range_m': [.02, .02]},
              'layout_sha256': 'abc', 'control_hz': 5}
    (tmp_path/'metadata.json').write_text(json.dumps({
        'seed': 100, 'object_name': 'a', 'expert_profile': 'test',
        'randomization_profile': 'tabletop_v1', 'randomization_config': {'xy_half_range_m': [.001, .001]},
        'experiment_layout_sha256': 'abc', 'fps': 5, 'randomized_reset': True}))
    assert not module.review_attempt(tmp_path, target='a', seed=100, configuration=config)['accepted']
