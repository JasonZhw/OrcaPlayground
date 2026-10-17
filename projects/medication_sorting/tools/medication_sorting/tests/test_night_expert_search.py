from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import json
import pytest


def module():
    name = 'tools.medication_sorting.night_expert_search'
    assert importlib.util.find_spec(name) is not None, '夜间任务需要可恢复、有截止时间的后台入口'
    return importlib.import_module(name)


def test_candidates_keep_physics_size_and_acceptance_out_of_the_search_space():
    candidates = module().candidates()
    assert len(candidates) == 25
    assert len({row['name'] for row in candidates}) == 25
    assert candidates[0]['name'] == 'baseline'
    allowed = {'park_left_arm', 'herb_y_correction', 'upright_herb',
               'melatonin_extra_tilt', 'herb_extra_tilt'}
    assert all(set(row['parameters']) <= allowed for row in candidates)


def test_ranking_prioritizes_complete_rounds_and_keeps_failure_denominators():
    score = module().score
    partial = {'success': False, 'objects': [{'success': True}]*3}
    complete = {'success': True, 'objects': [{'success': True}]*4}
    assert score([complete, partial, {'success': False}]) > score([partial]*3)
    assert score([partial]*3) > score([{'success': False}]*3)


def test_interrupted_round_is_kept_as_failure_without_overwriting_any_trace(tmp_path):
    folder = tmp_path/'round_0000'; folder.mkdir()
    trace = folder/'continuous_trace.jsonl'; trace.write_text('original evidence\n')
    row = module().existing_round(folder)
    assert row['success'] is False and row['interrupted'] is True
    assert trace.read_text() == 'original evidence\n'
    (folder/'report.json').write_text(json.dumps({'success': False, 'error': '物理失败'}))
    assert module().existing_round(folder)['error'] == '物理失败'


def test_watchdog_only_recovers_owned_stalled_or_failed_jobs_and_has_a_cap():
    m = module()
    now = datetime(2026, 10, 16, 15, tzinfo=timezone.utc)
    kw = dict(now=now, until=now+timedelta(hours=10), state='running', active='active',
              owned=True, ports_occupied=False, age_seconds=1, restarts=0)
    assert m.watch_decision(**kw) == 'wait'
    assert m.watch_decision(**dict(kw, age_seconds=1900)) == 'restart'
    assert m.watch_decision(**dict(kw, active='failed')) == 'restart'
    assert m.watch_decision(**dict(kw, active='failed', owned=False)) == 'blocked'
    assert m.watch_decision(**dict(kw, active='failed', ports_occupied=True)) == 'blocked'
    assert m.watch_decision(**dict(kw, active='failed', restarts=2)) == 'blocked'
    assert m.watch_decision(**dict(kw, state='complete')) == 'finished'
    assert m.watch_decision(**dict(kw, until=now)) == 'expired'


def test_search_pairs_screening_seeds_and_reviews_only_one_candidate(tmp_path, monkeypatch):
    m = module(); calls = []
    monkeypatch.setattr(m, 'fingerprints', lambda: {'fixture': 'unchanged'})
    def run(output, job, seeds, until):
        calls.append((job, seeds))
        wins = 8 if job['phase'] == 'review' else 0
        reports = [{'success': i < wins, 'objects': []} for i in range(len(seeds))]
        return {'candidate': job, 'reports': reports, 'rounds': len(seeds),
                'successes': wins, 'completed_object_stages': 0}
    monkeypatch.setattr(m, 'run_job', run)
    m.run_search(tmp_path, datetime.now(timezone.utc)+timedelta(hours=1))
    assert len(calls) == 26
    assert all(seeds == m.SCREEN_SEEDS for _, seeds in calls[:-1])
    assert calls[-1][1] == m.REVIEW_SEEDS
    assert not set(m.REVIEW_SEEDS) & set(m.SCREEN_SEEDS)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['complete'] and summary['expert_gate_passed']
    assert summary['training_started'] is False


def test_expired_search_does_not_launch_the_simulator(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'fingerprints', lambda: {'fixture': 'unchanged'})
    def forbidden(*args, **kwargs): pytest.fail('到期后不得启动OrcaLab')
    monkeypatch.setattr(m, 'Experiment', forbidden)
    m.run_search(tmp_path, datetime.now(timezone.utc)-timedelta(seconds=1))
    assert json.loads((tmp_path/'status.json').read_text())['state'] == 'expired'


def test_changed_sources_stop_resume_instead_of_mixing_results(tmp_path, monkeypatch):
    m = module()
    (tmp_path/'plan.json').write_text(json.dumps({'source_sha256': {'old': 'source'}}))
    monkeypatch.setattr(m, 'fingerprints', lambda: {'new': 'source'})
    with pytest.raises(RuntimeError, match='指纹'):
        m.run_search(tmp_path, datetime.now(timezone.utc)+timedelta(hours=1))
    assert json.loads((tmp_path/'status.json').read_text())['state'] == 'blocked'
