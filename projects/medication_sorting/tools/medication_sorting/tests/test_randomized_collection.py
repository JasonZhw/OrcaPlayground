"""随机摆放的离线契约测试；不作为真实抓取成功证据。"""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def randomization():
    return importlib.import_module('tools.medication_sorting.randomization')


def test_legacy_seed_keeps_exact_existing_distribution():
    rng = np.random.default_rng(413)
    expected = rng.uniform([-.006, -.008], [.006, .008])
    expected_yaw = float(rng.uniform(-.025, .025))
    offset, yaw = randomization().sample_reset(413)
    np.testing.assert_array_equal(offset, expected)
    assert yaw == expected_yaw


def test_tabletop_has_reproducible_nontrivial_position_and_yaw_coverage():
    module = randomization()
    samples = [module.sample_reset(seed, profile='tabletop_v1') for seed in range(100)]
    offsets = np.asarray([row[0] for row in samples])
    yaws = np.asarray([row[1] for row in samples])
    assert np.max(np.abs(offsets)) <= .02
    assert np.all(np.ptp(offsets, axis=0) > .035)
    assert np.max(np.abs(yaws)) <= np.deg2rad(15)
    assert np.ptp(yaws) > np.deg2rad(25)
    a, b = module.sample_reset(42, profile='tabletop_v1'), module.sample_reset(42, profile='tabletop_v1')
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]
    assert module.profile_config('tabletop_v1')['upright_only'] is True


@pytest.mark.parametrize('seed', [-1, 1.5, True])
def test_randomization_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):
        randomization().sample_reset(seed)


def test_randomization_rejects_unknown_profile():
    with pytest.raises(ValueError):
        randomization().sample_reset(42, profile='undefined')


def test_grasp_orientation_can_follow_randomized_object_yaw():
    from tools.medication_sorting.run_expert import grasp_rotation, DEFAULT_TARGET
    expected = Rotation.from_euler('z', .2).as_matrix() @ grasp_rotation(DEFAULT_TARGET)
    np.testing.assert_allclose(grasp_rotation(DEFAULT_TARGET, yaw=.2), expected)


def test_randomized_profile_opens_herb_gripper_without_changing_old_baselines():
    from tools.medication_sorting.task_specs import target_spec
    name = 'fu_shi_gao_herb_drink_packaging_box_1'
    assert target_spec(name, profile='scaled_bottles_70')['initial_closure'] == .5
    assert target_spec(name, profile='randomized_tabletop_70')['initial_closure'] == 0
    from tools.medication_sorting.run_expert import placement_rotation, grasp_rotation
    rotated = grasp_rotation(name, yaw=.2)
    np.testing.assert_allclose(placement_rotation(name, rotated, align_grasp_to_object=True), grasp_rotation(name))
    np.testing.assert_allclose(placement_rotation(name, rotated), rotated)


def test_release_bias_only_changes_randomized_herb_expert():
    from tools.medication_sorting.task_specs import target_spec, TARGETS
    name = 'fu_shi_gao_herb_drink_packaging_box_1'
    recovered = target_spec(name, profile='randomized_tabletop_70')
    assert recovered.get('placement_bias') == (0., .025)
    assert recovered.get('expert_revision') == 'herb_release_offset_v1'
    assert 'placement_bias' not in target_spec(name, profile='scaled_bottles_70')
    assert 'placement_bias' not in target_spec(name)
    for target in TARGETS:
        if target != name:
            assert 'placement_bias' not in target_spec(target, profile='randomized_tabletop_70')


def test_collection_records_actual_randomization_and_passes_it_to_expert(tmp_path, monkeypatch):
    from tools.medication_sorting import collect_episodes as module
    captured = {}
    def fake_execute(robot, output, **kwargs):
        captured.update(kwargs)
        output.mkdir()
        return {'success': True}
    monkeypatch.setattr(module, 'execute', fake_execute)
    monkeypatch.setattr(module, 'validate_live_profile', lambda *a, **k: None)
    output = tmp_path/'episode_0000'
    robot = SimpleNamespace(model=None, dt=.001, frame_skip=200)
    module.collect(robot, SimpleNamespace(), output, seed=42, randomize=True,
                   randomization_profile='tabletop_v1')
    offset, yaw = randomization().sample_reset(42, profile='tabletop_v1')
    np.testing.assert_array_equal(captured['reset_offset'], offset)
    assert captured['reset_yaw'] == yaw
    assert captured['align_grasp_to_object'] is True
    metadata = json.loads((output/'metadata.json').read_text())
    assert metadata['randomization_profile'] == 'tabletop_v1'
    assert metadata['randomization_config'] == randomization().profile_config('tabletop_v1')
    assert metadata.get('expert_parameters') == module.target_spec('pharyngitis_tablets_box_1')
    implementation = metadata.get('expert_implementation_sha256', {})
    assert set(implementation) == {'run_expert.py', 'robot_control.py', 'task_specs.py'}
    assert all(len(value) == 64 for value in implementation.values())


def test_collection_records_effective_release_offset_instead_of_requested_only(tmp_path, monkeypatch):
    from tools.medication_sorting import collect_episodes as module
    captured = {}
    def fake_execute(robot, output, **kwargs):
        captured.update(kwargs)
        output.mkdir()
        return {'success': True}
    layout = tmp_path/'layout.json'
    layout.write_text('{}')
    monkeypatch.setattr(module, 'resolve_experiment_layout', lambda *a, **k: layout)
    monkeypatch.setattr(module, 'execute', fake_execute)
    monkeypatch.setattr(module, 'validate_live_profile', lambda *a, **k: None)
    output = tmp_path/'episode'
    module.collect(SimpleNamespace(model=None, dt=.001, frame_skip=200), SimpleNamespace(), output,
                   target='fu_shi_gao_herb_drink_packaging_box_1', profile='randomized_tabletop_70',
                   placement_offset=(.01, -.005))
    metadata = json.loads((output/'metadata.json').read_text())
    np.testing.assert_allclose(metadata['placement_offset'], [.01, .02])
    assert metadata['expert_parameters']['expert_revision'] == 'herb_release_offset_v1'
    assert captured['placement_offset'] == (.01, -.005), '补偿只在专家内叠加一次'
