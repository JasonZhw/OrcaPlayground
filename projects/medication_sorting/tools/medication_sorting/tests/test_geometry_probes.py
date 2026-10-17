import copy

from tools.medication_sorting import geometry_probes as probes


def test_strict_probe_requires_three_contact_lift_steps_and_twenty_final_steps():
    lifted = [{'contact_sides': ['inner', 'outer'], 'box_pos': [0, 0, .07]}] * 3
    stable = [{'inside': True, 'released': True, 'settled': True}] * 20
    assert probes.strict_evidence(lifted, stable, start_z=0)['success']
    assert not probes.strict_evidence(lifted[:2], stable, start_z=0)['success']
    assert not probes.strict_evidence(lifted, stable[:19], start_z=0)['success']


def test_strict_probe_rejects_one_unstable_frame_or_one_finger_contact():
    lifted = [{'contact_sides': ['inner'], 'box_pos': [0, 0, .1]}] * 30
    stable = [{'inside': True, 'released': True, 'settled': True}] * 20
    assert not probes.strict_evidence(lifted, stable, start_z=0)['success']
    lifted = [{'contact_sides': ['inner', 'outer'], 'box_pos': [0, 0, .07]}] * 3
    stable[-1] = {'inside': True, 'released': False, 'settled': True}
    assert not probes.strict_evidence(lifted, stable, start_z=0)['success']


def test_paired_cases_change_only_alignment_or_requested_drop_y():
    cases = {row['name']: row for row in probes.probe_cases()}
    first, second = [dict(cases[name]) for name in ('herb_yaw15_fixed', 'herb_yaw15_aligned')]
    first.pop('name'); second.pop('name')
    assert first.pop('align_grasp_to_object') is False
    assert second.pop('align_grasp_to_object') is True
    assert first == second
    first, second = [dict(cases[name]) for name in ('herb_slot_bias', 'herb_slot_centered_y')]
    first.pop('name'); second.pop('name')
    assert first.pop('placement_offset') == (.018, 0.)
    assert second.pop('placement_offset') == (.018, -.025)
    assert first == second


def test_base_layout_changes_only_root_robot_position_and_preserves_source():
    source = {'actors': [{'name': 'unitree_g1_1', 'transform': {'position': [1., 2., 3.],
                       'scale': 1., 'rotation': [1., 0., 0., 0.]}},
                        {'name': 'medicine', 'transform': {'position': [4., 5., 6.]}}]}
    before = copy.deepcopy(source)
    result = probes.make_base_layout(source, -.015)
    assert source == before
    assert result['actors'][0]['transform']['position'] == [1., 1.985, 3.]
    result['actors'][0]['transform']['position'] = [1., 2., 3.]
    assert result == before
