import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.medication_sorting import task_specs


SOURCE = Path(__file__).resolve().parents[3] / 'data/medication_sorting/scenes/drag_classfication_capture.json'
BOTTLES = [name for name in task_specs.TARGETS if 'bottle' in name]


def scale(source, ratio=.7):
    from tools.medication_sorting import scaled_layout
    return scaled_layout.make_scaled_bottle_layout(source, ratio=ratio)


def test_scale_only_two_bottles_and_preserve_source():
    source = json.loads(SOURCE.read_text())
    before = copy.deepcopy(source)
    result = scale(source)
    assert source == before
    for old, new in zip(before['actors'], result['actors']):
        if old['name'] in BOTTLES:
            expected = copy.deepcopy(old)
            expected['transform']['scale'] *= .7
            assert new == expected
        else:
            assert new == old


def test_relative_scaling_updates_only_root_uniform_scale_override():
    source = json.loads(SOURCE.read_text())
    bottle = next(a for a in source['actors'] if a['name'] == BOTTLES[0])
    bottle['transform']['scale'] = .8
    bottle['entity_overrides'] = [
        {'entity_path': [], 'group_overrids': [{'property_overrides': [
            {'name': 'uniformScale', 'value': .8}, {'name': 'translate.z', 'value': 4.75}]}]},
        {'entity_path': [{'name': 'child'}], 'group_overrids': [{'property_overrides': [
            {'name': 'uniformScale', 'value': .9}]}]},
    ]
    changed = next(a for a in scale(source)['actors'] if a['name'] == BOTTLES[0])
    assert changed['transform']['scale'] == pytest.approx(.56)
    assert changed['entity_overrides'][0]['group_overrids'][0]['property_overrides'][0]['value'] == pytest.approx(.56)
    assert changed['entity_overrides'][0]['group_overrids'][0]['property_overrides'][1]['value'] == 4.75
    assert changed['entity_overrides'][1] == bottle['entity_overrides'][1]


@pytest.mark.parametrize('ratio', [0, -1, 1.1, float('nan'), float('inf')])
def test_invalid_shrink_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match='比例'):
        scale(json.loads(SOURCE.read_text()), ratio)


@pytest.mark.parametrize('duplicate', [False, True])
def test_requires_unique_bottles(duplicate):
    source = json.loads(SOURCE.read_text())
    bottle = next(a for a in source['actors'] if a['name'] == BOTTLES[0])
    if duplicate:
        source['actors'].append(copy.deepcopy(bottle))
    else:
        source['actors'].remove(bottle)
    with pytest.raises(ValueError, match='唯一'):
        scale(source)


def test_scaled_profile_is_explicit_and_preserves_old_parameters():
    before = copy.deepcopy(task_specs.TARGETS)
    helper = getattr(task_specs, 'target_spec', None)
    assert callable(helper), '需要显式区分原尺寸和缩小药瓶的抓取配置'
    for name in before:
        assert helper(name) == before[name]
        scaled = helper(name, profile='scaled_bottles_70')
        if name in BOTTLES:
            assert scaled['grasp_clearance'] == -.005
            assert scaled.get('tilt_x_degrees', 0) == (-10 if 'melatonin' in name else 0)
            assert scaled['travel_z'] == 5.04
            assert scaled['lower_z'] == (4.99 if 'melatonin' in name else 4.91)
            assert scaled['initial_closure'] == 0
        else:
            assert scaled == before[name]
    assert task_specs.TARGETS == before
    with pytest.raises(ValueError, match='配置'):
        helper(BOTTLES[0], profile='unknown')


def test_scaled_large_bottle_orientation_is_top_down():
    expert = importlib.import_module('tools.medication_sorting.run_expert')
    np.testing.assert_allclose(
        expert.grasp_rotation(BOTTLES[0], profile='scaled_bottles_70'), expert.TOP_DOWN_ROTATION)


def test_scaled_collection_requires_explicit_layout_before_output(tmp_path):
    from tools.medication_sorting.collect_episodes import collect
    with pytest.raises(ValueError, match='显式'):
        collect(None, None, tmp_path/'bad', profile='scaled_bottles_70')
    assert not (tmp_path/'bad').exists()


def test_scaled_layout_guard_rejects_original_and_accepts_exact_copy(tmp_path):
    from tools.medication_sorting import scaled_layout
    helper = getattr(scaled_layout, 'resolve_experiment_layout', None)
    assert callable(helper), '缩小配置必须核对所记录的场景'
    with pytest.raises(ValueError, match='不一致'):
        helper(SOURCE, profile='scaled_bottles_70')
    path = tmp_path/'scaled.json'
    path.write_text(json.dumps(scale(json.loads(SOURCE.read_text()))))
    assert helper(path, profile='scaled_bottles_70') == path
    assert helper(None, profile='original') == SOURCE


@pytest.mark.parametrize('radius, accepted', [(.0315, True), (.045, False), (.015, False)])
def test_scaled_profile_checks_live_collision_size(radius, accepted):
    import mujoco
    expert = importlib.import_module('tools.medication_sorting.run_expert')
    helper = getattr(expert, 'validate_live_profile', None)
    assert callable(helper), '实际物理模型也必须与缩小配置匹配'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="b_Medical_32"><geom type="cylinder" size="{radius} .05"/></body>
      <body name="b_Medical_31"><geom type="cylinder" size=".028 .04"/></body>
    </worldbody></mujoco>''')
    if accepted:
        helper(model, profile='scaled_bottles_70')
    else:
        with pytest.raises(ValueError, match='瓶径'):
            helper(model, profile='scaled_bottles_70')


def test_collector_rejects_wrong_live_scale_without_masking_error(tmp_path):
    import mujoco
    from types import SimpleNamespace
    from tools.medication_sorting.collect_episodes import collect
    path = tmp_path/'scaled.json'
    path.write_text(json.dumps(scale(json.loads(SOURCE.read_text()))))
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="b_Medical_32"><geom type="cylinder" size=".045 .05"/></body>
      <body name="b_Medical_31"><geom type="cylinder" size=".04 .04"/></body>
    </worldbody></mujoco>''')
    with pytest.raises(ValueError, match='瓶径'):
        collect(SimpleNamespace(model=model), None, tmp_path/'bad',
                profile='scaled_bottles_70', layout=path)
    assert not (tmp_path/'bad').exists()
