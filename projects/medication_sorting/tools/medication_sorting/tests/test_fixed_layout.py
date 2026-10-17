import copy
import json
from pathlib import Path

import pytest

from tools.medication_sorting.fixed_layout import make_fixed_layout


SOURCE = Path(__file__).resolve().parents[3] / 'data/medication_sorting/scenes/drag_classfication.json'


def test_fixed_root_preserves_original_and_dynamic_boxes():
    source = json.loads(SOURCE.read_text())
    before = copy.deepcopy(source)
    fixed = make_fixed_layout(source)
    assert source == before
    robot = next(a for a in fixed['actors'] if a['name'] == 'unitree_g1_1')
    overrides = [o for o in robot['entity_overrides'] if len(o['entity_path']) == 4]
    assert len(overrides) == 1
    group = overrides[0]['group_overrids'][0]
    assert group['type_id'] == '{E544201F-83F6-419F-B3DD-1D6559FFDFB0}'
    assert group['property_overrides'] == [{'name':'jointType','value':'fixed','type':'enum'}]
    for actor in source['actors']:
        if actor['name'] != 'unitree_g1_1':
            assert actor == next(a for a in fixed['actors'] if a['name'] == actor['name'])


def test_fixed_layout_requires_unique_robot():
    source = json.loads(SOURCE.read_text())
    source['actors'] = [a for a in source['actors'] if a['name'] != 'unitree_g1_1']
    with pytest.raises(ValueError, match='机器人'):
        make_fixed_layout(source)


def test_robot_offset_keeps_task_objects_in_place():
    source = json.loads(SOURCE.read_text())
    fixed = make_fixed_layout(source, robot_offset=[.07, -.15, 0])
    original = next(a for a in source['actors'] if a['name'] == 'unitree_g1_1')
    robot = next(a for a in fixed['actors'] if a['name'] == 'unitree_g1_1')
    assert robot['transform']['position'] == pytest.approx([
        original['transform']['position'][0]+.07,
        original['transform']['position'][1]-.15,
        original['transform']['position'][2],
    ])
    assert 'translate.x' not in [p['name'] for o in robot['entity_overrides'] if not o['entity_path']
                                for g in o['group_overrids'] for p in g['property_overrides']]


def test_basket_scale_changes_experiment_copy_only():
    source = json.loads(SOURCE.read_text())
    before = copy.deepcopy(source)
    fixed = make_fixed_layout(source, basket_scale_factor=1.2)
    original = next(a for a in source['actors'] if a['name']=='purple_plastic_crates_1')
    basket = next(a for a in fixed['actors'] if a['name']=='purple_plastic_crates_1')
    assert source == before
    assert basket['transform']['scale'] == pytest.approx(original['transform']['scale']*1.2)
