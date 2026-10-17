import mujoco
import numpy as np
import pytest
import importlib.util
import importlib

from tools.medication_sorting import run_expert


def test_four_targets_have_correct_bins_and_distinct_language():
    targets = getattr(run_expert, 'TARGETS', {})
    assert len(targets) == 4, '必须显式支持全部四个物品'
    assert len({t['task'] for t in targets.values()}) == 4
    for name, target in targets.items():
        assert target['basket_suffix'] == ('_Box_Blue' if 'bottle' in name else '_Box_Purple')
        assert target['task'].isascii()


def test_collision_bounds_include_cylinder_and_child_but_not_visual_geom():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="bottle" pos="1 2 3">
        <geom type="cylinder" size=".045 .073" pos="0 0 .081"/>
        <geom type="box" size="2 2 2" contype="0" conaffinity="0"/>
        <body name="cap" pos="0 0 .16"><geom type="sphere" size=".03"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = run_expert.box_bounds(model, data, 'bottle')
    np.testing.assert_allclose(low, [1-.045, 2-.045, 3+.008])
    np.testing.assert_allclose(high, [1+.045, 2+.045, 3+.19])


def test_bottle_must_not_pass_placement_using_only_its_smaller_cap():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="basket"><geom type="box" size=".04 .1 .005"/></body>
      <body name="bottle" pos="0 0 .035">
        <geom type="cylinder" size=".045 .03"/>
        <geom type="box" size=".02 .02 .01" pos="0 0 -.02"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert not run_expert.placement_geometry(model, data, 'bottle', 'basket')['inside']


def test_rotated_cylinder_bounds_are_computed_in_basket_frame():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="frame" quat=".70710678 0 0 .70710678"/>
      <body name="bottle" quat=".70710678 0 .70710678 0">
        <geom type="cylinder" size=".02 .08"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = run_expert.box_bounds(model, data, 'bottle', frame_name='frame')
    np.testing.assert_allclose(low, [-.02, -.08, -.02], atol=1e-8)
    np.testing.assert_allclose(high, [.02, .08, .02], atol=1e-8)


def test_unknown_target_is_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='物品'):
        run_expert.execute(None, tmp_path/'bad', target='not_a_medicine')
    assert not (tmp_path/'bad').exists()


def test_collector_rejects_unknown_object_before_capture(tmp_path):
    from tools.medication_sorting.collect_episodes import collect
    with pytest.raises(ValueError, match='物品'):
        collect(None, None, tmp_path/'bad', target='not_a_medicine')
    assert not (tmp_path/'bad').exists()


def test_continuation_cannot_reset_object_positions(tmp_path):
    with pytest.raises(ValueError, match='禁止重置'):
        run_expert.execute(None, tmp_path/'bad', reset=False, reset_offset=(.01, 0))
    assert not (tmp_path/'bad').exists()


def test_bottle_grasp_orientation_does_not_change_box_baseline():
    helper = getattr(run_expert, 'grasp_rotation', None)
    assert callable(helper)
    np.testing.assert_allclose(helper(run_expert.DEFAULT_TARGET), run_expert.TOP_DOWN_ROTATION)
    rotation = helper('melatonin_tablet_bottle_1')
    assert not np.allclose(rotation, run_expert.TOP_DOWN_ROTATION)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_four_object_summary_cannot_hide_missing_or_failed_targets():
    module_name='tools.medication_sorting.run_four_objects'
    assert importlib.util.find_spec(module_name) is not None, '需要四物品批次入口'
    summarize=importlib.import_module(module_name).summarize_results
    rows=[{'target':name,'success':True,'quality_accepted':True}
          for name in run_expert.TARGETS]
    assert summarize(rows, rgb=True)['all_four_objects_have_accepted_data']
    assert not summarize(rows[:-1], rgb=True)['all_four_objects_have_accepted_data']
    rows[-1]['quality_accepted']=False
    assert not summarize(rows, rgb=True)['all_four_objects_have_accepted_data']
    assert not summarize(rows, rgb=False)['all_four_objects_have_accepted_data']


def test_continuous_summary_requires_final_scene_proof():
    module_name='tools.medication_sorting.run_four_objects'
    assert importlib.util.find_spec(module_name) is not None, '需要四物品批次入口'
    summarize=importlib.import_module(module_name).summarize_results
    rows=[{'target':name,'success':True,'quality_accepted':True}
          for name in run_expert.TARGETS]
    assert not summarize(rows, rgb=True, continuous=True)['continuous_sorting_success']
    final={name:{'inside':True,'stable':True} for name in run_expert.TARGETS}
    assert summarize(rows, rgb=True, continuous=True, final=final)['continuous_sorting_success']
    final[next(iter(final))]['inside']=False
    assert not summarize(rows, rgb=True, continuous=True, final=final)['continuous_sorting_success']


def test_continuous_rgb_exit_requires_quality_as_well_as_physics():
    summarize=importlib.import_module('tools.medication_sorting.run_four_objects').summarize_results
    rows=[{'target':name,'success':True,'quality_accepted':True} for name in run_expert.TARGETS]
    final={name:{'inside':True,'stable':True} for name in run_expert.TARGETS}
    rows[-1]['quality_accepted']=False
    summary=summarize(rows, rgb=True, continuous=True, final=final)
    assert summary['continuous_sorting_success']
    assert summary.get('success') is False, '连续采集不能忽略RGB或同步验收失败'
