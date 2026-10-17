"""离线控制契约测试：仅调用纯函数，不创建机器人、相机或模型服务。

失败用例是待主线程修复的真实安全缺口，不用 xfail 隐藏。
0.06 rad/帧与 0.08 closure/帧在 5 Hz 下分别为指令的
0.3 rad/s 与 0.4 closure/s；在 20 Hz 下则为 1.2 与 1.6。
这些约束不等于实际关节速度限制。
"""

import numpy as np
import pytest
from types import SimpleNamespace

from tools.medication_sorting.infer_policy import limit_action
from tools.medication_sorting import infer_policy


@pytest.fixture
def limits():
    return np.array([[-1.0, 1.0], [-0.3, 0.4], [-2.0, 2.0],
                     [-1.5, 0.7], [-0.6, 0.8], [-1.1, 1.4], [-0.9, 0.9]])


def test_small_absolute_targets_are_preserved(limits):
    previous = np.array([0.2, 0.1, -0.4, 0.1, 0.3, -0.2, 0.4, 0.5])
    target = previous + np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0, -0.03, 0.04])
    result = limit_action(target, previous, limits)
    np.testing.assert_allclose(result, target)
    assert not np.shares_memory(result, target)


def test_large_requests_obey_each_axis_rate_without_mutating_inputs(limits):
    previous = np.array([0.0] * 7 + [0.5])
    target = np.array([100, -100, 100, -100, 100, -100, 100, -10.0])
    copies = [value.copy() for value in (target, previous, limits)]
    actual = limit_action(target, previous, limits)
    np.testing.assert_allclose(actual, [0.06, -0.06, 0.06, -0.06, 0.06, -0.06, 0.06, 0.42])
    for value, original in zip((target, previous, limits), copies, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("direction", [-1, 1])
def test_joint_and_closure_limits_hold_near_the_boundaries(limits, direction):
    previous = np.r_[limits[:, 1] - 0.01, 0.99] if direction > 0 else np.r_[limits[:, 0] + 0.01, 0.01]
    actual = limit_action(np.full(8, direction * 100.0), previous, limits)
    np.testing.assert_allclose(actual, np.r_[limits[:, 1], 1] if direction > 0 else np.r_[limits[:, 0], 0])
    assert np.all(np.abs(actual[:7] - previous[:7]) <= 0.06 + 1e-12)
    assert abs(actual[7] - previous[7]) <= 0.08 + 1e-12


def test_multistep_reversal_cannot_bypass_custom_rate_limits(limits):
    previous = np.array([0.0] * 7 + [0.5])
    for target in [np.full(8, 100.0)] * 40 + [np.full(8, -100.0)] * 80:
        actual = limit_action(target, previous, limits, max_joint_step=0.02, max_gripper_step=0.03)
        assert np.isfinite(actual).all()
        assert np.all((actual[:7] >= limits[:, 0]) & (actual[:7] <= limits[:, 1]))
        assert 0 <= actual[7] <= 1
        assert np.max(np.abs(actual[:7] - previous[:7])) <= 0.02 + 1e-12
        assert abs(actual[7] - previous[7]) <= 0.03 + 1e-12
        previous = actual


def test_zero_rate_means_hold_current_command(limits):
    previous = np.array([0.1] * 7 + [0.5])
    np.testing.assert_array_equal(limit_action(np.ones(8), previous, limits,
                                             max_joint_step=0, max_gripper_step=0), previous)


@pytest.mark.parametrize("fps", [5, 10, 20])
def test_same_speed_contract_scales_command_step_with_control_frequency(limits, fps):
    previous = np.r_[np.zeros(7), 0.5]
    actual = limit_action(np.full(8, 100.0), previous, limits,
                          max_joint_step=1.2 / fps, max_gripper_step=1.6 / fps)
    np.testing.assert_allclose(actual[:7], np.full(7, 1.2 / fps))
    assert actual[7] - previous[7] == pytest.approx(1.6 / fps)


@pytest.mark.parametrize("target", [np.zeros(7), np.zeros((1, 8)), np.full(8, np.nan), np.full(8, np.inf)])
def test_invalid_model_actions_are_rejected(limits, target):
    with pytest.raises(ValueError):
        limit_action(target, np.r_[np.zeros(7), 0.5], limits)


@pytest.mark.parametrize("index,value", [(0, 1.2), (0, -1.2), (7, 1.2), (7, -0.2), (0, np.nan)])
def test_invalid_previous_command_is_rejected_before_clipping(limits, index, value):
    previous = np.r_[np.zeros(7), 0.5]
    previous[index] = value
    with pytest.raises(ValueError):
        limit_action(np.r_[np.zeros(7), 0.5], previous, limits)


@pytest.mark.parametrize("keyword,value", [("max_joint_step", -0.1), ("max_gripper_step", -0.1),
                                          ("max_joint_step", np.nan), ("max_gripper_step", np.inf)])
def test_invalid_rate_configuration_fails_closed(limits, keyword, value):
    with pytest.raises(ValueError):
        limit_action(np.ones(8), np.r_[np.zeros(7), 0.5], limits, **{keyword: value})


@pytest.mark.parametrize("bad_pair", [[1.0, -1.0], [np.nan, 1.0]])
def test_invalid_joint_intervals_are_rejected(limits, bad_pair):
    limits[0] = bad_pair
    with pytest.raises(ValueError):
        limit_action(np.ones(8), np.r_[np.zeros(7), 0.5], limits)


def test_action_chunk_cannot_exceed_episode_step_budget():
    chunk=np.zeros((10,8),dtype=np.float32)
    assert len(infer_policy.bounded_chunk(chunk,execute_chunk=4,remaining=2))==2
    assert len(infer_policy.bounded_chunk(chunk,execute_chunk=4,remaining=9))==4
    with pytest.raises(ValueError):infer_policy.bounded_chunk(chunk,execute_chunk=0,remaining=2)


def test_inference_reset_uses_same_xy_yaw_ranges_as_collection():
    offset,yaw=infer_policy.reset_randomization(413)
    rng=np.random.default_rng(413)
    np.testing.assert_array_equal(offset,rng.uniform([-.006,-.008],[.006,.008]))
    assert yaw==rng.uniform(-.025,.025)


def test_grasp_requires_contacts_from_both_real_finger_tips():
    names=['box','unitree_g1_1_gripper_r_inner_link4',
           'unitree_g1_1_gripper_r_outer_link4','unitree_g1_1_gripper_r_inner_link1']
    model=SimpleNamespace(geom_bodyid=np.arange(4),body=lambda i:SimpleNamespace(name=names[i]))
    data=SimpleNamespace(ncon=2,contact=[SimpleNamespace(geom1=0,geom2=1),SimpleNamespace(geom1=0,geom2=2)])
    assert infer_policy.finger_contacts(model,data,'box')=={'inner','outer'}
    data.contact[1].geom2=3
    assert infer_policy.finger_contacts(model,data,'box')=={'inner'}


def test_robot_contract_mismatch_is_rejected_before_execution():
    expected={'robot_type':'unitree_g1_fixed_base_right_arm','state_dim':8,'action_dim':8,
              'joint_units':'rad','joint_action_semantics':'absolute_position',
              'gripper_semantics':{'open':0.,'closed':1.}}
    expected['observation_config']=infer_policy.CAPTURE_CONFIG
    infer_policy.validate_policy_metadata(expected)
    for key in expected:
        wrong=expected.copy();wrong.pop(key)
        with pytest.raises(ValueError):infer_policy.validate_policy_metadata(wrong)


@pytest.mark.parametrize('hz,backend',[(10,'viewport'),(20,'viewport'),(5,'sensor')])
def test_camera_or_frequency_mismatch_cannot_execute_checkpoint(hz,backend):
    metadata={'robot_type':'unitree_g1_fixed_base_right_arm','state_dim':8,'action_dim':8,
              'joint_units':'rad','joint_action_semantics':'absolute_position',
              'gripper_semantics':{'open':0.,'closed':1.},'observation_config':infer_policy.CAPTURE_CONFIG}
    with pytest.raises(ValueError):infer_policy.validate_policy_metadata(metadata,control_hz=hz,camera_backend=backend)


@pytest.mark.parametrize('joint', ['', '<joint type="hinge" axis="0 0 1"/>'])
def test_non_free_object_cannot_be_reset_as_seven_coordinates(joint):
    import mujoco
    model=mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body name="box">{joint}<geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError):infer_policy.free_body_qpos_address(model,'box')


def test_free_object_qpos_address_is_validated():
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="box"><freejoint/><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
    assert infer_policy.free_body_qpos_address(model,'box')==0


def test_sustained_grasp_cannot_be_triggered_by_one_collision():
    count=infer_policy.grasp_streak(0,{'inner','outer'},.1)
    assert count==1
    assert infer_policy.grasp_streak(count,{'inner'},.1)==0
    assert infer_policy.grasp_streak(count,{'inner','outer'},.01)==0
    assert infer_policy.grasp_streak(count,{'inner','outer'},.1)==2
