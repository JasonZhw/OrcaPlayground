import numpy as np
import mujoco
import pytest
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from tools.medication_sorting.robot_control import gripper_angle, solve_site_ik, validate_action
from tools.medication_sorting import robot_control


def test_ik_reaches_position_and_orientation_with_actual_forward_kinematics():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
      <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
      <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" range="-2 2"/>
      <geom type="sphere" size=".01" mass="1"/><site name="tip"/>
      </body></worldbody></mujoco>''')
    rotation = Rotation.from_euler('z', .4).as_matrix()
    solution = solve_site_ik(model, model.qpos0, ['x','y','z','yaw'], 'tip', [.2,-.3,.1], rotation)
    data = mujoco.MjData(model)
    data.qpos[:] = solution['qpos']
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[0], [.2,-.3,.1], atol=1e-4)
    np.testing.assert_allclose(data.site_xmat[0].reshape(3,3), rotation, atol=1e-4)


def test_ik_does_not_claim_unreachable_pose():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0" range="-.1 .1"/>
      <geom size=".01" mass="1"/><site name="tip"/>
      </body></worldbody></mujoco>''')
    with pytest.raises(RuntimeError, match='不可达'):
        solve_site_ik(model, model.qpos0, ['x'], 'tip', [1,0,0], np.eye(3))


def test_redundant_ik_keeps_nearby_joint_branch():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x1" type="slide" axis="1 0 0" range="-1 1"/>
      <joint name="x2" type="slide" axis="1 0 0" range="-1 1"/>
      <geom size=".01" mass="1"/><site name="tip"/>
      </body></worldbody></mujoco>''')
    seed = np.array([.7, -.5])
    result = solve_site_ik(model, seed, ['x1', 'x2'], 'tip', [.21, 0, 0],
                           np.eye(3), attempts=1, joint_regularization=.003)
    np.testing.assert_allclose(result['joints'], [.705, -.495], atol=1e-4)
    assert result['position_error'] < 1e-4


def test_ik_keeps_feasible_later_candidate_even_if_regularized_cost_is_larger(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x1" type="slide" axis="1 0 0" range="-10 10"/>
      <joint name="x2" type="slide" axis="1 0 0" range="-10 10"/>
      <geom size=".01"/><site name="tip"/>
    </body></worldbody></mujoco>''')
    candidates = iter([np.array([0., 0.]), np.array([8., -7.991])])

    def optimizer(fun, seed, **kwargs):
        q = next(candidates)
        return SimpleNamespace(x=q, fun=fun(q))

    monkeypatch.setattr(robot_control, 'least_squares', optimizer)
    result = solve_site_ik(model, model.qpos0, ['x1', 'x2'], 'tip', [.009, 0, 0],
                           np.eye(3), attempts=2, position_tolerance=.008,
                           joint_regularization=.001)
    np.testing.assert_allclose(result['joints'], [8., -7.991])
    assert result['position_error'] < 1e-8


@pytest.mark.parametrize('action', [np.zeros(7), [np.nan]*8, [0]*7+[1.1]])
def test_invalid_action_rejected(action):
    with pytest.raises(ValueError):
        validate_action(action, np.tile([-2,2], (7,1)))


def test_gripper_closure_is_reversible_within_physical_joint_limits():
    for closure in np.linspace(0,1,11):
        angle = gripper_angle(closure)
        assert -.785 < angle < .7
        assert .65-angle == pytest.approx(closure)
    with pytest.raises(ValueError):
        gripper_angle(-.1)


def test_action_transition_obeys_same_joint_and_gripper_rate_as_policy():
    start=np.zeros(8)
    goal=np.array([.61]*7+[1.])
    actions=np.array(list(robot_control.interpolate_actions(start,goal,
                         np.tile([-2.,2.],(7,1)),.2)))
    differences=np.diff(np.vstack([start,actions]),axis=0)
    assert np.max(np.abs(differences[:,:7]))<=.24+1e-12
    assert np.max(np.abs(differences[:,7]))<=.32+1e-12
    np.testing.assert_allclose(actions[-1],goal)
