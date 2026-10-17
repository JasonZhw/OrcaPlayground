import mujoco
import numpy as np

from tools.medication_sorting import physics_diagnostics as diagnostics


def test_snapshot_retains_full_physics_and_contact_forces_without_mutating_state():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <site name="ee" pos="0 0 .4"/>
      <body name="target" pos="0 0 .049"><freejoint name="target_free"/>
        <geom name="medicine" type="sphere" size=".05" mass=".2"/></body>
      <body name="basket" pos=".5 0 .1"><geom type="box" size=".1 .1 .01"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
    result = diagnostics.capture_snapshot(model, data, target='target', basket='basket',
                                          site='ee', joint_names=[])
    np.testing.assert_array_equal(result['qpos'], data.qpos)
    np.testing.assert_array_equal(result['qvel'], data.qvel)
    np.testing.assert_allclose(result['ee_position_m'], [0, 0, .4])
    np.testing.assert_allclose(result['target_quaternion_wxyz'], [1, 0, 0, 0])
    assert any(row['normal_force_n'] > 0 and 'target' in row['bodies']
               for row in result['contacts'])
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    assert data.time == before[2]


def test_failure_diagnosis_does_not_equate_solver_failure_to_proven_unreachability():
    helper = getattr(diagnostics, 'compare_ik_attempts', None)
    assert callable(helper)
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="x" type="slide" axis="1 0 0" range="-.1 .1"/>
      <geom size=".01"/><site name="tip"/>
    </body></worldbody></mujoco>''')
    report = helper(model, model.qpos0, ['x'], 'tip', [1, 0, 0], np.eye(3), attempts=(1, 4))
    assert report['classification'] == 'no_feasible_solution_found'
    assert report['mathematically_proven_unreachable'] is False
    assert all(not row['solved'] for row in report['attempts'])
