import importlib
import importlib.util
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tools.medication_sorting.task_specs import TARGETS


def module():
    name = 'tools.medication_sorting.continuous_expert'
    assert importlib.util.find_spec(name) is not None, '需要独立的严格连续专家验收入口'
    return importlib.import_module(name)


def sample(*, good=True, contact=False, z=.1):
    return {name: {'inside': good, 'released': not contact, 'settled': good,
                   'contact_sides': ['inner', 'outer'] if contact else [],
                   'position': [0., 0., z]}
            for name in TARGETS}


def test_joint_final_evidence_requires_twenty_shared_frames_and_real_lifts():
    cls = module().RoundEvidence
    evidence = cls({name: 0. for name in TARGETS})
    for _ in range(3): evidence.append(sample(contact=True))
    for _ in range(19): evidence.append(sample())
    assert not evidence.final_passed()
    evidence.append(sample())
    assert evidence.final_passed()
    broken = sample(); broken[next(iter(TARGETS))]['inside'] = False
    evidence.append(broken)
    assert not evidence.final_passed(), '后放物品碰出先前物品时必须整轮失败'


def test_one_frame_lift_and_supported_object_cannot_pass():
    evidence = module().RoundEvidence({name: 0. for name in TARGETS})
    evidence.append(sample(contact=True))
    for _ in range(20): evidence.append(sample())
    assert not evidence.final_passed()
    for _ in range(3): evidence.append(sample(contact=True))
    supported = sample(); supported[next(iter(TARGETS))]['released'] = False
    for _ in range(20): evidence.append(supported)
    assert not evidence.final_passed()


def test_budget_is_checked_before_the_next_physical_step():
    evidence = module().RoundEvidence({name: 0. for name in TARGETS}, per_target_limit=2)
    name = next(iter(TARGETS))
    evidence.begin_target(name)
    for _ in range(2):
        evidence.check_budget()
        evidence.append(sample())
    with pytest.raises(RuntimeError, match='预算'): evidence.check_budget()
    assert evidence.total_steps == 2


def test_reset_round_randomizes_all_four_once_and_preserves_source_model(monkeypatch):
    m = module()
    body_xml = ''.join(f'<body name="object{spec["body_suffix"]}" pos="{i} 0 1">'
                       '<freejoint/><geom type="box" size=".02 .02 .02"/></body>'
                       for i, spec in enumerate(TARGETS.values()))
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{body_xml}</worldbody></mujoco>')
    data = mujoco.MjData(model)

    class Robot:
        reset_count = 0
        def reset(self, **kwargs):
            self.reset_count += 1
            mujoco.mj_resetData(model, data)
        def body_pose(self, name):
            return {'Pos': data.xpos[model.body(name).id].copy()}
    robot = Robot(); robot.model = model; robot.data = data
    monkeypatch.setattr(m, 'home_joints', lambda robot: np.zeros(7))
    before = model.qpos0.copy()
    first = m.initialize_round(robot, seed=60001, randomize=True)
    assert robot.reset_count == 1
    assert set(first) == set(TARGETS)
    assert all(np.any(np.abs(row['offset_m']) > 0) for row in first.values())
    qpos = data.qpos.copy()
    m.initialize_round(robot, seed=60001, randomize=True)
    assert robot.reset_count == 2
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(model.qpos0, before)


def test_failed_round_stops_without_resetting_or_attempting_later_objects(tmp_path, monkeypatch):
    m = module()
    robot = SimpleNamespace(sim_time=0.); calls = []
    monkeypatch.setattr(m, 'initialize_round', lambda r, **kw:
                        {name: {'position': [0., 0., 0.]} for name in TARGETS})
    def fail(r, folder, **kwargs):
        calls.append((r, kwargs))
        raise RuntimeError('抓取失败')
    monkeypatch.setattr(m, 'execute', fail)
    result = m.execute_round(robot, tmp_path/'round', seed=6)
    assert len(calls) == 1 and calls[0][0] is robot
    assert calls[0][1]['reset'] is False
    assert result['success'] is False
    assert len([r for r in result['objects'] if r['attempted']]) == 1
    assert len([r for r in result['objects'] if not r['attempted']]) == 3
    assert (tmp_path/'round'/'report.json').exists()


def test_successful_round_uses_one_initializer_and_the_same_robot_for_all_targets(tmp_path, monkeypatch):
    m = module()
    robot = SimpleNamespace(sim_time=0., status=sample())
    robot.state = lambda: np.zeros(8)
    calls, resets = [], []
    def initialize(r, **kwargs):
        resets.append(r)
        return {name: {'position': [0., 0., 0.]} for name in TARGETS}
    def simulate(r, folder, **kwargs):
        calls.append((r, kwargs['target'], kwargs['reset']))
        for i in range(23):
            r.sim_time += .2
            r.status = sample()
            r.status[kwargs['target']]['contact_sides'] = ['inner', 'outer'] if i < 3 else []
            r.status[kwargs['target']]['released'] = i >= 3
            kwargs['after_step'](r, {'action': np.zeros(8), 'phase': 'lift' if i < 3 else 'release'})
    monkeypatch.setattr(m, 'initialize_round', initialize)
    monkeypatch.setattr(m, 'execute', simulate)
    monkeypatch.setattr(m, 'scene_status', lambda r: r.status)
    result = m.execute_round(robot, tmp_path/'round', seed=6)
    assert result['success']
    assert resets == [robot]
    assert [row[1] for row in calls] == list(m.ORDER)
    assert all(row[0] is robot and row[2] is False for row in calls)
    assert result['total_steps'] == 92


def test_expert_exposes_post_step_and_budget_hooks():
    import inspect
    from tools.medication_sorting.run_expert import execute
    parameters = inspect.signature(execute).parameters
    assert 'after_step' in parameters, '连续验收必须看到实际执行后的物理帧'
    assert 'max_steps' in parameters, '插值产生的额外动作也必须计入预算'


def test_grasp_loss_guard_requires_sustained_loss_and_never_relabels_success():
    guard = module().GraspLossGuard()
    guard.check('carry', ['inner', 'outer'])
    guard.check('carry', [])
    guard.check('carry', ['inner'])
    guard.check('carry', [])
    with pytest.raises(RuntimeError, match='夹持丢失'): guard.check('carry', [])


def test_guard_does_not_mistake_intentional_release_for_lost_grasp():
    guard = module().GraspLossGuard()
    for _ in range(20): guard.check('release', [])


def test_left_arm_parking_is_initialization_only_and_holds_the_matching_servo():
    m = module()
    name = 'unitree_g1_1_left_shoulder_roll_joint'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body>
      <joint name="{name}" range="-1.58 2.25" limited="true"/>
      <geom type="sphere" size=".01"/></body></worldbody>
      <actuator><position name="{name}" joint="{name}"/></actuator></mujoco>'''.replace(
          '<mujoco>', '<mujoco><compiler angle="radian"/>'))
    data = mujoco.MjData(model)
    robot = SimpleNamespace(model=model, data=data, ctrl=np.zeros(1))
    before = model.qpos0.copy()
    m.park_left_arm_at_initialization(robot)
    assert data.qpos[0] == pytest.approx(1.2)
    assert robot.ctrl[0] == pytest.approx(1.2)
    np.testing.assert_array_equal(model.qpos0, before)


def test_round_drop_correction_is_opt_in_and_does_not_change_other_targets():
    m = module()
    from tools.medication_sorting.run_four_objects import DROP_OFFSETS
    for name in TARGETS:
        assert m.round_drop_offset(name) == DROP_OFFSETS[name]
        expected = np.asarray(DROP_OFFSETS[name]).copy()
        if name == 'fu_shi_gao_herb_drink_packaging_box_1': expected[1] -= .025
        np.testing.assert_allclose(m.round_drop_offset(name, herb_y_correction=-.025), expected)
    with pytest.raises(ValueError): m.round_drop_offset(next(iter(TARGETS)), herb_y_correction=.1)


def test_upright_phase_is_guarded_and_opt_in():
    import inspect
    from tools.medication_sorting.run_expert import execute
    parameter = inspect.signature(execute).parameters.get('upright_before_lower')
    assert parameter is not None and parameter.default is False
    guard = module().GraspLossGuard()
    guard.check('upright', [])
    with pytest.raises(RuntimeError, match='夹持丢失'): guard.check('upright', [])
