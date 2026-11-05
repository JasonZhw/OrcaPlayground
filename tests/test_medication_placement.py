"""Release geometry and command transitions, without a live OrcaLab connection."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.euler.medication_sorting import execution, workflow
from examples.euler.medication_sorting.planning import time_samples
from tests.test_medication_geometry import geometry_scene


def test_retiming_limits_acceleration_including_starts_stops():
    # Include a short reversal: velocity-only retiming used to accelerate sharply here.
    path = [[0, 0], [0.30, -0.01], [0.299, 0.002], [0.31, 0]]
    dt = 0.02
    samples = time_samples(path, dt=dt, max_velocity=0.4, max_acceleration=0.8)
    padded = np.vstack([samples[0], samples, samples[-1]])
    assert np.abs(np.diff(padded, axis=0) / dt).max() <= 0.4 + 1e-10
    assert np.abs(np.diff(padded, n=2, axis=0) / dt**2).max() <= 0.8 + 1e-10
    # Quintic easing starts/ends with near-zero acceleration as well as velocity.
    assert np.abs(np.diff(padded[:4], n=2, axis=0) / dt**2).max() < 0.15
    np.testing.assert_allclose(samples[-1], path[-1])


def test_release_uses_object_bottom_and_bin_floor_not_body_origin():
    scene = geometry_scene()
    scene.bodies["calcium"] = scene.bodies["item"]
    scene.bodies["left_bin"] = scene.bodies["bin"]
    carried = SimpleNamespace(local_position=np.array([0, 0, -0.15]), local_rotation=np.eye(3))
    result = workflow.placement_targets(scene, "calcium", carried, np.eye(3))
    object_center = result["release"] + carried.local_position
    # This test object's origin is its centre, not its bottom.
    assert object_center[2] - 0.05 == pytest.approx(0.01 + 0.08)
    assert result["above"][2] > result["release"][2]
    np.testing.assert_allclose(result["above"][:2], result["release"][:2])
    # Rotation of the bin must rotate the chosen slot.
    np.testing.assert_allclose(object_center[:2], [1.045, 2])


def test_release_rejects_item_too_wide_for_bin():
    scene = geometry_scene()
    scene.bodies["calcium"] = scene.bodies["item"]
    scene.bodies["left_bin"] = scene.bodies["bin"]
    scene.m.geom_size[next(iter(scene.body_geoms("item"))), :2] = [0.5, 0.5]
    carried = SimpleNamespace(local_position=np.zeros(3), local_rotation=np.eye(3))
    with pytest.raises(ValueError, match="fit"):
        workflow.placement_targets(scene, "calcium", carried, np.eye(3))


def test_gradual_release_keeps_arm_and_other_gripper_commands_fixed():
    grippers = SimpleNamespace(closure={"left": 1.0, "right": 0.7})
    grippers.set = lambda arm, value: grippers.closure.__setitem__(arm, float(value))
    runner = object.__new__(execution.Executor)
    runner.env = SimpleNamespace(grippers=grippers, command=np.arange(14, dtype=float))
    samples = []
    runner.tick = lambda command, stage: samples.append((command.copy(), grippers.closure.copy(), stage))
    runner.ramp_gripper("left", 0, 1.2, "release")
    values = np.array([row[1]["left"] for row in samples])
    assert values[0] == 1 and values[-1] == 0
    assert len(values) >= 61
    assert np.all(np.diff(values) <= 0)
    assert np.abs(np.diff(values)).max() < 0.04
    for command, closure, stage in samples:
        np.testing.assert_array_equal(command, runner.env.command)
        assert closure["right"] == 0.7 and stage == "release"


def test_move_phase_boundary_preserves_command_despite_tracking_error(tmp_path, monkeypatch):
    runner = object.__new__(execution.Executor)
    runner.out, runner.index, runner.plans = tmp_path, 0, []
    names = [f"joint_{i}" for i in range(7)]
    env = SimpleNamespace(command=np.zeros(7), data=SimpleNamespace(qpos=np.zeros(7)), joint_names=names)
    runner.env = env
    runner.scene = SimpleNamespace(
        arms={"left": SimpleNamespace(joint_names=names, qpos_ids=np.arange(7))},
        end_pose=lambda arm: (np.zeros(3), np.eye(3)),
    )
    runner.sync = lambda: None
    sent = []

    def tick(command, stage):
        sent.append(command.copy())
        env.command = command.copy()
        env.data.qpos = command - 0.001  # The physical arm trails its command.

    runner.tick = tick
    starts = []

    def plan(scene, arm, position, rotation, qpos, **kwargs):
        starts.append(qpos.copy())
        return np.array([qpos, np.full(7, 0.1 * len(starts))])

    monkeypatch.setattr(execution, "plan_motion", plan)
    runner.move("left", np.zeros(3), np.eye(3), "first")
    prior = env.command.copy()
    runner.move("left", np.zeros(3), np.eye(3), "second")
    np.testing.assert_array_equal(starts[1], prior)
    assert np.abs(np.diff(np.asarray(sent), n=2, axis=0) / 0.02**2).max() <= 0.8 + 1e-9


def test_startup_alignment_only_updates_arms_and_is_acceleration_limited():
    runner = object.__new__(execution.Executor)
    names = [f"joint_{i}" for i in range(9)]
    runner.env = SimpleNamespace(
        joint_names=names,
        command=np.arange(9, dtype=float),
        data=SimpleNamespace(qpos=np.arange(9, dtype=float) + 0.05),
    )
    runner.scene = SimpleNamespace(
        arms={"left": SimpleNamespace(joint_names=names[:7], qpos_ids=np.arange(7))},
        collision_free=lambda *args, **kwargs: True,
    )
    commands = []

    def tick(command, stage):
        commands.append(command.copy())
        runner.env.command = command.copy()

    runner.tick = tick
    runner.align_initial_command()
    np.testing.assert_allclose(commands[-1][:7], np.arange(7) + 0.05)
    np.testing.assert_array_equal(np.asarray(commands)[:, 7:], np.tile([7, 8], (len(commands), 1)))
    padded = np.vstack([commands[0], commands, commands[-1]])
    assert np.abs(np.diff(padded, n=2, axis=0) / 0.02**2).max() <= 0.4 + 1e-9


@pytest.mark.parametrize('error_type', [workflow.IKError, workflow.PlanningError])
def test_unplannable_lowering_can_release_over_bin(tmp_path, monkeypatch, error_type):
    runner = object.__new__(workflow.Trial)
    runner.out = tmp_path
    runner.sync = lambda: None
    runner.scene = object()
    runner.move = lambda *args, **kwargs: (_ for _ in ()).throw(error_type('blocked lower'))
    monkeypatch.setattr(workflow, 'release_over_bin', lambda *args: True, raising=False)
    assert runner.lower_for_release('left', 'calcium', np.zeros(3), np.eye(3), None) is False
    assert 'blocked lower' in (tmp_path / 'release_fallbacks.json').read_text()


def test_lowering_failure_outside_bin_is_not_ignored(tmp_path, monkeypatch):
    runner = object.__new__(workflow.Trial)
    runner.out = tmp_path
    runner.sync = lambda: None
    runner.scene = object()
    runner.move = lambda *args, **kwargs: (_ for _ in ()).throw(workflow.IKError('blocked lower'))
    monkeypatch.setattr(workflow, 'release_over_bin', lambda *args: False, raising=False)
    with pytest.raises(workflow.IKError):
        runner.lower_for_release('left', 'calcium', np.zeros(3), np.eye(3), None)


def test_execution_failure_does_not_trigger_release(tmp_path, monkeypatch):
    runner = object.__new__(workflow.Trial)
    runner.move = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('lost grasp'))
    with pytest.raises(RuntimeError, match='lost grasp'):
        runner.lower_for_release('left', 'calcium', np.zeros(3), np.eye(3), None)


@pytest.mark.parametrize('position,allowed', [
    ([1, 2, .10], True),
    ([1.15, 2, .10], True),
    ([1, 2.09, .10], False),  # Centre is inside, but footprint crosses the rim.
    ([1, 2, .30], False),
    ([1, 2, .03], False),
])
def test_hover_release_checks_entire_footprint_and_height(position, allowed):
    scene = geometry_scene()
    q = scene.d.qpos.copy()
    q[:3] = position
    scene.set_qpos(q)
    assert workflow.release_over_bin(scene, 'item', 'bin') is allowed


@pytest.mark.parametrize('key,bin_key,extra', [
    ('calcium', 'left_bin', .03), ('melatonin', 'left_bin', .03),
    ('herbbox', 'right_bin', 0), ('pharyngitis', 'right_bin', 0),
])
def test_bottles_enter_bin_three_cm_higher_without_changing_release(key, bin_key, extra):
    scene = geometry_scene()
    scene.bodies[key] = scene.bodies['item']
    scene.bodies[bin_key] = scene.bodies['bin']
    carried = SimpleNamespace(local_position=np.array([0, 0, -.15]), local_rotation=np.eye(3))
    placement = workflow.placement_targets(scene, key, carried, np.eye(3))
    np.testing.assert_allclose(placement['above'] - placement['release'], [0, 0, .06 + extra])
    bottom = placement['release'][2] + carried.local_position[2] - .05
    assert bottom == pytest.approx(.01 + (.08 if bin_key == 'left_bin' else .05))


def test_failed_plan_keeps_exact_input_for_offline_diagnosis(tmp_path, monkeypatch):
    import json
    runner = object.__new__(execution.Executor)
    runner.out, runner.index, runner.plans = tmp_path, 123, []
    names = [f'j{i}' for i in range(7)]
    runner.env = SimpleNamespace(command=np.ones(7)*.2, data=SimpleNamespace(qpos=np.ones(7)*.19), joint_names=names)
    runner.scene = SimpleNamespace(arms={'left': SimpleNamespace(joint_names=names, qpos_ids=np.arange(7))})
    runner.sync = lambda: None
    monkeypatch.setattr(execution, 'plan_motion', lambda *a, **kw: (_ for _ in ()).throw(workflow.IKError('no valid goal')))
    with pytest.raises(workflow.IKError):
        runner.move('left', np.zeros(3), np.eye(3), 'bottle_transfer', target='melatonin', allow=True)
    context = json.loads((tmp_path/'motion_context.json').read_text())
    assert context['seed'] == 123 and context['stage'] == 'bottle_transfer'
    np.testing.assert_array_equal(context['qpos'], runner.env.command)
    assert context['target'] == 'melatonin' and context['allow'] is True
