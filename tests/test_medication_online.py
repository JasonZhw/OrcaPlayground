"""Online control must preserve semantic order and never replay stale actions."""

import importlib

import numpy as np
import pytest


def online():
    return importlib.import_module("examples.euler.medication_sorting.pi05.online")


def test_online_module_exists():
    assert importlib.util.find_spec("examples.euler.medication_sorting.pi05.online") is not None


def conditioner(initial=None, tau=0.06):
    limits = np.tile([-2.0, 2.0], (16, 1))
    limits[[7, 15]] = [0, 1]
    return online().ActionConditioner(
        np.zeros(16) if initial is None else initial,
        limits,
        max_arm_delta=0.08,
        dt=0.02,
        arm_tau=tau,
        close_speed=0.625,
        open_speed=1.6,
    )


def test_observed_closure_jump_is_ramped_without_weakening_arm_guard():
    initial = np.zeros(16)
    initial[7] = 0.5460332058094166
    c = conditioner(initial)
    requested = initial.copy()
    requested[7] = 0.6417924194736148
    out = c.step(requested)
    assert out[7] == pytest.approx(initial[7] + 0.0125)
    for _ in range(7):
        out = c.step(requested)
    assert out[7] == requested[7]
    requested[7] = 0
    assert c.step(requested)[7] == pytest.approx(out[7] - 0.032)
    previous = c.executed.copy()
    requested[0] = 0.2
    with pytest.raises(ValueError, match="discontinuity"):
        c.step(requested)
    np.testing.assert_array_equal(c.executed, previous)


def test_arm_smoothing_reduces_alternation_but_reaches_sustained_target():
    c = conditioner()
    outputs = []
    target = np.zeros(16)
    for i in range(100):
        target[0] = 0.01 if i % 2 else -0.01
        outputs.append(c.step(target)[0])
    assert np.max(np.abs(outputs[-20:])) < 0.003
    target[0] = 0.04
    for _ in range(50):
        result = c.step(target)
    assert result[0] == pytest.approx(0.04, abs=1e-7)
    assert np.all(result[1:] == 0)
    np.testing.assert_array_equal(target, np.r_[0.04, np.zeros(15)])


@pytest.mark.parametrize("index,value", [(7, -0.01), (15, 1.01), (0, float("nan")), (0, 3)])
def test_conditioner_rejects_bad_input_before_filtering(index, value):
    c = conditioner()
    target = np.zeros(16)
    target[index] = value
    with pytest.raises(ValueError):
        c.step(target)
    np.testing.assert_array_equal(c.executed, np.zeros(16))


def test_delayed_chunk_discards_elapsed_actions_and_replaces_only_future():
    q = online().ActionQueue()
    a = np.repeat(np.arange(50)[:, None], 16, axis=1)
    q.replace(a, observed_step=0, current_step=0)
    np.testing.assert_array_equal(q.pop(0), np.zeros(16))
    b = a + 100
    assert q.replace(b, observed_step=10, current_step=15) == 5
    np.testing.assert_array_equal(q.pop(15), np.full(16, 105))
    assert q.remaining(16) == 44
    with pytest.raises(RuntimeError, match="exhausted"):
        q.pop(60)
    with pytest.raises(ValueError, match="expired"):
        q.replace(b, observed_step=10, current_step=60)


def test_chunk_refuses_future_observation_and_nonfinite_prediction():
    q = online().ActionQueue()
    for a, observed in [(np.zeros((49, 16)), 0), (np.full((50, 16), np.nan), 0), (np.zeros((50, 16)), 1)]:
        with pytest.raises(ValueError):
            q.replace(a, observed, 0)


def test_joint_and_closure_limits_and_discontinuity_are_not_silently_clipped():
    limits = np.tile([-2.0, 2.0], (16, 1))
    limits[[7, 15]] = [0, 1]
    prev = np.zeros(16)
    safe = prev.copy()
    safe[8] = 0.02
    safe[15] = 0.03
    np.testing.assert_array_equal(online().checked_action(safe, prev, limits, 0.08), safe)
    for idx, val in [(0, 3), (7, -0.01), (15, 1.01), (8, 0.2), (2, float("nan"))]:
        bad = prev.copy()
        bad[idx] = val
        with pytest.raises(ValueError):
            online().checked_action(bad, prev, limits, 0.08)


def test_observation_joint_order_is_two_fifteen_dimensional_groups():
    qpos = np.arange(50.0)
    got = online().state_vector(qpos, [np.arange(7), np.arange(25, 32)], [np.arange(7, 15), np.arange(32, 40)])
    np.testing.assert_array_equal(got, np.r_[np.arange(15), np.arange(25, 40)])


def test_overlap_transition_starts_on_existing_trajectory_and_ends_on_new_one():
    q = online().ActionQueue()
    q.replace(np.zeros((50, 16)), 0, 0)
    replacement = np.full((50, 16), 0.12)
    q.replace(replacement, 20, 31, blend_steps=10)
    got = np.asarray([q.pop(i) for i in range(31, 42)])
    np.testing.assert_array_equal(got[0], np.zeros(16))
    np.testing.assert_allclose(got[9:], 0.12)
    assert np.max(np.abs(np.diff(got, axis=0))) < 0.03
    np.testing.assert_array_equal(replacement, np.full((50, 16), 0.12))


def test_no_history_is_invented_when_transition_has_run_out_of_overlap():
    q = online().ActionQueue()
    q.replace(np.zeros((50, 16)), 0, 0)
    with pytest.raises(ValueError, match="overlap"):
        q.replace(np.ones((50, 16)), 20, 49, blend_steps=10)


def test_small_gripper_extrapolation_is_explicitly_saturated_without_clipping_arm_targets():
    raw = np.full((50, 16), 0.2)
    raw[0, 7] = -0.05
    raw[0, 15] = 1.05
    raw[1, 0] = 3.0
    actions, count = online().bounded_closures(raw)
    assert count == 2
    np.testing.assert_array_equal(actions[0, [7, 15]], [0, 1])
    assert actions[1, 0] == 3.0
    assert raw[0, 7] == -0.05
    raw[0, 7] = -0.2
    with pytest.raises(ValueError, match="extrapolation"):
        online().bounded_closures(raw)


def test_camera_listener_is_enabled_before_connecting_and_restored():
    import asyncio
    from types import SimpleNamespace

    from examples.euler.medication_sorting.pi05 import simulate

    enabled = {n: False for n in simulate.CAMERAS}

    class Bridge:
        async def get_camera_names(self):
            return list(enabled)

        async def get_camera_properties(self, name):
            return SimpleNamespace(
                width=640,
                height=480,
                streaming_enabled=enabled[name],
                capture_rgb=False,
                capture_depth=False,
                capture_normal=False,
                capture_object_color=False,
                color_port=7000,
            )

        async def set_streaming_enabled(self, name, value):
            enabled[name] = value

        async def set_camera_properties(self, name, **fields):
            pass

    class Receiver:
        def __init__(self, name, port, size):
            self.name = name

        def start(self):
            if not enabled[self.name]:
                raise ConnectionError("Camera listener disabled")

        def stop(self):
            pass

    loop = asyncio.new_event_loop()
    cameras = simulate.NativeCameras(Bridge(), loop, factory=Receiver)
    try:
        cameras.start()
        assert all(enabled.values())
    finally:
        assert cameras.close() == []
        loop.close()
    assert not any(enabled.values())


@pytest.mark.parametrize("latency", [1, 11, 20])
def test_oracle_chunk_timing_preserves_recorded_timestep_and_last_frame(latency):
    from examples.euler.medication_sorting.action_pipeline import process_recorded_actions

    actions = np.zeros((137, 16))
    actions[:, 0] = np.arange(137) * 0.001
    actions[:, 8] = -actions[:, 0]
    limits = np.tile([-2.0, 2.0], (16, 1))
    limits[[7, 15]] = [0, 1]
    requested, executed, events = process_recorded_actions(actions, actions[0], limits, "queue", latency)
    np.testing.assert_allclose(requested, actions, atol=1e-15)
    np.testing.assert_allclose(executed, actions, atol=1e-15)
    assert len(events) > 1 and all(e["discarded"] == latency for e in events[1:])
    assert executed[-1, 0] == pytest.approx(0.136)


def test_oracle_uses_existing_conditioner_without_modifying_source():
    from examples.euler.medication_sorting.action_pipeline import process_recorded_actions

    actions = np.zeros((100, 16))
    actions[10:, 0] = 0.04
    actions[10:, 7] = 1
    source = actions.copy()
    limits = np.tile([-2.0, 2.0], (16, 1))
    limits[[7, 15]] = [0, 1]
    requested, executed, _ = process_recorded_actions(actions, actions[0], limits, "inference", 11)
    expected = conditioner()
    expected_values = np.array([expected.step(row) for row in requested])
    np.testing.assert_allclose(executed, expected_values)
    np.testing.assert_array_equal(source, actions)
    assert executed[10, 0] < 0.04 and executed[10, 7] == pytest.approx(0.0125)
