"""CPU-only tests for the G1 point-goal navigation policy boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from envs.euler.g1_vision_nav.config import NavigationConfig, VisualAvoidanceConfig

from envs.euler.g1_vision_nav.contracts import (
    NavigationObservation,
    VelocityCommand,
)
from envs.euler.g1_vision_nav.point_goal_navigator import (
    PointGoalNavigator,
    world_goal_in_body_frame,
)
from envs.euler.g1_vision_nav.green_table_navigator import (
    GreenTableDetector,
    GreenTableNavigator,
)


def _observation(goal_xy_robot_m: tuple[float, float]) -> NavigationObservation:
    return NavigationObservation(
        rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        goal_xy_robot_m=np.asarray(goal_xy_robot_m, dtype=np.float32),
        base_velocity_xy_mps=np.zeros(2, dtype=np.float32),
        base_yaw_rate_rps=0.0,
        base_yaw_world_rad=np.pi / 2,
        previous_command=VelocityCommand.stopped(),
        frame_index=0,
        sim_time_s=0.0,
    )


def _observation_with_rgb(
    goal_xy_robot_m: tuple[float, float],
    rgb: np.ndarray,
    frame_index: int = 1,
) -> NavigationObservation:
    return replace(
        _observation(goal_xy_robot_m),
        rgb=rgb,
        frame_index=frame_index,
        sim_time_s=0.1,
    )


def test_world_goal_transform_respects_robot_yaw() -> None:
    yaw_left_90 = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    goal_body = world_goal_in_body_frame(
        robot_xy_world=np.asarray([2.0, 3.0]),
        body_xmat_world=yaw_left_90,
        goal_xy_world=np.asarray([2.0, 4.0]),
    )
    np.testing.assert_allclose(goal_body, [1.0, 0.0], atol=1e-6)


def test_point_goal_navigator_stops_inside_tolerance() -> None:
    command = PointGoalNavigator().act(_observation((0.2, 0.0)))
    assert command == VelocityCommand.stopped()


@pytest.mark.parametrize("yaw_deg, turn_sign", [(0, 1), (180, -1), (-179, -1)])
def test_goal_position_requires_final_heading(yaw_deg, turn_sign) -> None:
    observation = replace(_observation((0.1, 0.0)), base_yaw_world_rad=np.deg2rad(yaw_deg))
    command = PointGoalNavigator().act(observation)
    assert command.walk_enabled
    assert 0 < command.forward_mps <= 0.04
    assert command.lateral_mps == 0
    assert command.yaw_rate_rps * turn_sign > 0


@pytest.mark.parametrize("distance,yaw_deg,reached", [
    (0.2, 90, True), (0.2, 81, True), (0.2, 79, False),
    (1.01, 90, False), (1.0, 90, True), (0.825, 87.9, True), (0, 0, False),
])
def test_goal_pose_requires_position_and_yaw(distance, yaw_deg, reached) -> None:
    assert NavigationConfig().goal_reached(distance, np.deg2rad(yaw_deg)) == reached


def test_goal_heading_wraps_at_pi() -> None:
    config = NavigationConfig(goal_yaw_deg=-179)
    assert config.heading_error_rad(np.deg2rad(179)) == pytest.approx(np.deg2rad(2))


@pytest.mark.parametrize("xy,yaw_deg,should_stop", [
    ((5.248, -0.177), 144.9, False),
    ((5.144, -0.698), 87.9, True),
])
def test_terminal_heading_priority_replays_reported_orbit_poses(xy, yaw_deg, should_stop):
    # Observed in the user's run: positive position bearing conflicted with
    # negative final-heading error, or good heading was rejected by distance.
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    relative = world_goal_in_body_frame(np.array(xy),
        np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), np.array([4.7, 0]))
    observation = replace(_observation(tuple(relative)), base_yaw_world_rad=yaw)
    navigator = GreenTableNavigator()
    command = navigator.act(observation)
    if should_stop:
        assert command == VelocityCommand.stopped()
    else:
        assert command.yaw_rate_rps < 0, "near goal, turn toward 90 degrees, not the point bearing"
        assert navigator.latest_mode == "align_goal"


def test_terminal_alignment_survives_small_position_drift_but_is_bounded():
    navigator = GreenTableNavigator()
    for distance in (0.79, 0.85, 0.99):
        observation = replace(_observation((0, distance)), base_yaw_world_rad=np.deg2rad(140))
        assert navigator.act(observation).yaw_rate_rps < 0
        assert navigator.latest_mode == "align_goal"
    far = replace(observation, goal_xy_robot_m=np.array([0, 1.1]))
    assert navigator.act(far).yaw_rate_rps > 0  # reacquire the point outside the allowed area
    navigator.reset()
    assert navigator.act(observation).yaw_rate_rps > 0  # no stale alignment state across episodes


@pytest.mark.parametrize("radius", [0, -0.1, 1.0, float("nan"), float("inf")])
def test_terminal_alignment_radius_requires_a_valid_hysteresis_band(radius):
    with pytest.raises(ValueError):
        NavigationConfig(goal_alignment_radius_m=radius)


def test_goal_controller_rechecks_pose_after_arrival() -> None:
    navigator = PointGoalNavigator()
    arrived = _observation((0.1, 0.0))
    assert not navigator.act(arrived).walk_enabled
    assert navigator.act(replace(arrived, base_yaw_world_rad=0)).walk_enabled
    assert navigator.act(replace(arrived, goal_xy_robot_m=np.array([1.1, 0]))).walk_enabled


@pytest.mark.parametrize("field,value", [
    ("goal_yaw_deg", float("nan")), ("goal_yaw_deg", float("inf")),
    ("goal_yaw_tolerance_deg", 0), ("goal_yaw_tolerance_deg", 181),
    ("goal_yaw_tolerance_deg", float("nan")),
])
def test_goal_heading_config_rejects_invalid_values(field, value) -> None:
    with pytest.raises(ValueError):
        NavigationConfig(**{field: value})


def test_environment_pose_to_navigation_arrival_and_drift(monkeypatch) -> None:
    """Only simulator pose/RGB are fixtures; use real environment and navigation code."""
    import time
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/euler/07_locomotion"))
    from envs.euler.g1_vision_nav.camera_stream import CameraFrame
    from envs.euler.g1_vision_nav.command_bridge import CommandLimiter
    from envs.euler.g1_vision_nav.config import CameraConfig
    from envs.euler.g1_vision_nav.g1_vision_nav_env import G1VisionNavEnv

    class PoseHarness(G1VisionNavEnv):
        def __init__(self):
            # Do not connect to Studio or initialize physics/ONNX in a CPU test.
            self.agent_name = "g1"
            self.goal_xy_world = np.array([4.7, 0.0])
            self.navigation_config = NavigationConfig()
            self.camera_config = CameraConfig()
            self.navigator = GreenTableNavigator(navigation=self.navigation_config)
            self.command_limiter = CommandLimiter()
            self._startup_steps = 0
            self._goal_reached = False
            self._emergency_stop = False
            self._unsafe_collision = None
            self._goal_bonus_awarded = False
            self._reward_previous_distance_m = None
            self._initial_distance_m = 7.5
            self._minimum_distance_m = 0.1
            self._latest_frame = CameraFrame(np.zeros((24, 32, 3), dtype=np.uint8), 1, time.monotonic())
            self.xy = np.array([4.7, -0.1])
            self.yaw_deg = 0.0

        @property
        def data(self):
            return SimpleNamespace(time=3.0)

        def get_body_xpos_xmat_xquat(self, names):
            yaw = np.deg2rad(self.yaw_deg)
            c, s = np.cos(yaw), np.sin(yaw)
            return {names[0]: {"xpos": [*self.xy, 0.75], "xmat": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}}

        def query_joint_qvel(self, names):
            return {names[0]: np.zeros(6)}

        def request(self):
            return self._requested_command(101)

        def reward(self):
            return self._compute_reward({}, np.zeros(1))

        def terminated(self):
            return self._is_terminated({})

    class Results:
        def __init__(self):
            self.checks = {}

        def check(self, name, passed, *args):
            self.checks[name] = passed

    env = PoseHarness()
    assert env.request().yaw_rate_rps > 0
    assert not env.terminated()
    assert env.reward() < 0  # distance-only arrival must not award the goal bonus
    result = Results()
    env.verify_final(result)
    assert not result.checks["point_goal_reached"]
    env.yaw_deg = 90
    assert env.request() == VelocityCommand.stopped()
    assert env.terminated()
    assert env.reward() > 4
    env.verify_final(result)
    assert result.checks["point_goal_reached"]
    env.xy[1] = -0.9
    env.yaw_deg = 140
    assert env.request().yaw_rate_rps < 0  # slight position/yaw drift keeps final heading priority
    assert not env.terminated()
    env.xy[1] = -1.2  # outside the permitted area, resume position following
    assert env.request().forward_mps > 0
    assert not env.terminated()
    env.verify_final(result)
    assert not result.checks["point_goal_reached"]


def test_point_goal_navigator_uses_safe_forward_arc_for_side_goal() -> None:
    command = PointGoalNavigator().act(_observation((0.0, 2.0)))
    assert command.walk_enabled
    assert 0.0 < command.forward_mps <= PointGoalNavigator().navigation.turning_forward_mps
    assert command.yaw_rate_rps > 0.0


def test_point_goal_navigator_uses_slow_arc_for_moderate_bearing() -> None:
    command = PointGoalNavigator().act(_observation((1.0, 0.5)))
    assert command.walk_enabled
    assert command.forward_mps > 0.0
    assert command.lateral_mps == 0.0
    assert command.yaw_rate_rps > 0.0


def test_point_goal_navigator_moves_slowly_toward_forward_goal() -> None:
    navigator = PointGoalNavigator()
    command = navigator.act(_observation((2.0, 0.0)))
    assert command.walk_enabled
    assert 0.0 < command.forward_mps <= navigator.limits.max_forward_mps
    assert command.yaw_rate_rps == 0.0


def test_green_workbench_detector_ignores_gray_floor() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    risk = GreenTableDetector().estimate(rgb)
    assert risk.risk_fraction == 0.0
    assert not risk.obstacle_visible


def test_green_workbench_detector_reports_image_regions() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    risk = GreenTableDetector().estimate(rgb)
    assert risk.obstacle_visible
    assert risk.right_fraction > risk.left_fraction
    assert risk.center_fraction > 0.0


def test_green_workbench_at_image_edge_does_not_block_route() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, :25, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    risk = GreenTableDetector().estimate(rgb)
    assert risk.left_fraction > 0.0
    assert risk.center_fraction == 0.0
    assert risk.risk_fraction == 0.0
    assert not risk.obstacle_visible


def test_green_table_navigation_matches_point_goal_on_clear_rgb() -> None:
    observation = _observation((2.0, 0.0))
    baseline = PointGoalNavigator().act(observation)
    visual = GreenTableNavigator().act(observation)
    assert visual == baseline


@pytest.mark.parametrize("clear,slow", [
    (-0.01, 0.10), (0.08, 1.01), (0.10, 0.10), (0.20, 0.10),
    (float("nan"), 0.10), (0.08, float("nan")),
])
def test_visual_risk_thresholds_require_valid_range_and_hysteresis(clear, slow) -> None:
    with pytest.raises(ValueError):
        VisualAvoidanceConfig(clear_risk_fraction=clear, slow_risk_fraction=slow)


def test_green_table_navigation_slows_and_steers_to_clear_side() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    navigator = GreenTableNavigator()
    command = navigator.act(_observation_with_rgb((2.0, 0.0), rgb))
    assert command.walk_enabled
    assert command.forward_mps == navigator.visual.turn_forward_mps
    assert command.lateral_mps == 0.0
    assert command.yaw_rate_rps > 0.0
    assert navigator.latest_mode == "turn_clear"
    assert navigator.intervention_count == 1


def test_green_table_navigation_confirms_clear_then_gradually_returns_to_live_target() -> None:
    obstacle_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    obstacle_rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    clear_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    navigator = GreenTableNavigator()
    navigator.act(_observation_with_rgb((2.0, 0.0), obstacle_rgb, frame_index=1))

    command = navigator.act(
        _observation_with_rgb((0.0, -2.0), clear_rgb, frame_index=2)
    )

    assert command.yaw_rate_rps == 0.0
    assert navigator.latest_mode == "confirm_clear"
    for frame_index in range(3, 30):
        observation = replace(
            _observation_with_rgb((0.0, -2.0), clear_rgb, frame_index=frame_index),
            sim_time_s=0.1 + (frame_index - 2) * 0.1,
        )
        command = navigator.act(observation)
        if navigator.latest_mode == "return_target":
            assert navigator.latest_base_command.yaw_rate_rps <= command.yaw_rate_rps <= 0.0

    assert command == navigator.latest_base_command
    assert navigator.avoidance_side == 0
    assert navigator.latest_mode == "follow_target"


def test_green_table_navigation_recomputes_pose_goal_command_below_threshold() -> None:
    obstacle_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    obstacle_rgb[20:21, 30:50, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    navigator = GreenTableNavigator()

    first_command = navigator.act(_observation_with_rgb((2.0, 0.0), obstacle_rgb))
    first_base = navigator.latest_base_command
    second_command = navigator.act(_observation_with_rgb((2.0, -0.5), obstacle_rgb))
    second_base = navigator.latest_base_command

    assert first_base.yaw_rate_rps == 0.0
    assert second_base.yaw_rate_rps < 0.0
    assert second_command.yaw_rate_rps < first_command.yaw_rate_rps


def _clearance_observation(t, frame, blocked=False, goal=(0.0, -2.0)):
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    if blocked:
        rgb[:36, 45:] = [70, 145, 130]
    return replace(_observation_with_rgb(goal, rgb, frame), sim_time_s=t)


def test_repeated_rgb_cannot_confirm_clearance():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    for step in range(1, 100):
        command = navigator.act(_clearance_observation(step * .02, 2))
        assert navigator.latest_mode == "confirm_clear"
        assert command.yaw_rate_rps == 0


def test_clearance_requires_055_seconds_then_blends_for_one_second():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    for step in range(1, 7):
        navigator.act(_clearance_observation(step * .1, step + 1))
        assert navigator.latest_mode == "confirm_clear"  # first clear frame at 0.1 s
    command = navigator.act(_clearance_observation(.65, 8))
    assert navigator.latest_mode == "return_target"
    assert command.yaw_rate_rps == 0
    for step in range(1, 10):
        command = navigator.act(_clearance_observation(.65 + step * .1, 8 + step))
        assert navigator.latest_mode == "return_target"
        assert command.yaw_rate_rps == pytest.approx(step * .1 * navigator.latest_base_command.yaw_rate_rps)
    command = navigator.act(_clearance_observation(1.65, 18))
    assert navigator.latest_mode == "follow_target"
    assert command == navigator.latest_base_command


def test_green_reappears_during_return_immediately_resumes_same_avoidance_side():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    for step in range(1, 11):
        navigator.act(_clearance_observation(step * .1, step + 1))
    assert navigator.latest_mode == "return_target"
    command = navigator.act(_clearance_observation(1.1, 12, blocked=True))
    assert navigator.latest_mode == "turn_clear"
    assert command.yaw_rate_rps > 0
    navigator.act(_clearance_observation(1.2, 13))
    assert navigator.latest_mode == "confirm_clear"


def test_camera_gap_requires_new_clear_confirmation():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    for step in range(1, 11):
        navigator.act(_clearance_observation(step * .1, step + 1))
    assert navigator.latest_mode == "return_target"
    navigator.act(_clearance_observation(3, 30))
    assert navigator.latest_mode == "confirm_clear"


def test_return_blends_instead_of_replaying_old_goal_direction():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    for step in range(1, 10):
        command = navigator.act(_clearance_observation(step * .1, step + 1))
    assert navigator.latest_mode == "return_target"
    assert navigator.latest_base_command.yaw_rate_rps < command.yaw_rate_rps < 0
    changed = navigator.act(_clearance_observation(1.0, 11, goal=(1.0, 1.0)))
    assert 0 < changed.yaw_rate_rps < navigator.latest_base_command.yaw_rate_rps


def test_green_flicker_restarts_confirmation():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    for step in range(1, 6):
        navigator.act(_clearance_observation(step * .1, step + 1))
    navigator.act(_clearance_observation(.6, 7, blocked=True))
    for step in range(7, 12):
        navigator.act(_clearance_observation(step * .1, step + 1))
        assert navigator.latest_mode == "confirm_clear"


def test_goal_stop_and_reset_override_clearance():
    navigator = GreenTableNavigator()
    navigator.act(_clearance_observation(0, 1, blocked=True))
    assert navigator.act(_clearance_observation(.1, 2, goal=(.1, 0))) == VelocityCommand.stopped()
    assert not navigator.avoidance_active
    navigator.reset()
    observation = _clearance_observation(.2, 3)
    assert navigator.act(observation) == PointGoalNavigator().act(observation)


@pytest.mark.parametrize("name", ["clear_confirm_s", "return_blend_s", "clear_frame_gap_s"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_clearance_timings_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError):
        VisualAvoidanceConfig(**{name: value})
