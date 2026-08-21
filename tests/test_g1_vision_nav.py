"""CPU-only tests for the G1 point-goal navigation policy boundary."""

from __future__ import annotations

import numpy as np

from envs.euler.g1_vision_nav.contracts import (
    NavigationObservation,
    VelocityCommand,
)
from envs.euler.g1_vision_nav.point_goal_navigator import (
    PointGoalNavigator,
    world_goal_in_body_frame,
)
from envs.euler.g1_vision_nav.visual_avoidance import (
    GreenWorkbenchDetector,
    VisualAvoidanceNavigator,
)


def _observation(goal_xy_robot_m: tuple[float, float]) -> NavigationObservation:
    return NavigationObservation(
        rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        goal_xy_robot_m=np.asarray(goal_xy_robot_m, dtype=np.float32),
        base_velocity_xy_mps=np.zeros(2, dtype=np.float32),
        base_yaw_rate_rps=0.0,
        previous_command=VelocityCommand.stopped(),
        frame_index=0,
        sim_time_s=0.0,
    )


def _observation_with_rgb(
    goal_xy_robot_m: tuple[float, float],
    rgb: np.ndarray,
) -> NavigationObservation:
    observation = _observation(goal_xy_robot_m)
    return NavigationObservation(
        rgb=rgb,
        goal_xy_robot_m=observation.goal_xy_robot_m,
        base_velocity_xy_mps=observation.base_velocity_xy_mps,
        base_yaw_rate_rps=observation.base_yaw_rate_rps,
        previous_command=observation.previous_command,
        frame_index=1,
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


def test_point_goal_navigator_turns_in_place_for_side_goal() -> None:
    command = PointGoalNavigator().act(_observation((0.0, 1.0)))
    assert command.walk_enabled
    assert command.forward_mps == 0.0
    assert command.yaw_rate_rps > 0.0


def test_point_goal_navigator_uses_slow_arc_for_moderate_bearing() -> None:
    command = PointGoalNavigator().act(_observation((1.0, 0.5)))
    assert command.walk_enabled
    assert command.forward_mps > 0.0
    assert 0.0 < command.lateral_mps <= PointGoalNavigator().limits.max_lateral_mps
    assert command.yaw_rate_rps > 0.0


def test_point_goal_navigator_moves_slowly_toward_forward_goal() -> None:
    navigator = PointGoalNavigator()
    command = navigator.act(_observation((2.0, 0.0)))
    assert command.walk_enabled
    assert 0.0 < command.forward_mps <= navigator.limits.max_forward_mps
    assert command.yaw_rate_rps == 0.0


def test_green_workbench_detector_ignores_gray_floor() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    risk = GreenWorkbenchDetector().estimate(rgb)
    assert risk.risk_fraction == 0.0
    assert not risk.obstacle_visible


def test_green_workbench_detector_reports_image_regions() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    risk = GreenWorkbenchDetector().estimate(rgb)
    assert risk.obstacle_visible
    assert risk.right_fraction > risk.left_fraction
    assert risk.center_fraction > 0.0


def test_visual_avoidance_matches_point_goal_on_clear_rgb() -> None:
    observation = _observation((2.0, 0.0))
    baseline = PointGoalNavigator().act(observation)
    visual = VisualAvoidanceNavigator().act(observation)
    assert visual == baseline


def test_visual_avoidance_slows_and_steers_to_clear_side() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    navigator = VisualAvoidanceNavigator()
    command = navigator.act(_observation_with_rgb((2.0, 0.0), rgb))
    assert command.walk_enabled
    assert 0.0 <= command.forward_mps < navigator.limits.max_forward_mps
    assert command.lateral_mps > 0.0
    assert command.yaw_rate_rps > 0.0
    assert navigator.intervention_count == 1


def test_visual_avoidance_holds_clearance_after_obstacle_leaves_view() -> None:
    obstacle_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    obstacle_rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    navigator = VisualAvoidanceNavigator()
    navigator.act(_observation_with_rgb((2.0, 0.0), obstacle_rgb))

    command = navigator.act(_observation((0.0, -2.0)))
    assert navigator.avoidance_side == 1
    assert command.forward_mps == navigator.visual.approach_forward_mps
    assert command.lateral_mps > 0.0
    assert command.yaw_rate_rps == 0.0
