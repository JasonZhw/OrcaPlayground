"""CPU-only tests for the G1 point-goal navigation policy boundary."""

from __future__ import annotations

import numpy as np

from envs.euler.g1_vision_nav.camera_preview import CameraPreviewWindow
from envs.euler.g1_vision_nav.command_bridge import (
    CommandLimiter,
    recovery_velocity_command,
)
from envs.euler.g1_vision_nav.config import NavigationConfig, VisualAvoidanceConfig
from envs.euler.g1_vision_nav.contracts import (
    NavigationObservation,
    VelocityCommand,
)
from envs.euler.g1_vision_nav.point_goal_navigator import (
    PointGoalNavigator,
    world_goal_in_body_frame,
)
from envs.euler.g1_vision_nav.safety_monitor import NavigationSafetyMonitor
from envs.euler.g1_vision_nav.simulation_gait_clock import SimulationGaitClock
from envs.euler.g1_vision_nav.factory_inspection_navigator import (
    FactoryColorObstacleDetector,
    FactoryInspectionNavigator,
)
from envs.euler.g1_vision_nav.waypoint_route import WaypointRoute


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
    frame_index: int = 1,
) -> NavigationObservation:
    observation = _observation(goal_xy_robot_m)
    return NavigationObservation(
        rgb=rgb,
        goal_xy_robot_m=observation.goal_xy_robot_m,
        base_velocity_xy_mps=observation.base_velocity_xy_mps,
        base_yaw_rate_rps=observation.base_yaw_rate_rps,
        previous_command=observation.previous_command,
        frame_index=frame_index,
        sim_time_s=0.1,
    )


def test_default_collision_recovery_last_five_seconds() -> None:
    config = NavigationConfig()

    assert config.collision_recovery_grace_steps == 50


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


def test_point_goal_navigator_uses_safe_forward_arc_for_side_goal() -> None:
    navigator = PointGoalNavigator()
    command = navigator.act(_observation((0.0, 1.0)))
    assert command.walk_enabled
    assert (
        navigator.navigation.minimum_forward_mps
        < command.forward_mps
        < navigator.navigation.turning_forward_mps
    )
    assert command.yaw_rate_rps == 0.65


def test_point_goal_navigator_does_not_walk_away_from_goal_behind() -> None:
    navigator = PointGoalNavigator()
    command = navigator.act(_observation((-2.0, 0.0)))

    assert command.walk_enabled
    assert np.isclose(command.forward_mps, navigator.navigation.minimum_forward_mps)
    assert abs(command.yaw_rate_rps) == navigator.limits.max_yaw_rate_rps


def test_new_waypoint_uses_slow_arc_until_heading_is_acquired() -> None:
    navigator = PointGoalNavigator()
    navigator.on_waypoint_changed()

    turning = navigator.act(_observation((1.0, 1.0)))
    assert 0.0 < turning.forward_mps <= navigator.navigation.waypoint_turning_forward_mps
    assert turning.yaw_rate_rps > 0.0

    aligned = navigator.act(_observation((2.0, 0.2)))
    assert aligned.forward_mps > turning.forward_mps


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


def test_point_goal_navigator_reaches_six_tenths_cruise_on_clear_route() -> None:
    navigator = PointGoalNavigator()
    command = navigator.act(_observation((3.0, 0.0)))
    assert command.forward_mps == 0.60


def test_green_workbench_detector_ignores_gray_floor() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    risk = FactoryColorObstacleDetector().estimate(rgb)
    assert risk.risk_fraction == 0.0
    assert not risk.obstacle_visible


def test_color_detector_ignores_dark_hands_and_peripheral_shelving() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:, :30] = np.asarray([25, 25, 25], dtype=np.uint8)
    rgb[:, 60:] = np.asarray([25, 25, 25], dtype=np.uint8)
    rgb[33:, :] = np.asarray([20, 20, 20], dtype=np.uint8)
    risk = FactoryColorObstacleDetector().estimate(rgb)
    assert risk.dark_fraction > 0.0
    assert risk.risk_fraction == 0.0
    assert not risk.obstacle_visible


def test_green_workbench_detector_reports_image_regions() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    risk = FactoryColorObstacleDetector().estimate(rgb)
    assert risk.obstacle_visible
    assert risk.right_fraction > risk.left_fraction
    assert risk.center_fraction > 0.0


def test_color_detector_recognizes_yellow_and_dark_surfaces() -> None:
    detector = FactoryColorObstacleDetector()
    yellow = np.full((60, 90, 3), 150, dtype=np.uint8)
    yellow[:36, 30:60] = np.asarray([210, 180, 40], dtype=np.uint8)
    yellow_risk = detector.estimate(yellow)
    assert yellow_risk.yellow_fraction > 0.0
    assert yellow_risk.obstacle_visible

    dark = np.full((60, 90, 3), 150, dtype=np.uint8)
    dark[:36, 30:60] = np.asarray([35, 40, 45], dtype=np.uint8)
    dark_risk = detector.estimate(dark)
    assert dark_risk.dark_fraction > 0.0
    assert dark_risk.obstacle_visible


def test_factory_navigation_matches_point_goal_on_clear_rgb() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    observation = _observation_with_rgb((2.0, 0.0), rgb)
    baseline = PointGoalNavigator().act(observation)
    visual = FactoryInspectionNavigator().act(observation)
    assert visual == baseline


def test_factory_navigation_slows_and_steers_to_clear_side() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    navigator = FactoryInspectionNavigator()
    command = navigator.act(_observation_with_rgb((2.0, 0.0), rgb))
    assert command.walk_enabled
    assert command.forward_mps < navigator.limits.max_forward_mps
    assert command.lateral_mps > 0.0
    assert command.yaw_rate_rps > 0.0
    assert navigator.intervention_count == 1


def test_factory_navigation_reduces_forward_speed_as_risk_increases() -> None:
    moderate_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    moderate_rgb[15:18, 30:45] = np.asarray([25, 25, 25], dtype=np.uint8)
    blocked_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    blocked_rgb[15:33, 30:60] = np.asarray([25, 25, 25], dtype=np.uint8)

    moderate = FactoryInspectionNavigator().act(
        _observation_with_rgb((2.0, 0.0), moderate_rgb)
    )
    blocked = FactoryInspectionNavigator().act(
        _observation_with_rgb((2.0, 0.0), blocked_rgb)
    )

    assert 0.05 < moderate.forward_mps <= 0.15
    assert blocked.forward_mps == 0.05
    assert blocked.forward_mps < moderate.forward_mps


def test_blocked_visual_obstacle_turns_without_reverse() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[15:33, 30:60] = np.asarray([25, 25, 25], dtype=np.uint8)
    navigator = FactoryInspectionNavigator()
    command = navigator.act(_observation_with_rgb((2.0, 1.0), rgb))

    assert command.walk_enabled
    assert command.forward_mps == navigator.visual.near_obstacle_forward_mps
    assert command.forward_mps == 0.05
    assert command.forward_mps >= 0.0
    assert command.lateral_mps != 0.0
    assert abs(command.yaw_rate_rps) == 0.65
    assert navigator.latest_mode == "visual_correction"
    assert navigator.avoidance_active


def test_factory_navigation_fades_continuously_back_to_path() -> None:
    obstacle_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    obstacle_rgb[:36, 45:, :] = np.asarray([70, 145, 130], dtype=np.uint8)
    clear_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    navigator = FactoryInspectionNavigator(
        visual=VisualAvoidanceConfig(
            clear_confirmation_steps=3,
            stale_frame_limit_steps=20,
        )
    )

    command = navigator.act(
        _observation_with_rgb((2.0, 0.0), obstacle_rgb, frame_index=1)
    )
    assert navigator.latest_mode == "visual_correction"
    assert command.forward_mps == 0.05

    first_clear = navigator.act(
        _observation_with_rgb((2.0, 0.0), clear_rgb, frame_index=2)
    )
    second_clear = navigator.act(
        _observation_with_rgb((2.0, 0.0), clear_rgb, frame_index=3)
    )
    assert navigator.latest_mode == "return_to_path"
    assert navigator.avoidance_active
    assert navigator.avoidance_side == 1
    assert first_clear.yaw_rate_rps > second_clear.yaw_rate_rps > 0.0
    assert first_clear.lateral_mps > second_clear.lateral_mps > 0.0

    command = navigator.act(
        _observation_with_rgb((2.0, 0.0), clear_rgb, frame_index=4)
    )
    assert navigator.latest_mode == "follow_path"
    assert not navigator.avoidance_active
    assert navigator.avoidance_side == 0
    assert command == PointGoalNavigator().act(
        _observation_with_rgb((2.0, 0.0), clear_rgb, frame_index=4)
    )


def test_waypoint_change_preserves_active_avoidance_direction() -> None:
    obstacle_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    obstacle_rgb[:36, 45:] = np.asarray([70, 145, 130], dtype=np.uint8)
    clear_rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    navigator = FactoryInspectionNavigator()

    previous = navigator.act(
        _observation_with_rgb((2.0, 0.0), obstacle_rgb, frame_index=1)
    )
    assert navigator.avoidance_side == 1
    assert previous.yaw_rate_rps > 0.0
    assert navigator.intervention_count == 1

    navigator.on_waypoint_changed()
    current = navigator.act(
        _observation_with_rgb((2.0, -1.0), clear_rgb, frame_index=2)
    )

    assert navigator.latest_mode == "return_to_path"
    assert navigator.avoidance_active
    assert navigator.avoidance_side == 1
    assert current.yaw_rate_rps > 0.0
    assert navigator.intervention_count == 2


def test_collision_recovery_steers_away_from_occupied_image_edge() -> None:
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    rgb[15:33, 60:90] = np.asarray([25, 25, 25], dtype=np.uint8)
    navigator = FactoryInspectionNavigator()

    route_command = navigator.act(
        _observation_with_rgb((2.0, -1.0), rgb, frame_index=1)
    )
    assert route_command.yaw_rate_rps < 0.0
    assert navigator.latest_risk is not None
    assert navigator.latest_risk.center_fraction == 0.0
    assert navigator.latest_risk.right_fraction > 0.0

    escape = navigator.recovery_command(route_command, forward_mps=0.30)
    assert escape.forward_mps == 0.30
    assert escape.lateral_mps == 0.15
    assert escape.yaw_rate_rps == 0.65
    assert navigator.latest_mode == "collision_escape"


def test_factory_navigation_waits_for_first_rgb_frame() -> None:
    navigator = FactoryInspectionNavigator()
    command = navigator.act(_observation((3.0, 0.0)))
    assert command == VelocityCommand(walk_enabled=True)


def test_left_and_right_routes_advance_to_shared_inspection_point() -> None:
    routes = (
        (
            (9.0, 3.5),
            (10.0, 3.5),
            (13.0, 7.1),
            (19.2, 4.3),
        ),
        ((13.0, -5.0), (19.2, 4.3)),
    )
    for points in routes:
        route = WaypointRoute(
            points,
            waypoint_tolerance_m=0.40,
            passage_tolerance_m=0.40,
            goal_tolerance_m=0.20,
        )
        for index, point in enumerate(points):
            update = route.update(point)
            if index < len(points) - 1:
                assert update.advanced
                assert not update.completed
                np.testing.assert_allclose(route.current_goal, points[index + 1])
            else:
                assert update.completed
        assert route.completed


def test_route_advances_when_robot_passes_near_intermediate_waypoint() -> None:
    route = WaypointRoute(
        ((1.0, 0.0), (2.0, 0.0)),
        waypoint_tolerance_m=0.20,
        passage_tolerance_m=0.40,
        goal_tolerance_m=0.10,
    )
    route.update((0.0, 0.0))

    update = route.update((1.1, 0.3))

    assert update.advanced
    assert not update.completed
    np.testing.assert_allclose(route.current_goal, (2.0, 0.0))


def test_command_limiter_drops_old_segment_steering_at_waypoint() -> None:
    limiter = CommandLimiter()
    old_segment = VelocityCommand(
        forward_mps=0.60,
        lateral_mps=0.15,
        yaw_rate_rps=0.65,
        walk_enabled=True,
    )
    for _ in range(10):
        limiter.apply(old_segment)

    limiter.prepare_waypoint_transition(max_forward_mps=0.25)
    assert limiter.previous == VelocityCommand(forward_mps=0.25, walk_enabled=True)

    new_segment = limiter.apply(
        VelocityCommand(
            forward_mps=0.25,
            lateral_mps=-0.15,
            yaw_rate_rps=-0.65,
            walk_enabled=True,
        )
    )
    assert new_segment.forward_mps == 0.25
    assert np.isclose(new_segment.lateral_mps, -0.04)
    assert np.isclose(new_segment.yaw_rate_rps, -0.13)


def test_command_limiter_caps_corner_speed_without_resetting_yaw() -> None:
    limiter = CommandLimiter()
    old_segment = VelocityCommand(
        forward_mps=0.60,
        lateral_mps=0.12,
        yaw_rate_rps=0.50,
        walk_enabled=True,
    )
    for _ in range(10):
        limiter.apply(old_segment)

    limiter.cap_forward_speed(max_forward_mps=0.20)

    assert limiter.previous == VelocityCommand(
        forward_mps=0.20,
        lateral_mps=0.12,
        yaw_rate_rps=0.50,
        walk_enabled=True,
    )


def test_simulation_gait_clock_ignores_wall_clock_and_pauses_while_standing() -> None:
    clock = SimulationGaitClock(gait_period_s=0.8)

    assert clock.update(0.0, walking=False) == 0.0
    assert clock.update(1.0, walking=False) == 0.0
    assert np.isclose(clock.update(1.2, walking=True), 0.25)
    assert np.isclose(clock.update(1.2, walking=True), 0.25)
    assert np.isclose(clock.update(1.4, walking=True), 0.50)


def _safety_monitor() -> NavigationSafetyMonitor:
    return NavigationSafetyMonitor(
        fall_confirmation_steps=3,
        collision_confirmation_steps=3,
        collision_recovery_steps=4,
        collision_escape_steps=6,
        collision_recovery_grace_steps=3,
    )


def test_transient_collision_is_debounced_without_stopping() -> None:
    monitor = _safety_monitor()
    contact = monitor.update(fallen=False, collision="hand <-> table")
    clear = monitor.update(fallen=False, collision=None)

    assert not contact.stop_active
    assert not contact.terminal
    assert not clear.stop_active
    assert not clear.recovered


def test_persistent_collision_requires_clear_hold_before_resume() -> None:
    monitor = _safety_monitor()
    for _ in range(3):
        blocked = monitor.update(fallen=False, collision="torso <-> table")
    assert blocked.stop_active
    assert blocked.collision_steps == 3

    for _ in range(3):
        waiting = monitor.update(fallen=False, collision=None)
        assert waiting.stop_active
    recovered = monitor.update(fallen=False, collision=None)
    assert not recovered.stop_active
    assert recovered.recovered
    assert recovered.recovery_grace_steps == 3


def test_persistent_collision_forces_bounded_escape_recovery() -> None:
    monitor = _safety_monitor()
    for _ in range(3):
        blocked = monitor.update(fallen=False, collision="torso <-> track")
    assert blocked.stop_active
    assert blocked.collision_hold_steps == 1

    for expected_hold_steps in range(2, 6):
        waiting = monitor.update(fallen=False, collision="torso <-> track")
        assert waiting.stop_active
        assert waiting.collision_hold_steps == expected_hold_steps

    recovered = monitor.update(fallen=False, collision="torso <-> track")
    assert not recovered.stop_active
    assert recovered.recovered
    assert recovered.collision_steps == 0
    assert recovered.collision_hold_steps == 0
    assert recovered.recovery_grace_steps == 3


def test_recovery_grace_cannot_be_interrupted_by_collision() -> None:
    monitor = _safety_monitor()
    for _ in range(3):
        monitor.update(fallen=False, collision="torso <-> table")
    for _ in range(4):
        recovered = monitor.update(fallen=False, collision=None)
    assert recovered.recovery_grace_steps == 3

    for expected_remaining in (2, 1, 0):
        grace = monitor.update(fallen=False, collision="hand <-> table")
        assert not grace.stop_active
        assert grace.recovery_grace_steps == expected_remaining
        assert grace.collision_steps == 0

    first = monitor.update(fallen=False, collision="hand <-> table")
    second = monitor.update(fallen=False, collision="hand <-> table")
    stopped = monitor.update(fallen=False, collision="hand <-> table")
    assert not first.stop_active
    assert not second.stop_active
    assert stopped.stop_active


def test_confirmed_fall_remains_terminal() -> None:
    monitor = _safety_monitor()
    first = monitor.update(fallen=True, collision=None)
    second = monitor.update(fallen=True, collision=None)
    confirmed = monitor.update(fallen=True, collision=None)
    still_stopped = monitor.update(fallen=False, collision=None)

    assert first.stop_active and not first.terminal
    assert second.stop_active and not second.terminal
    assert confirmed.stop_active and confirmed.terminal
    assert still_stopped.stop_active and still_stopped.terminal


def test_transient_fall_recovery_also_starts_collision_grace() -> None:
    monitor = _safety_monitor()
    suspected = monitor.update(fallen=True, collision=None)
    recovered = monitor.update(fallen=False, collision=None)
    contact_during_grace = monitor.update(
        fallen=False,
        collision="hand <-> table",
    )

    assert suspected.stop_active and not suspected.terminal
    assert recovered.recovered
    assert recovered.recovery_grace_steps == 3
    assert not contact_during_grace.stop_active


def test_disabled_camera_preview_is_a_safe_noop() -> None:
    preview = CameraPreviewWindow(enabled=False)
    preview.show(np.zeros((24, 32, 3), dtype=np.uint8), ["test"])
    preview.close()


def test_recovery_velocity_restarts_gait_without_losing_steering() -> None:
    requested = VelocityCommand(
        forward_mps=0.15,
        lateral_mps=0.12,
        yaw_rate_rps=0.40,
        walk_enabled=True,
    )
    recovered = recovery_velocity_command(requested, forward_mps=0.30)
    assert recovered == VelocityCommand(
        forward_mps=0.30,
        lateral_mps=0.12,
        yaw_rate_rps=0.40,
        walk_enabled=True,
    )
