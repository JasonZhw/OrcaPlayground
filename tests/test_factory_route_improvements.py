"""Regression tests for segment following and fresh-frame obstacle release."""

import math
from dataclasses import replace

import numpy as np
import pytest

from envs.euler.g1_vision_nav.command_bridge import CommandLimiter, NavigationObservation, VelocityCommand
from envs.euler.g1_vision_nav.config import CommandLimits, VisualAvoidanceConfig
from envs.euler.g1_vision_nav.factory_inspection_navigator import FactoryInspectionNavigator
from envs.euler.g1_vision_nav.point_goal_navigator import WaypointRoute, world_goal_in_body_frame


def observation(frame, seconds, blocked=False):
    rgb = np.full((60, 90, 3), 120, dtype=np.uint8)
    if blocked:
        rgb[15:33, 45:] = [40, 140, 90]
    return NavigationObservation(rgb=rgb, goal_xy_robot_m=np.array([5.0, -2.0]),
        base_velocity_xy_mps=np.zeros(2), base_yaw_rate_rps=0.0,
        previous_command=VelocityCommand.stopped(), frame_index=frame, sim_time_s=seconds)


def test_route_keeps_current_waypoint_until_reached_without_nearest_point_skip():
    route = WaypointRoute(((5, 0), (5, 5)), waypoint_tolerance_m=.4,
        passage_tolerance_m=.4, goal_tolerance_m=.2)
    route.update((0, 0))
    # 下一点即使更近，也不能跳过当前待到达点。
    route.update((4., 4.))
    np.testing.assert_allclose(route.current_goal, (5, 0))
    assert route.current_index == 0
    assert route.update((5., 0.)).advanced
    np.testing.assert_allclose(route.current_goal, (5, 5))


def test_planar_distance_does_not_shrink_when_robot_pitches():
    pitch = .5
    rotation = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
    relative = world_goal_in_body_frame(np.zeros(2), rotation, np.array([2., 1.]))
    np.testing.assert_allclose(relative, [2., 1.])


def test_braking_faster_than_acceleration_but_not_instant():
    limits = CommandLimits()
    limiter = CommandLimiter(limits, navigation_hz=10)
    for _ in range(30):
        limiter.apply(VelocityCommand(forward_mps=.8, walk_enabled=True))
    before = limiter.previous.forward_mps
    after = limiter.apply(VelocityCommand(forward_mps=.05, walk_enabled=True)).forward_mps
    assert before == .8
    assert before - after == pytest.approx(limits.max_forward_decel_mps2 / 10)
    assert before - after > limits.max_forward_accel_mps2 / 10
    assert after > .05


def test_faster_cruise_keeps_turn_obstacle_and_stop_limits():
    nav = FactoryInspectionNavigator()
    clear = replace(observation(1, 0), goal_xy_robot_m=np.array([5., 0.]))
    assert nav.act(clear).forward_mps == 1.2
    turning = replace(clear, frame_index=2, goal_xy_robot_m=np.array([0., 5.]))
    assert nav.act(turning).forward_mps <= .25
    blocked = nav.act(observation(3, .2, blocked=True))
    assert blocked.forward_mps <= .18

    limiter = CommandLimiter()
    for _ in range(20):
        command = limiter.apply(VelocityCommand(forward_mps=1.5, walk_enabled=True))
    assert command.forward_mps == 1.2
    assert limiter.apply(VelocityCommand.stopped()) == VelocityCommand.stopped()


def test_clear_frame_must_be_confirmed_before_turning_back():
    nav = FactoryInspectionNavigator()
    nav.act(observation(1, 0, blocked=True))
    for i in range(2, 7):
        command = nav.act(observation(i, (i - 1) * .1))
        assert nav.avoidance_active
        assert command.yaw_rate_rps == 0
        assert command.forward_mps <= nav.visual.clear_forward_mps
    # 持续新帧，确认之后恢复转向，最后回到实时目标指令。
    for i in range(7, 22):
        command = nav.act(observation(i, (i - 1) * .1))
    assert not nav.avoidance_active
    assert command.yaw_rate_rps < 0


def test_repeated_frame_never_completes_clear_confirmation():
    nav = FactoryInspectionNavigator()
    nav.act(observation(1, 0, blocked=True))
    nav.act(observation(2, .1))
    for i in range(1, 9):
        nav.act(observation(2, .1 + i * .1))
    assert nav.avoidance_active


def test_long_gap_restarts_clear_confirmation():
    nav = FactoryInspectionNavigator()
    nav.act(observation(1, 0, blocked=True))
    nav.act(observation(2, .1))
    for i in range(3, 7):
        nav.act(observation(i, i * .1))
    cmd = nav.act(observation(7, 2.0))
    assert nav.avoidance_active and cmd.yaw_rate_rps == 0


def fraction_observation(frame, risk, seconds=None):
    """构造确切中央占比，复现实测 3.5%～4.3% 的边界抖动。"""
    rgb = np.full((100, 300, 3), 120, dtype=np.uint8)
    center = rgb[25:55, 100:200].copy().reshape(-1, 3)
    center[:round(risk * 3000)] = [40, 140, 90]
    rgb[25:55, 100:200] = center.reshape(30, 100, 3)
    return replace(observation(frame, frame * .1 if seconds is None else seconds),
                   rgb=rgb, goal_xy_robot_m=np.array([-2.29, -.12]))


@pytest.mark.parametrize('bearing', [-.05, -1.3, -3.0, 1.3])
@pytest.mark.parametrize('risk', [.08, .10, .15, .18, .19])
@pytest.mark.parametrize('already_avoiding', [False, True])
def test_mild_center_risk_preserves_full_route_turn(bearing, risk, already_avoiding):
    nav = FactoryInspectionNavigator()
    if already_avoiding:
        nav.act(replace(fraction_observation(1, .3), goal_xy_robot_m=np.array([1., 3.])))
        nav.on_waypoint_changed()
    obs = fraction_observation(2, risk)
    obs.rgb[25:55, 200:210] = [40, 140, 90]
    obs = replace(obs, goal_xy_robot_m=3 * np.array([math.cos(bearing), math.sin(bearing)]))
    command = nav.act(obs)
    assert command.yaw_rate_rps == nav.latest_base_command.yaw_rate_rps
    assert command.lateral_mps == nav.latest_base_command.lateral_mps
    if not already_avoiding:
        assert not nav.avoidance_active
    assert 0 < command.forward_mps <= nav.latest_base_command.forward_mps


def test_strong_center_risk_still_overrides_route_toward_clear_side():
    nav = FactoryInspectionNavigator()
    obs = fraction_observation(1, .25)
    obs.rgb[25:55, 200:230] = [40, 140, 90]
    command = nav.act(replace(obs, goal_xy_robot_m=np.array([1., -3.])))
    assert nav.latest_base_command.yaw_rate_rps < 0
    assert command.yaw_rate_rps > 0
    assert command.forward_mps <= nav.visual.near_obstacle_forward_mps


def test_mild_risk_then_clear_does_not_pause_waypoint_turn():
    nav = FactoryInspectionNavigator()
    nav.act(fraction_observation(1, .15))
    assert not nav.avoidance_active
    command = nav.act(fraction_observation(2, 0.))
    assert command == nav.latest_base_command
    assert nav.latest_mode == 'follow_path'


def test_waypoint_right_turn_recovers_despite_clear_threshold_flicker():
    route = WaypointRoute(((9., 3.5), (10., 3.5)), waypoint_tolerance_m=.4,
                          passage_tolerance_m=.4, goal_tolerance_m=.2)
    route.update((8., 0.))
    nav = FactoryInspectionNavigator()
    limiter = CommandLimiter()
    # 上一段先向左避障；到点后第二段需要右转，不能沿用左转修正。
    limiter.apply(nav.act(replace(fraction_observation(1, .3), goal_xy_robot_m=np.array([1., 3.5]))))
    assert nav.avoidance_side == 1
    assert route.update((9., 3.5)).advanced
    nav.on_waypoint_changed()
    limiter.cap_forward_speed(nav.navigation.waypoint_turning_forward_mps)
    yaw = math.atan2(3.5, 1.)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    relative = world_goal_in_body_frame(np.array([9., 3.5]), rotation, route.current_goal)
    for i in range(2, 42):
        obs = replace(fraction_observation(i, .043 if i % 5 == 0 else .035),
                      goal_xy_robot_m=relative, previous_command=limiter.previous)
        command = limiter.apply(nav.act(obs))
    assert not nav.avoidance_active
    assert nav.latest_mode == "follow_path"
    assert command.yaw_rate_rps == pytest.approx(-nav.limits.max_yaw_rate_rps)


def test_release_buffer_requires_clear_evidence_and_rejects_real_obstacles():
    nav = FactoryInspectionNavigator()
    nav.act(fraction_observation(1, .3))
    for i in range(2, 22):
        nav.act(fraction_observation(i, .05))
    assert nav.latest_mode == "visual_caution"  # 未见到 <=4%，保留强避障后的释放记录。
    nav.act(fraction_observation(22, .035))
    for i in range(23, 30):
        nav.act(fraction_observation(i, .05))
    assert nav.latest_mode == "return_to_path"
    nav.act(fraction_observation(30, .07))
    assert nav.latest_mode == "visual_caution"  # >6%，打断清晰确认，但低风险不削弱转向。
    command = nav.act(fraction_observation(31, .035))
    assert nav.latest_mode == "clear_confirm"
    assert command.yaw_rate_rps == 0


@pytest.mark.parametrize("seconds", [0.1, 2.0])
def test_frame_gap_invalidates_release_buffer(seconds):
    nav = FactoryInspectionNavigator()
    nav.act(fraction_observation(1, .3))
    nav.act(fraction_observation(2, .035))
    nav.act(fraction_observation(3, .05, seconds=seconds))
    assert nav.latest_mode == "visual_caution"


def test_buffered_repeated_frame_cannot_finish_clear_confirmation():
    nav = FactoryInspectionNavigator()
    nav.act(fraction_observation(1, .3))
    nav.act(fraction_observation(2, .035))
    nav.act(fraction_observation(3, .05))
    for i in range(4, 20):
        command = nav.act(fraction_observation(3, .05, seconds=i * .1))
    assert nav.avoidance_active
    assert nav.latest_mode == "stale_rgb_stop"
    assert command.yaw_rate_rps == command.forward_mps == 0


@pytest.mark.parametrize("value", [.03, .04, .08, float("nan")])
def test_invalid_release_buffer_rejected(value):
    with pytest.raises(ValueError):
        replace(VisualAvoidanceConfig(), clear_release_risk_fraction=value)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_clear_confirmation_rejected(value):
    with pytest.raises(ValueError):
        replace(VisualAvoidanceConfig(), clear_confirm_s=value)


def test_left_route_closed_loop_from_user_start_in_planar_model():
    """验证真实路径/导航/限速器组合，不将平面模型当成 G1 物理验收。"""
    from examples.euler.g1_vision_nav.run_factory_navi import ROUTES
    nav = FactoryInspectionNavigator()
    route = WaypointRoute(ROUTES["left"], waypoint_tolerance_m=.4,
        passage_tolerance_m=.4, goal_tolerance_m=nav.navigation.goal_tolerance_m)
    limiter = CommandLimiter(navigation_hz=10)
    position = np.array([8., 0.])
    yaw = 0.
    max_speed = 0.
    for step in range(2400):
        update = route.update(position)
        if update.completed:
            break
        if update.advanced:
            nav.on_waypoint_changed()
            limiter.cap_forward_speed(nav.navigation.waypoint_turning_forward_mps)
        target = route.current_goal
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obs = replace(observation(step + 1, step * .1),
            goal_xy_robot_m=world_goal_in_body_frame(position, rotation, target),
            previous_command=limiter.previous)
        command = limiter.apply(nav.act(obs))
        max_speed = max(max_speed, command.forward_mps)
        position += .1 * np.array([c * command.forward_mps - s * command.lateral_mps,
                                  s * command.forward_mps + c * command.lateral_mps])
        yaw += .1 * command.yaw_rate_rps
    assert route.completed, (position, route.current_index)
    assert np.linalg.norm(position - route.final_goal) <= .38
    assert 1.1 < max_speed <= 1.2
