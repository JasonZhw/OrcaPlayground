"""Hierarchical point-goal navigation on the fixed ``g1_pick_usda`` asset."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from online_verifier import OnlineVerifier

from envs.euler.g1_vision_nav.camera_stream import CameraFrame
from envs.euler.g1_vision_nav.command_bridge import (
    CommandLimiter,
    NavigationObservation,
    VelocityCommand,
    VisualNavigator,
    apply_velocity_command,
)
from envs.euler.g1_vision_nav.config import (
    BluePanelInspectionConfig,
    CameraConfig,
    CommandLimits,
    NavigationConfig,
    TimingConfig,
)
from envs.euler.g1_vision_nav.factory_inspection_navigator import (
    BluePanelInspection,
    FactoryInspectionNavigator,
)
from envs.euler.g1_vision_nav.g1_camera_stream_env import (
    G1CameraStreamEnv,
)
from envs.euler.g1_vision_nav.point_goal_navigator import (
    PointGoalNavigator,
    WaypointRoute,
    world_goal_in_body_frame,
)
from envs.euler.g1_vision_nav.safety_monitor import (
    NavigationSafetyMonitor,
    NavigationSafetyStatus,
)


def format_turn_rate(rate: float) -> str:
    """只解释最终指令，不将 rad/s 混同于机器人当前朝向。"""
    if rate >= 0.005:
        direction = "左转/逆时针"
    elif rate <= -0.005:
        direction = "右转/顺时针"
    else:
        direction = "不转向"
    return f"{rate:+.2f} rad/s（{direction}）"


class G1VisionNavEnv(G1CameraStreamEnv):
    """Drive G1 to a world-frame point while keeping the RGB pipeline active."""

    NAVIGATION_LOG_INTERVAL = 100
    NAVIGATION_CHECK_INTERVAL = 50
    _SUPPORT_BODY_TOKENS = ("ankle_roll_link", "foot", "toe")

    def __init__(
        self,
        *args,
        goal_xy_world: tuple[float, float] | None = None,
        waypoints_xy_world: Sequence[Sequence[float]] | None = None,
        camera_config: CameraConfig,
        sample_path: Path,
        timing_config: TimingConfig | None = None,
        command_limits: CommandLimits | None = None,
        navigation_config: NavigationConfig | None = None,
        navigator: VisualNavigator | None = None,
        **kwargs,
    ) -> None:
        self.timing_config = timing_config or TimingConfig()
        self.command_limits = command_limits or CommandLimits()
        self.navigation_config = navigation_config or NavigationConfig()
        if (goal_xy_world is None) == (waypoints_xy_world is None):
            raise ValueError("provide exactly one of goal_xy_world or waypoints_xy_world")
        route_points = [goal_xy_world] if goal_xy_world is not None else waypoints_xy_world
        self.route = WaypointRoute(
            route_points,
            waypoint_tolerance_m=self.navigation_config.waypoint_tolerance_m,
            passage_tolerance_m=(
                self.navigation_config.waypoint_passage_tolerance_m
            ),
            goal_tolerance_m=self.navigation_config.goal_tolerance_m,
        )
        self.goal_xy_world = self.route.current_goal
        self.navigator = navigator or PointGoalNavigator(
            navigation=self.navigation_config,
            limits=self.command_limits,
        )
        self.command_limiter = CommandLimiter(
            limits=self.command_limits,
            navigation_hz=self.timing_config.navigation_hz,
        )
        control_period_s = kwargs.get("frame_skip", 20) * kwargs.get("time_step", 0.001)
        if control_period_s <= 0.0:
            raise ValueError("control period must be positive")
        self._navigation_stride = self.timing_config.locomotion_steps_per_navigation_step
        self._startup_steps = max(
            0,
            math.ceil(self.navigation_config.startup_stand_s / control_period_s),
        )
        self._initial_distance_m: float | None = None
        self._minimum_distance_m = math.inf
        self._reward_previous_distance_m: float | None = None
        self._goal_reached = False
        self._goal_bonus_awarded = False
        self._emergency_stop = False
        self._terminal_safety_stop = False
        self._unsafe_collision: str | None = None
        self._last_safety_reason: str | None = None
        self._safety_recovery_count = 0
        self._latest_safety_status: NavigationSafetyStatus | None = None
        self._latest_navigation_observation: NavigationObservation | None = None
        self.safety_monitor = NavigationSafetyMonitor(
            fall_confirmation_steps=self.navigation_config.fall_confirmation_steps,
            collision_confirmation_steps=self.navigation_config.collision_confirmation_steps,
            collision_recovery_steps=self.navigation_config.collision_recovery_steps,
            collision_escape_steps=self.navigation_config.collision_escape_steps,
            collision_recovery_grace_steps=(
                self.navigation_config.collision_recovery_grace_steps
            ),
        )
        super().__init__(
            *args,
            camera_config=camera_config,
            sample_path=sample_path,
            **kwargs,
        )

    def reset_model(self) -> tuple[dict, dict]:
        """Reset locomotion, navigation state, and point-goal metrics."""
        observation, info = super().reset_model()
        self.locomotion.reset()
        self.navigator.reset()
        self.command_limiter.reset()
        self._goal_reached = False
        self._goal_bonus_awarded = False
        self._emergency_stop = False
        self._terminal_safety_stop = False
        self._unsafe_collision = None
        self._last_safety_reason = None
        self._safety_recovery_count = 0
        self._latest_safety_status = None
        self.safety_monitor.reset()
        self._latest_navigation_observation = None
        self.route.reset()
        self.goal_xy_world = self.route.current_goal
        # The actor's world transform can still contain pre-publish data during
        # reset. Latch metrics after the first real simulation step instead.
        self._initial_distance_m = None
        self._minimum_distance_m = math.inf
        self._reward_previous_distance_m = None
        return observation, info

    def before_loop(self, verifier) -> None:
        """Start camera capture and announce the autonomous target."""
        super().before_loop(verifier)
        verifier.observe(
            "point_goal_navigation",
            "G1 将按顺序用当前位置、路径点和 base yaw 跟随直线路径："
            + " → ".join(
                f"({point[0]:.2f}, {point[1]:.2f})" for point in self.route.waypoints
            ),
        )

    def compute_ctrl(self, step: int) -> np.ndarray:
        """Update the 10 Hz navigator, then run the frozen 50 Hz ONNX policy."""
        # Hierarchical control boundary:
        #   live pose + RGB -> 10 Hz navigation VelocityCommand
        #   limited VelocityCommand -> frozen 50 Hz locomotion observation
        #   ONNX joint targets -> mixed G1 actuator control at physics rate
        if step % self._navigation_stride == 0:
            requested = self._requested_command(step)
            command = self.command_limiter.apply(
                requested,
                emergency_stop=self._emergency_stop or not self.camera_motion_ready,
            )
            apply_velocity_command(self.locomotion, command)
        return super().compute_ctrl(step)

    def verify_step(self, step: int, verifier) -> None:
        """Reuse camera/locomotion checks and add navigation safety metrics."""
        super().verify_step(step, verifier)
        if step % self._navigation_stride != 0:
            return

        state = self._read_planar_state()
        distance = float(np.linalg.norm(state["goal_xy_robot_m"]))
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)[:2]
        route_remaining = self.route.remaining_distance_m(position)
        if self._initial_distance_m is None:
            self._initial_distance_m = route_remaining
            self._minimum_distance_m = route_remaining
        else:
            self._minimum_distance_m = min(self._minimum_distance_m, route_remaining)

        collision = None
        if self.navigation_config.collision_stop:
            collision = self._find_unsafe_collision()
        safety = self.safety_monitor.update(
            fallen=self._has_fallen(state),
            collision=collision,
        )
        was_stopped = self._emergency_stop
        self._emergency_stop = safety.stop_active
        self._terminal_safety_stop = safety.terminal
        self._latest_safety_status = safety
        self._unsafe_collision = collision
        if safety.reason is not None:
            self._last_safety_reason = safety.reason
        if safety.stop_active and not was_stopped:
            verifier.observe(f"safety_stop_{step}", f"[安全] 暂停行走：{safety.reason}", step=step)
        if safety.recovered:
            self._safety_recovery_count += 1
            print(
                f"[安全] 暂停已解除，恢复路线（累计恢复 {self._safety_recovery_count} 次）。"
            )

        if step % self.NAVIGATION_CHECK_INTERVAL != 0:
            return
        verifier.check(
            f"goal_distance_finite_{step}",
            np.isfinite(distance),
            distance,
            "finite",
            f"点目标距离（step={step}）",
        )
        verifier.check(
            f"navigation_safe_{step}",
            not self._terminal_safety_stop,
            safety.reason or "safe",
            "no confirmed fall",
            f"导航安全状态（短暂接触可恢复，step={step}）",
        )
        if step % self.NAVIGATION_LOG_INTERVAL != 0 or self.route.completed:
            return
        command = self.command_limiter.previous
        if not self.camera_motion_ready:
            status = "等待图像"
        elif safety.stop_active:
            status = "安全暂停"
        elif self._goal_reached:
            status = "巡检完成"
        elif self.route.completed:
            status = "终点取景"
        else:
            status = "跟随路线"
        verifier.observe(
            f"navigation_state_{step}",
            f"[导航] {status} | 路径点 {self.route.current_index + 1}/{len(self.route.waypoints)} "
            f"| 位置 ({position[0]:.2f}, {position[1]:.2f}) | 距路径点 {distance:.2f} m "
            f"| 朝向 {math.degrees(state['base_heading_rad']):.0f}°\n"
            f"       指令：前进 {command.forward_mps:.2f}、侧移 {command.lateral_mps:.2f} m/s，"
            f"转向 {format_turn_rate(command.yaw_rate_rps)}",
            step=step,
        )

    def observe_step(self, step: int, verifier) -> None:
        """Navigation is autonomous and has no terminal keypress pauses."""
        return None

    def after_loop(self, verifier) -> None:
        """Save the last RGB sample and report the final point-goal state."""
        super().after_loop(verifier)
        distance = self._final_goal_distance_m()
        remaining = self._route_remaining_distance_m()
        minimum_distance = min(self._minimum_distance_m, remaining)
        verifier.observe(
            "point_goal_final_state",
            f"[结束] 距终点 {distance:.2f} m，最小剩余路线距离 {minimum_distance:.2f} m",
        )

    def verify_final(self, verifier) -> None:
        """Require measurable progress, goal arrival, RGB, and no safety stop."""
        final_distance = self._final_goal_distance_m()
        initial_distance = self._initial_distance_m
        remaining = self._route_remaining_distance_m()
        minimum_distance = min(self._minimum_distance_m, remaining)
        progress = 0.0 if initial_distance is None else initial_distance - minimum_distance
        required_progress = math.inf if initial_distance is None else max(0.20, initial_distance * 0.30)
        verifier.check(
            "point_goal_progress",
            progress >= required_progress,
            progress,
            f">={required_progress:.3f}",
            "G1 朝目标产生有效位移",
        )
        verifier.check(
            "point_goal_reached",
            self.arrival_confirmed,
            final_distance,
            "已进入最终到达范围",
            "G1 已到达最终路径点",
        )
        verifier.check(
            "inspection_photo_saved",
            self.route.completed
            and self._sample_saved
            and self.sample_path.is_file()
            and self.sample_path.stat().st_size > 100,
            str(self.sample_path),
            "non-empty PNG after final waypoint",
            "到达电气柜巡检点后保存 RGB 照片",
        )
        verifier.check(
            "navigation_no_emergency_stop",
            not self._terminal_safety_stop,
            self._last_safety_reason or "safe",
            "no confirmed fall",
            "导航期间未发生连续确认的跌倒；短暂安全暂停允许自动恢复",
        )
        verifier.observe(
            "navigation_safety_recoveries",
            f"短暂碰撞/倾斜后自动恢复次数：{self._safety_recovery_count}",
        )
        verifier.check(
            "navigation_rgb_available",
            self._latest_frame is not None,
            None if self._latest_frame is None else self._latest_frame.index,
            "camera frame index",
            "导航环境获得 UI 相机 RGB",
        )

    def _requested_command(self, step: int) -> VelocityCommand:
        if not self.camera_motion_ready:
            return VelocityCommand.stopped()
        if step == 0:
            return VelocityCommand(walk_enabled=True)
        if self._goal_reached:
            return VelocityCommand(walk_enabled=True)
        if step < self._startup_steps or self._emergency_stop:
            return VelocityCommand.stopped()

        state = self._read_planar_state()
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)[:2]
        previous_index = self.route.current_index
        update = self.route.update(position)
        if update.advanced:
            self.goal_xy_world = self.route.current_goal
            self.navigator.on_waypoint_changed()
            self.command_limiter.cap_forward_speed(
                self.navigation_config.waypoint_turning_forward_mps
            )
            print(
                f"[路线] 已到路径点 {previous_index + 1}，下一点 "
                f"({self.goal_xy_world[0]:.2f}, {self.goal_xy_world[1]:.2f})"
            )
        if update.completed:
            return self._arrival_command(state)

        observation = self._build_navigation_observation()
        self._latest_navigation_observation = observation
        requested = self.navigator.act(observation)
        safety = self._latest_safety_status
        if safety is not None and safety.recovery_grace_steps > 0:
            return self.navigator.recovery_command(
                requested,
                self.navigation_config.recovery_forward_mps,
            )
        return requested

    @property
    def arrival_tolerance_m(self) -> float:
        return self.navigation_config.goal_tolerance_m

    @property
    def arrival_confirmed(self) -> bool:
        return self.route.completed and self._final_goal_distance_m() <= self.arrival_tolerance_m

    def _arrival_command(self, state: dict) -> VelocityCommand:
        self._goal_reached = True
        return VelocityCommand(walk_enabled=True)

    def _build_navigation_observation(self) -> NavigationObservation:
        state = self._read_planar_state()
        if self._latest_frame is None:
            rgb = np.zeros(
                (self.camera_config.height, self.camera_config.width, 3),
                dtype=np.uint8,
            )
            frame_index = 0
        else:
            rgb = self._latest_frame.image
            frame_index = self._latest_frame.index
        return NavigationObservation(
            rgb=rgb,
            # 到点后由 route 按顺序切换；绕障后直接重新朝当前点走，不拉回旧线段。
            goal_xy_robot_m=state["goal_xy_robot_m"],
            base_velocity_xy_mps=state["base_velocity_xy_mps"],
            base_yaw_rate_rps=state["base_yaw_rate_rps"],
            previous_command=self.command_limiter.previous,
            frame_index=frame_index,
            sim_time_s=float(self.data.time),
        )

    def _camera_preview_lines(self, frame: CameraFrame) -> list[str]:
        lines = super()._camera_preview_lines(frame)
        command = self.command_limiter.previous
        observation = self._latest_navigation_observation
        distance_text = "waiting"
        pose_text = "waiting"
        bearing_text = "waiting"
        heading_text = "waiting"
        if observation is not None:
            distance_text = f"{self._goal_distance_m():.2f} m"
            state = self._read_planar_state()
            position = np.asarray(state["pelvis_position_world"], dtype=np.float64)
            pose_text = f"({position[0]:.2f}, {position[1]:.2f})"
            bearing_text = f"{observation.goal_bearing_rad:+.2f}"
            heading_text = f"{math.degrees(state['base_heading_rad']):+.0f}"
        safety = self._latest_safety_status
        if not self.camera_motion_ready:
            safety_text = "WAIT CAMERA"
        elif self._emergency_stop:
            safety_text = "STOP"
        elif safety is not None and safety.recovery_grace_steps > 0:
            grace_seconds = safety.recovery_grace_steps / self.timing_config.navigation_hz
            safety_text = f"RECOVER {grace_seconds:.1f}s"
        elif safety is not None and safety.collision_steps > 0:
            safety_text = (
                f"CAUTION {safety.collision_steps}/"
                f"{self.navigation_config.collision_confirmation_steps}"
            )
        else:
            safety_text = "GO"
        lines.extend(
            (
                f"Waypoint: {self.route.current_index + 1}/{len(self.route.waypoints)}"
                f" | distance: {distance_text}",
                f"Pose: {pose_text} | heading={heading_text} deg",
                f"Target error: {bearing_text} rad | +left/CCW, -right/CW",
                f"Command: vx={command.forward_mps:.2f} vy={command.lateral_mps:.2f}"
                f" yaw_rate={command.yaw_rate_rps:+.2f} rad/s",
                f"Safety: {safety_text}"
                f" | recoveries={self._safety_recovery_count}",
            )
        )
        if safety is not None and safety.reason is not None:
            lines.append(f"Safety reason: {safety.reason}")
        return lines

    def _read_planar_state(self) -> dict[str, np.ndarray | float]:
        pelvis_name = f"{self.agent_name}_pelvis"
        pelvis = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
        position = np.asarray(pelvis["xpos"], dtype=np.float64)
        xmat = np.asarray(pelvis["xmat"], dtype=np.float64).reshape(3, 3)
        goal_robot = world_goal_in_body_frame(
            position[:2],
            xmat,
            self.goal_xy_world,
        )

        base_joint_name = f"{self.agent_name}_floating_base_joint"
        base_qvel = self.query_joint_qvel([base_joint_name])[base_joint_name]
        linear_velocity_body = xmat.T @ np.asarray(base_qvel[:3], dtype=np.float64)
        angular_velocity_body = xmat.T @ np.asarray(base_qvel[3:6], dtype=np.float64)
        pitch = float(np.arcsin(np.clip(-xmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(xmat[2, 1], xmat[2, 2]))
        return {
            "pelvis_position_world": position,
            "goal_xy_robot_m": goal_robot,
            "goal_bearing_rad": float(np.arctan2(goal_robot[1], goal_robot[0])),
            "base_heading_rad": float(np.arctan2(xmat[1, 0], xmat[0, 0])),
            "base_velocity_xy_mps": linear_velocity_body[:2].astype(np.float32),
            "base_yaw_rate_rps": float(angular_velocity_body[2]),
            "pitch_rad": pitch,
            "roll_rad": roll,
        }

    def _goal_distance_m(self) -> float:
        state = self._read_planar_state()
        return float(np.linalg.norm(state["goal_xy_robot_m"]))

    def _final_goal_distance_m(self) -> float:
        state = self._read_planar_state()
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)[:2]
        return float(np.linalg.norm(self.route.final_goal - position))

    def _route_remaining_distance_m(self) -> float:
        state = self._read_planar_state()
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)[:2]
        return self.route.remaining_distance_m(position)

    def _should_save_rgb_sample(self) -> bool:
        """The final camera frame is an inspection result, not a debug sample."""
        return self.route.completed

    def _has_fallen(self, state: dict[str, np.ndarray | float]) -> bool:
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)
        tilt = max(abs(float(state["pitch_rad"])), abs(float(state["roll_rad"])))
        return bool(position[2] < self.navigation_config.fall_height_m or tilt >= self.navigation_config.fall_tilt_rad)

    def _find_unsafe_collision(self) -> str | None:
        robot_prefix = f"{self.agent_name}_"
        for contact in self.query_contact_simple():
            geom1_id = int(contact["geom1"])
            geom2_id = int(contact["geom2"])
            geom1 = self.model.get_geom_byid(geom1_id)
            geom2 = self.model.get_geom_byid(geom2_id)
            body1 = str(geom1.get("BodyName", ""))
            body2 = str(geom2.get("BodyName", ""))
            robot1 = body1.startswith(robot_prefix)
            robot2 = body2.startswith(robot_prefix)
            if robot1 == robot2:
                continue

            robot_id, environment_id = (geom1_id, geom2_id) if robot1 else (geom2_id, geom1_id)
            robot_body = body1 if robot1 else body2
            if any(token in robot_body for token in self._SUPPORT_BODY_TOKENS):
                continue
            return f"{self.model.geom_id2name(robot_id)} <-> {self.model.geom_id2name(environment_id)}"
        return None

    def _compute_reward(self, obs: dict, action: np.ndarray) -> float:
        distance = self._route_remaining_distance_m()
        previous_distance = self._reward_previous_distance_m
        progress = 0.0 if previous_distance is None else previous_distance - distance
        self._reward_previous_distance_m = distance
        reward = 10.0 * progress - 0.002
        if distance <= self.navigation_config.goal_tolerance_m and not self._goal_bonus_awarded:
            reward += 5.0
            self._goal_bonus_awarded = True
        if self._unsafe_collision is not None:
            reward -= 5.0
        if self._terminal_safety_stop:
            reward -= 10.0
        return float(reward)

    def _is_terminated(self, obs: dict) -> bool:
        return self._goal_reached or self._terminal_safety_stop


class G1FactoryInspectionEnv(G1VisionNavEnv):
    """Add RGB intervention and cabinet-arrival telemetry to route navigation."""

    def __init__(self, *args, navigator: FactoryInspectionNavigator,
                 inspection_config: BluePanelInspectionConfig | None = None, **kwargs) -> None:
        self.visual_navigator = navigator
        self.inspection = BluePanelInspection(inspection_config)
        self._inspection_last_log: str | None = None
        super().__init__(*args, navigator=navigator, **kwargs)

    def reset_model(self) -> tuple[dict, dict]:
        self.inspection.reset()
        self._inspection_last_log = None
        return super().reset_model()

    @property
    def arrival_tolerance_m(self) -> float:
        # 路线先进入配置的严格到达半径，再允许极小的原地取景漂移。
        return self.inspection.config.station_radius_m

    @property
    def arrival_confirmed(self) -> bool:
        # 工厂任务先到点再取景；转向微漂移不能抹掉已完成的到点判定。
        return self.route.completed and self.inspection.active

    def _requested_command(self, step: int) -> VelocityCommand:
        if self._sample_saved or self.inspection.failed:
            return VelocityCommand.stopped()
        if self.inspection.ready:
            return VelocityCommand.stopped()
        if self.inspection.active:
            # 在断流/接触期间也更新时间上限，不让取景无限等待。
            return self._arrival_command(self._read_planar_state())
        return super()._requested_command(step)

    def _arrival_command(self, state: dict) -> VelocityCommand:
        if not self.inspection.active:
            self.inspection.start(self.route.final_goal, float(self.data.time))
            print("[巡检] 已到最后导航点，先停稳；有电气柜就拍照，没有才旋转。")
        previous = self.command_limiter.previous
        command_still = (abs(previous.forward_mps) < .03 and abs(previous.lateral_mps) < .03
                         and abs(previous.yaw_rate_rps) < .03)
        return self.inspection.update(
            rgb=self._latest_frame.image if self.camera_motion_ready else None,
            frame_index=self._latest_frame.index if self.camera_motion_ready else -1,
            sim_time_s=float(self.data.time),
            position_xy=np.asarray(state['pelvis_position_world'])[:2],
            heading_rad=float(state['base_heading_rad']),
            linear_speed_mps=float(np.linalg.norm(state['base_velocity_xy_mps'])),
            yaw_rate_rps=float(state['base_yaw_rate_rps']),
            command_still=command_still,
            safe=not self._emergency_stop and not self._terminal_safety_stop,
        )

    def _should_save_rgb_sample(self) -> bool:
        # 工厂只在运行中的确认时刻保存；禁止 after_loop 另取一帧凑数。
        return False

    def _save_inspection_photo(self, verifier) -> None:
        if (not self.inspection.ready or self._sample_saved or self._emergency_stop
                or self._terminal_safety_stop):
            return
        photo = self.inspection.photo_rgb
        if photo is None:
            raise RuntimeError("电气柜画面已确认，但缺少对应图像，未保存照片")
        self._save_rgb_image(photo, verifier)
        self._goal_reached = True
        # 已经停稳才会拍照；成功后立刻清空残余指令并切为站立，不继续原地踏步。
        self.command_limiter.reset()
        apply_velocity_command(self.locomotion, VelocityCommand.stopped())
        verifier.observe("blue_panel_photo", "[巡检] 电气柜照片已保存，巡检完成。")

    def before_loop(self, verifier) -> None:
        super().before_loop(verifier)
        verifier.observe(
            "visual_avoidance_mode",
            "[避障] HSV 检测红/黄/绿/蓝，忽略低饱和度灰色区域；轻度风险只减前进速度，"
            f"保留目标转向。中央彩色占比达到 {self.visual_navigator.visual.blocked_risk_fraction:.0%} "
            "时优先避障，强避障后确认清晰再恢复巡航。",
        )
        verifier.observe(
            "blue_panel_inspection",
            "[巡检] 到达终点后停止移动，获取电气柜照片后结束巡检。",
        )

    def verify_step(self, step: int, verifier) -> None:
        super().verify_step(step, verifier)
        if self.inspection.active:
            # 此时安全检查已更新。若刚发生接触，废弃本轮候选，重新确认。
            if (self._emergency_stop or not self.camera_motion_ready) and self.inspection.ready and not self._sample_saved:
                self.inspection.invalidate_confirmation()
            self._save_inspection_photo(verifier)
            mode = self.inspection.mode
            if self.inspection.failed and mode != self._inspection_last_log:
                self._inspection_last_log = mode
                verifier.observe(
                    f"inspection_state_{step}",
                    f"[巡检] {mode}", step=step,
                )
            return  # 不再展示途中遗留的避障状态。
        if step % self.NAVIGATION_LOG_INTERVAL != 0:
            return
        risk = self.visual_navigator.latest_risk
        if risk is None:
            return
        base = self.visual_navigator.latest_base_command
        phase = {
            "visual_caution": "彩色提示减速（保留转向）",
            "visual_correction": "避障修正",
            "clear_confirm": "清晰确认",
            "return_to_path": "恢复目标转向",
            "follow_path": "跟随路线",
            "stale_rgb_stop": "图像停更",
        }.get(self.visual_navigator.latest_mode, self.visual_navigator.latest_mode)
        verifier.observe(
            f"visual_risk_{step}",
            f"[视觉] {phase} "
            f"| 障碍占比 左/中/右：{risk.left_fraction:.0%}/"
            f"{risk.center_fraction:.0%}/{risk.right_fraction:.0%} "
            f"| 路线要求 {format_turn_rate(base.yaw_rate_rps)}",
            step=step,
        )

    def verify_final(self, verifier) -> None:
        super().verify_final(verifier)
        verifier.check(
            "blue_panel_confirmed", self.inspection.ready and self._sample_saved,
            self.inspection.mode, "取得有效电气柜照片", "电气柜巡检拍照",
        )
        verifier.check(
            "visual_obstacle_observed",
            self.visual_navigator.maximum_risk_fraction >= self.visual_navigator.visual.slow_risk_fraction,
            self.visual_navigator.maximum_risk_fraction,
            f">={self.visual_navigator.visual.slow_risk_fraction}",
            "UI RGB 实际观察到红、黄、绿或蓝色候选障碍区域",
        )
        verifier.check(
            "visual_avoidance_intervened",
            self.visual_navigator.intervention_count > 0,
            self.visual_navigator.intervention_count,
            ">0",
            "视觉风险实际改变了点目标速度指令",
        )

    def _camera_preview_lines(self, frame: CameraFrame) -> list[str]:
        lines = super()._camera_preview_lines(frame)
        if self.inspection.active:
            status = "SAVED" if self._sample_saved else "FAILED" if self.inspection.failed else "SEARCH / ALIGN"
            lines.append(f"Inspection: {status}")
            return lines
        risk = self.visual_navigator.latest_risk
        if risk is None:
            lines.append(f"Avoidance: {self.visual_navigator.latest_mode} | risk: waiting")
        else:
            base = self.visual_navigator.latest_base_command
            lines.append(
                f"Avoidance: {self.visual_navigator.latest_mode}"
                f" | risk={risk.risk_fraction:.3f}"
                f" | side={self.visual_navigator.avoidance_side:+d}"
                f" | active={int(self.visual_navigator.avoidance_active)}"
            )
            lines.append(f"Route request: yaw_rate={base.yaw_rate_rps:+.2f} rad/s (before avoidance)")
        return lines


class FactoryReporter(OnlineVerifier):
    """保留官方报告格式，只调整本 Demo 的终端显示。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_events: set[str] = set()
        self.failed_checks: set[str] = set()

    def check(self, name, condition, actual=None, expected=None, detail=""):
        passed = bool(condition)
        self.checks.append({
            "name": name, "passed": passed,
            "actual": _json_value(actual), "expected": _json_value(expected),
            "detail": detail,
        })
        # 去除步号后按检查类型去重：异常初次出现立即提示，持续异常不刷屏。
        key = re.sub(r"_\d+$", "", name)
        if not passed and key not in self.failed_checks:
            print(f"[异常] {detail or name}；实际：{actual}，要求：{expected}")
            self.failed_checks.add(key)
        elif passed and key in self.failed_checks:
            print(f"[恢复] {detail or name}")
            self.failed_checks.remove(key)

    def observe(self, name, prompt, step=0):
        if name not in self.observed_events:
            self.observations.append({"name": name, "prompt": prompt, "step": step})
            self.observed_events.add(name)
        print(prompt if prompt.startswith("[") else f"[提示] {prompt}")


def _json_value(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
