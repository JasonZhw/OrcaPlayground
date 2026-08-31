"""Hierarchical point-goal navigation on the fixed ``g1_pick_usda`` asset."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from envs.euler.g1_vision_nav.camera_stream import CameraFrame
from envs.euler.g1_vision_nav.command_bridge import (
    CommandLimiter,
    apply_velocity_command,
)
from envs.euler.g1_vision_nav.config import (
    CameraConfig,
    CommandLimits,
    NavigationConfig,
    TimingConfig,
)
from envs.euler.g1_vision_nav.contracts import (
    NavigationObservation,
    VelocityCommand,
    VisualNavigator,
)
from envs.euler.g1_vision_nav.g1_camera_stream_env import (
    G1CameraStreamEnv,
)
from envs.euler.g1_vision_nav.point_goal_navigator import (
    PointGoalNavigator,
    world_goal_in_body_frame,
)
from envs.euler.g1_vision_nav.safety_monitor import (
    NavigationSafetyMonitor,
    NavigationSafetyStatus,
)
from envs.euler.g1_vision_nav.waypoint_route import WaypointRoute


class G1VisionNavEnv(G1CameraStreamEnv):
    """Drive G1 to a world-frame point while keeping the RGB pipeline active."""

    NAVIGATION_LOG_INTERVAL = 50
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
                emergency_stop=self._emergency_stop,
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
        self._emergency_stop = safety.stop_active
        self._terminal_safety_stop = safety.terminal
        self._latest_safety_status = safety
        self._unsafe_collision = collision
        if safety.reason is not None:
            self._last_safety_reason = safety.reason
        if safety.recovered:
            self._safety_recovery_count += 1
            print(
                "[INFO] Safety pause cleared; resuming route "
                f"(recoveries={self._safety_recovery_count})"
            )

        if step % self.NAVIGATION_LOG_INTERVAL != 0:
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
        command = self.command_limiter.previous
        verifier.observe(
            f"navigation_state_{step}",
            f"waypoint={self.route.current_index + 1}/{len(self.route.waypoints)}, "
            f"position=({position[0]:.3f}, {position[1]:.3f}), "
            f"heading={state['base_heading_rad']:.3f}rad, "
            f"distance={distance:.3f}m, remaining={route_remaining:.3f}m, "
            f"bearing={state['goal_bearing_rad']:.3f}rad, "
            f"command=({command.forward_mps:.3f}, {command.lateral_mps:.3f}, "
            f"{command.yaw_rate_rps:.3f}), safety_stop={safety.stop_active}, "
            f"safety_reason={safety.reason or 'none'}, "
            f"recovery_grace={safety.recovery_grace_steps}, "
            f"collision_samples={safety.collision_steps}, "
            f"collision_hold={safety.collision_hold_steps}, "
            f"recoveries={self._safety_recovery_count}",
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
            f"目标导航结束：final_distance={distance:.3f}m, minimum_distance={minimum_distance:.3f}m",
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
            self.route.completed and final_distance <= self.navigation_config.goal_tolerance_m,
            final_distance,
            f"<={self.navigation_config.goal_tolerance_m}",
            "G1 到达目标容差范围并停止",
        )
        verifier.check(
            "inspection_photo_saved",
            self.route.completed
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
            "导航环境同步获得 7072 RGB",
        )

    def _requested_command(self, step: int) -> VelocityCommand:
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
                f"[INFO] Waypoint {previous_index + 1} reached; "
                f"next=({self.goal_xy_world[0]:.2f}, {self.goal_xy_world[1]:.2f})"
            )
        if update.completed:
            self._goal_reached = True
            return VelocityCommand(walk_enabled=True)

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
        if observation is not None:
            distance_text = f"{observation.goal_distance_m:.2f} m"
            state = self._read_planar_state()
            position = np.asarray(state["pelvis_position_world"], dtype=np.float64)
            pose_text = f"({position[0]:.2f}, {position[1]:.2f})"
            bearing_text = f"{observation.goal_bearing_rad:+.2f}"
        safety = self._latest_safety_status
        if self._emergency_stop:
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
                f"Pose: {pose_text} | bearing={bearing_text}",
                f"Command: vx={command.forward_mps:.2f} vy={command.lateral_mps:.2f}"
                f" yaw={command.yaw_rate_rps:.2f}",
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
