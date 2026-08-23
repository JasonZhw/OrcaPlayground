"""Hierarchical point-goal navigation on the fixed ``g1_pick_usda`` asset."""

from __future__ import annotations

import math
from pathlib import Path

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
from envs.euler.g1_vision_nav.g1_camera_validation_env import (
    G1CameraValidationEnv,
)
from envs.euler.g1_vision_nav.point_goal_navigator import (
    PointGoalNavigator,
    world_goal_in_body_frame,
)


class G1VisionNavEnv(G1CameraValidationEnv):
    """Drive G1 to a world-frame point while keeping the RGB pipeline active."""

    NAVIGATION_LOG_INTERVAL = 50
    _SUPPORT_BODY_TOKENS = ("ankle_roll_link", "foot", "toe")

    def __init__(
        self,
        *args,
        goal_xy_world: tuple[float, float],
        camera_config: CameraConfig,
        sample_path: Path,
        timing_config: TimingConfig | None = None,
        command_limits: CommandLimits | None = None,
        navigation_config: NavigationConfig | None = None,
        navigator: VisualNavigator | None = None,
        **kwargs,
    ) -> None:
        self.goal_xy_world = np.asarray(goal_xy_world, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(self.goal_xy_world)):
            raise ValueError("goal_xy_world must contain two finite values")

        self.timing_config = timing_config or TimingConfig()
        self.command_limits = command_limits or CommandLimits()
        self.navigation_config = navigation_config or NavigationConfig()
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
            self._navigation_stride,
            math.ceil(self.navigation_config.startup_stand_s / control_period_s),
        )
        self._initial_distance_m: float | None = None
        self._minimum_distance_m = math.inf
        self._reward_previous_distance_m: float | None = None
        self._goal_reached = False
        self._goal_bonus_awarded = False
        self._emergency_stop = False
        self._unsafe_collision: str | None = None
        self._latest_navigation_observation: NavigationObservation | None = None
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
        self._unsafe_collision = None
        self._latest_navigation_observation = None
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
            "G1 将先原地站立等待相机，再以低速自动转向并前往目标："
            f"goal=({self.goal_xy_world[0]:.2f}, {self.goal_xy_world[1]:.2f})",
        )

    def compute_ctrl(self, step: int) -> np.ndarray:
        """Update the 10 Hz navigator, then run the frozen 50 Hz ONNX policy."""
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
        if self._initial_distance_m is None:
            self._initial_distance_m = distance
            self._minimum_distance_m = distance
        else:
            self._minimum_distance_m = min(self._minimum_distance_m, distance)
        if distance <= self.navigation_config.goal_tolerance_m:
            self._goal_reached = True

        if self._has_fallen(state):
            self._emergency_stop = True
        if self.navigation_config.collision_stop:
            collision = self._find_unsafe_collision()
            if collision is not None:
                self._unsafe_collision = collision
                self._emergency_stop = True

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
            not self._emergency_stop,
            self._unsafe_collision or "safe",
            "no fall or unsafe collision",
            f"导航安全状态（step={step}）",
        )
        command = self.command_limiter.previous
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)
        error_world = self.goal_xy_world - position[:2]
        verifier.observe(
            f"navigation_state_{step}",
            f"position=({position[0]:.3f}, {position[1]:.3f}), "
            f"error_world=({error_world[0]:+.3f}, {error_world[1]:+.3f}), "
            f"distance={distance:.3f}m, bearing={state['goal_bearing_rad']:.3f}rad, "
            f"command=({command.forward_mps:.3f}, {command.lateral_mps:.3f}, "
            f"{command.yaw_rate_rps:.3f})",
            step=step,
        )

    def observe_step(self, step: int, verifier) -> None:
        """Navigation is autonomous and has no terminal keypress pauses."""
        return None

    def after_loop(self, verifier) -> None:
        """Save the last RGB sample and report the final point-goal state."""
        super().after_loop(verifier)
        distance = self._goal_distance_m()
        minimum_distance = min(self._minimum_distance_m, distance)
        verifier.observe(
            "point_goal_final_state",
            f"目标导航结束：final_distance={distance:.3f}m, minimum_distance={minimum_distance:.3f}m",
        )

    def verify_final(self, verifier) -> None:
        """Require measurable progress, goal arrival, RGB, and no safety stop."""
        final_distance = self._goal_distance_m()
        initial_distance = self._initial_distance_m
        minimum_distance = min(self._minimum_distance_m, final_distance)
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
            final_distance <= self.navigation_config.goal_tolerance_m,
            final_distance,
            f"<={self.navigation_config.goal_tolerance_m}",
            "G1 到达目标容差范围并停止",
        )
        verifier.check(
            "navigation_no_emergency_stop",
            not self._emergency_stop,
            self._unsafe_collision or "safe",
            "safe",
            "导航期间未跌倒且未发生非足部环境碰撞",
        )
        verifier.check(
            "navigation_rgb_available",
            self._latest_frame is not None,
            None if self._latest_frame is None else self._latest_frame.index,
            "camera frame index",
            "导航环境同步获得 7072 RGB",
        )

    def _requested_command(self, step: int) -> VelocityCommand:
        if step < self._startup_steps or self._goal_reached or self._emergency_stop:
            return VelocityCommand.stopped()

        observation = self._build_navigation_observation()
        self._latest_navigation_observation = observation
        if observation.goal_distance_m <= self.navigation_config.goal_tolerance_m:
            self._goal_reached = True
            return VelocityCommand.stopped()
        return self.navigator.act(observation)

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
        observation = self._latest_navigation_observation
        command = self.command_limiter.previous
        state = self._read_planar_state()
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)
        distance = "waiting" if observation is None else f"{observation.goal_distance_m:.2f} m"
        bearing = "waiting" if observation is None else f"{observation.goal_bearing_rad:+.2f} rad"
        lines.extend(
            (
                f"Position: ({position[0]:.2f}, {position[1]:.2f})",
                f"Goal: ({self.goal_xy_world[0]:.2f}, {self.goal_xy_world[1]:.2f}) | distance={distance}",
                f"Bearing: {bearing}",
                f"Command: vx={command.forward_mps:.2f} vy={command.lateral_mps:.2f} yaw={command.yaw_rate_rps:.2f}",
            )
        )
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
            "base_velocity_xy_mps": linear_velocity_body[:2].astype(np.float32),
            "base_yaw_rate_rps": float(angular_velocity_body[2]),
            "pitch_rad": pitch,
            "roll_rad": roll,
        }

    def _goal_distance_m(self) -> float:
        state = self._read_planar_state()
        return float(np.linalg.norm(state["goal_xy_robot_m"]))

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
        distance = self._goal_distance_m()
        previous_distance = self._reward_previous_distance_m
        progress = 0.0 if previous_distance is None else previous_distance - distance
        self._reward_previous_distance_m = distance
        reward = 10.0 * progress - 0.002
        if distance <= self.navigation_config.goal_tolerance_m and not self._goal_bonus_awarded:
            reward += 5.0
            self._goal_bonus_awarded = True
        if self._unsafe_collision is not None:
            reward -= 5.0
        if self._emergency_stop and self._unsafe_collision is None:
            reward -= 10.0
        return float(reward)

    def _is_terminated(self, obs: dict) -> bool:
        return self._goal_reached or self._emergency_stop
