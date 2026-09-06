"""Hierarchical point-goal navigation on the fixed ``g1_pick_usda`` asset."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from envs.euler.g1_vision_nav.camera_stream import CameraNotReadyError
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


class G1VisionNavEnv(G1CameraStreamEnv):
    """Drive G1 to a world-frame point while keeping the RGB pipeline active."""

    # 摘要约每 2 秒打印；数值检查保持原频率，状态与安全判断仍每个导航周期执行。
    NAVIGATION_LOG_INTERVAL = 100
    NAVIGATION_CHECK_INTERVAL = 50
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
        """Connect to the UI camera stream and announce the target."""
        super().before_loop(verifier)
        verifier.observe(
            "point_goal_navigation",
            f"[导航准备] 先站立约 {self.navigation_config.startup_stand_s:g} 秒仿真时间，"
            "收到 RGB 后自动出发。"
            f"目标坐标=({self.goal_xy_world[0]:.2f}, {self.goal_xy_world[1]:.2f}) 米，"
            f"终点朝向={self.navigation_config.goal_yaw_deg:+.1f}°"
            f"（允许误差 ±{self.navigation_config.goal_yaw_tolerance_deg:.1f}°），"
            f"位置容差={self.navigation_config.goal_tolerance_m:.2f} 米；"
            + ("碰撞急停已启用" if self.navigation_config.collision_stop else "碰撞急停未启用，仍保留跌倒保护"),
        )

    def compute_ctrl(self, step: int) -> np.ndarray:
        """Update navigation at its configured rate, then run the 50 Hz policy."""
        # Hierarchical control boundary:
        #   RGB + current pose -> navigation VelocityCommand
        #   VelocityCommand -> frozen G1 locomotion observation
        #   ONNX joint targets -> per-physics-step actuator control
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
        self._goal_reached = self.navigation_config.goal_reached(distance, state["base_yaw_world_rad"])

        was_emergency_stop = self._emergency_stop
        if self._has_fallen(state):
            self._emergency_stop = True
        if self.navigation_config.collision_stop:
            collision = self._find_unsafe_collision()
            if collision is not None:
                self._unsafe_collision = collision
                self._emergency_stop = True

        if self._emergency_stop and not was_emergency_stop:
            reason = self._unsafe_collision or "身体高度或倾角超限，判定为跌倒"
            verifier.check(
                f"navigation_emergency_stop_{step}", False, reason, "未触发急停",
                "[导航急停] 已触发保护；下一控制周期发送停止行走指令，请检查机器人状态后重新运行。",
            )

        if step % self.NAVIGATION_CHECK_INTERVAL == 0:
            verifier.check(
                f"goal_distance_finite_{step}", np.isfinite(distance), distance,
                "有效数值", f"目标距离检查（第 {step} 步，单位：米）",
            )
            verifier.check(
                f"navigation_safe_{step}", not self._emergency_stop,
                self._unsafe_collision or ("已触发急停" if self._emergency_stop else "未触发急停"),
                "未触发已启用的保护", f"导航保护检查（第 {step} 步）",
            )

        if step % self.NAVIGATION_LOG_INTERVAL != 0:
            return
        command = self.command_limiter.previous
        position = np.asarray(state["pelvis_position_world"], dtype=np.float64)
        if self._emergency_stop:
            status = "急停保护"
        elif step < self._startup_steps:
            status = "启动站立"
        elif not self.camera_motion_ready:
            status = "等待图像，暂停行走"
        elif self._goal_reached:
            status = "位置与朝向已达标"
        else:
            status = "导航中"
        verifier.observe(
            f"navigation_state_{step}",
            f"[导航] {status}｜第 {step} 步｜位置=({position[0]:.2f}, {position[1]:.2f}) 米｜"
            f"距目标={distance:.2f} 米｜朝向={math.degrees(state['base_yaw_world_rad']):+.1f}°｜"
            f"终点朝向误差={math.degrees(self.navigation_config.heading_error_rad(state['base_yaw_world_rad'])):+.1f}°\n"
            f"    下发指令：前进(vx)={command.forward_mps:.2f} m/s，"
            f"侧移(vy)={command.lateral_mps:.2f} m/s，转向={command.yaw_rate_rps:+.2f} rad/s（正左负右）",
            step=step,
        )

    def observe_step(self, step: int, verifier) -> None:
        """Navigation is autonomous and has no terminal keypress pauses."""
        return None

    def after_loop(self, verifier) -> None:
        """Save the last RGB sample and report the final point-goal state."""
        super().after_loop(verifier)
        state = self._read_planar_state()
        distance = float(np.linalg.norm(state["goal_xy_robot_m"]))
        minimum_distance = min(self._minimum_distance_m, distance)
        reached = self.navigation_config.goal_reached(distance, state["base_yaw_world_rad"])
        verifier.observe(
            "point_goal_final_state",
            f"[导航结束] 到达判定：{'已达标' if reached else '未达标'}｜"
            f"最终距离={distance:.2f} 米｜过程最近距离={minimum_distance:.2f} 米｜"
            f"终点朝向误差={math.degrees(self.navigation_config.heading_error_rad(state['base_yaw_world_rad'])):+.1f}°",
        )

    def verify_final(self, verifier) -> None:
        """Require measurable progress, goal arrival, RGB, and no safety stop."""
        state = self._read_planar_state()
        final_distance = float(np.linalg.norm(state["goal_xy_robot_m"]))
        initial_distance = self._initial_distance_m
        minimum_distance = min(self._minimum_distance_m, final_distance)
        progress = 0.0 if initial_distance is None else initial_distance - minimum_distance
        required_progress = math.inf if initial_distance is None else max(0.20, initial_distance * 0.30)
        verifier.check(
            "point_goal_progress",
            progress >= required_progress,
            progress,
            f"至少靠近目标 {required_progress:.2f} 米",
            "G1 朝目标产生有效位移",
        )
        verifier.check(
            "point_goal_reached",
            self.navigation_config.goal_reached(final_distance, state["base_yaw_world_rad"]),
            {"distance_m": final_distance, "yaw_error_deg": math.degrees(
                self.navigation_config.heading_error_rad(state["base_yaw_world_rad"]))},
            f"距离不超过 {self.navigation_config.goal_tolerance_m:.2f} 米，且"
            f"朝向误差不超过 {self.navigation_config.goal_yaw_tolerance_deg:.1f}°",
            "G1 到达目标位置且朝向符合要求",
        )
        verifier.check(
            "navigation_no_emergency_stop",
            not self._emergency_stop,
            self._unsafe_collision or ("已触发急停" if self._emergency_stop else "未触发急停"),
            "未触发已启用的保护",
            "导航保护结果；" + (
                "跌倒与碰撞急停均已启用"
                if self.navigation_config.collision_stop else "碰撞急停未启用，本项不证明全程无碰撞"
            ),
        )
        verifier.check(
            "navigation_rgb_available",
            self._latest_frame is not None,
            None if self._latest_frame is None else self._latest_frame.index,
            "已接收 RGB 图像（实际值为帧号）",
            f"导航环境获得 {self.camera_config.rgb_port} RGB",
        )

    def _requested_command(self, step: int) -> VelocityCommand:
        if step < self._startup_steps or self._emergency_stop:
            return VelocityCommand.stopped()
        if not self.camera_motion_ready:
            return VelocityCommand.stopped()

        observation = self._build_navigation_observation()
        self._latest_navigation_observation = observation
        self._goal_reached = self.navigation_config.goal_reached(
            observation.goal_distance_m, observation.base_yaw_world_rad,
        )
        # 让导航器也收到达标位姿，保留末端状态；轻微朝向漂移后继续
        # 对齐身体，而不是因环境提前 return 导致又去追逐目标坐标。
        requested = self.navigator.act(observation)
        return VelocityCommand.stopped() if self._goal_reached else requested

    def _build_navigation_observation(self) -> NavigationObservation:
        state = self._read_planar_state()
        if self._latest_frame is None:
            raise CameraNotReadyError("navigation requires a real RGB frame")
        rgb = self._latest_frame.image
        frame_index = self._latest_frame.index
        return NavigationObservation(
            rgb=rgb,
            goal_xy_robot_m=state["goal_xy_robot_m"],
            base_velocity_xy_mps=state["base_velocity_xy_mps"],
            base_yaw_rate_rps=state["base_yaw_rate_rps"],
            base_yaw_world_rad=state["base_yaw_world_rad"],
            previous_command=self.command_limiter.previous,
            frame_index=frame_index,
            sim_time_s=float(self.data.time),
        )

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
            "base_yaw_world_rad": float(np.arctan2(xmat[1, 0], xmat[0, 0])),
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
        state = self._read_planar_state()
        distance = float(np.linalg.norm(state["goal_xy_robot_m"]))
        previous_distance = self._reward_previous_distance_m
        progress = 0.0 if previous_distance is None else previous_distance - distance
        self._reward_previous_distance_m = distance
        reward = 10.0 * progress - 0.002
        if self.navigation_config.goal_reached(distance, state["base_yaw_world_rad"]) and not self._goal_bonus_awarded:
            reward += 5.0
            self._goal_bonus_awarded = True
        if self._unsafe_collision is not None:
            reward -= 5.0
        if self._emergency_stop and self._unsafe_collision is None:
            reward -= 10.0
        return float(reward)

    def _is_terminated(self, obs: dict) -> bool:
        return self._goal_reached or self._emergency_stop
