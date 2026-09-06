"""Low-speed point-goal baseline for the hierarchical G1 demo."""

from __future__ import annotations

import numpy as np

from envs.euler.g1_vision_nav.config import CommandLimits, NavigationConfig
from envs.euler.g1_vision_nav.contracts import NavigationObservation, VelocityCommand


def world_goal_in_body_frame(
    robot_xy_world: np.ndarray,
    body_xmat_world: np.ndarray,
    goal_xy_world: np.ndarray,
) -> np.ndarray:
    """Transform a planar world-frame goal displacement into the robot frame."""
    robot_xy = np.asarray(robot_xy_world, dtype=np.float64).reshape(2)
    goal_xy = np.asarray(goal_xy_world, dtype=np.float64).reshape(2)
    body_xmat = np.asarray(body_xmat_world, dtype=np.float64).reshape(3, 3)
    if not all(np.all(np.isfinite(value)) for value in (robot_xy, goal_xy, body_xmat)):
        raise ValueError("robot pose and goal must be finite")

    delta_world = np.asarray(
        [goal_xy[0] - robot_xy[0], goal_xy[1] - robot_xy[1], 0.0],
        dtype=np.float64,
    )
    delta_body = body_xmat.T @ delta_world
    return delta_body[:2].astype(np.float32)


class PointGoalNavigator:
    """Recompute a body-forward command from the current pose and target."""

    def __init__(
        self,
        navigation: NavigationConfig | None = None,
        limits: CommandLimits | None = None,
    ) -> None:
        self.navigation = navigation or NavigationConfig()
        self.limits = limits or CommandLimits()
        self._aligning_goal = False

    @property
    def aligning_goal(self) -> bool:
        """终点区域内优先保持朝向调整，而不是反复追逐坐标。"""
        return self._aligning_goal

    def reset(self) -> None:
        """Clear the terminal heading latch at the beginning of an episode."""
        self._aligning_goal = False

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Return a body-frame command for the current relative goal."""
        # --------------------------------------------------------------
        # Point-goal block
        # The goal is transformed from world coordinates into the current
        # robot frame before this method is called. Therefore the command is
        # recalculated from the live G1 position and yaw on every navigation
        # cycle; it is not a prerecorded vx/vy sequence.
        # --------------------------------------------------------------
        distance = observation.goal_distance_m
        if self.navigation.goal_reached(distance, observation.base_yaw_world_rad):
            self._aligning_goal = True
            return VelocityCommand.stopped()
        if distance <= self.navigation.goal_alignment_radius_m:
            self._aligning_goal = True
        elif distance > self.navigation.goal_tolerance_m:
            self._aligning_goal = False
        if self._aligning_goal:
            # 用不同的进入/退出半径留出转身空间。0.8–1 m 间保持已进入的朝向调整，不再因小幅漂移切回点跟随；超过 1 m 才重新靠近。
            # 保留低速步态，避免冻结策略在 vx=0 时难以转身。
            error = self.navigation.heading_error_rad(observation.base_yaw_world_rad)
            rate = min(self.limits.max_yaw_rate_rps, max(
                self.navigation.goal_turn_min_rate_rps,
                self.navigation.bearing_gain * abs(error),
            ))
            return VelocityCommand(
                forward_mps=min(self.navigation.minimum_forward_mps, self.limits.max_forward_mps),
                yaw_rate_rps=float(np.copysign(rate, error)),
                walk_enabled=True,
            )

        bearing = observation.goal_bearing_rad
        yaw_rate = float(
            np.clip(
                self.navigation.bearing_gain * bearing,
                -self.limits.max_yaw_rate_rps,
                self.limits.max_yaw_rate_rps,
            )
        )
        if abs(bearing) >= self.navigation.turn_in_place_bearing_rad:
            heading_fraction = float(
                np.clip(
                    (np.pi - abs(bearing))
                    / (np.pi - self.navigation.turn_in_place_bearing_rad),
                    0.0,
                    1.0,
                )
            )
            turning_forward = self.navigation.minimum_forward_mps + heading_fraction * (
                self.navigation.turning_forward_mps
                - self.navigation.minimum_forward_mps
            )
            return VelocityCommand(
                forward_mps=min(turning_forward, self.limits.max_forward_mps),
                yaw_rate_rps=yaw_rate,
                walk_enabled=True,
            )

        x_forward = max(0.0, float(observation.goal_xy_robot_m[0]))
        forward = float(
            np.clip(
                self.navigation.forward_gain * x_forward,
                self.navigation.minimum_forward_mps,
                self.limits.max_forward_mps,
            )
        )
        if distance < self.navigation.slowdown_radius_m:
            slowdown_span = (
                self.navigation.slowdown_radius_m
                - self.navigation.goal_alignment_radius_m
            )
            speed_fraction = float(
                np.clip(
                    (distance - self.navigation.goal_alignment_radius_m) / slowdown_span,
                    0.0,
                    1.0,
                )
            )
            forward = self.navigation.minimum_forward_mps + speed_fraction * (
                forward - self.navigation.minimum_forward_mps
            )
        forward *= max(0.0, float(np.cos(bearing)))
        return VelocityCommand(
            forward_mps=float(forward),
            lateral_mps=0.0,
            yaw_rate_rps=yaw_rate,
            walk_enabled=True,
        )
