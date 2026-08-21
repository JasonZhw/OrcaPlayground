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
    """Turn toward a goal, then approach it with a conservative forward speed.

    This baseline intentionally ignores RGB pixels. It proves the point-goal and
    high-level-to-low-level command loop before a learned visual policy replaces
    it through the same ``VisualNavigator`` contract.
    """

    def __init__(
        self,
        navigation: NavigationConfig | None = None,
        limits: CommandLimits | None = None,
    ) -> None:
        self.navigation = navigation or NavigationConfig()
        self.limits = limits or CommandLimits()

    def reset(self) -> None:
        """The proportional baseline has no recurrent state."""
        return None

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Return a body-frame command for the current relative goal."""
        distance = observation.goal_distance_m
        if distance <= self.navigation.goal_tolerance_m:
            return VelocityCommand.stopped()

        bearing = observation.goal_bearing_rad
        yaw_rate = float(
            np.clip(
                self.navigation.bearing_gain * bearing,
                -self.limits.max_yaw_rate_rps,
                self.limits.max_yaw_rate_rps,
            )
        )
        if abs(bearing) >= self.navigation.turn_in_place_bearing_rad:
            return VelocityCommand(
                yaw_rate_rps=yaw_rate,
                walk_enabled=True,
            )

        slowdown_span = self.navigation.slowdown_radius_m - self.navigation.goal_tolerance_m
        speed_fraction = float(
            np.clip(
                (distance - self.navigation.goal_tolerance_m) / slowdown_span,
                0.0,
                1.0,
            )
        )
        forward = self.navigation.minimum_forward_mps + speed_fraction * (
            self.limits.max_forward_mps - self.navigation.minimum_forward_mps
        )
        forward *= max(0.0, float(np.cos(bearing)))
        lateral = float(
            np.clip(
                self.navigation.lateral_gain * observation.goal_xy_robot_m[1],
                -self.limits.max_lateral_mps,
                self.limits.max_lateral_mps,
            )
        )
        return VelocityCommand(
            forward_mps=float(forward),
            lateral_mps=lateral,
            yaw_rate_rps=yaw_rate,
            walk_enabled=True,
        )
