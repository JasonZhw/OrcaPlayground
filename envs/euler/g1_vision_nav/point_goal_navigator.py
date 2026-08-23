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
                - self.navigation.goal_tolerance_m
            )
            speed_fraction = float(
                np.clip(
                    (distance - self.navigation.goal_tolerance_m) / slowdown_span,
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
            # As in demo-01, path following uses forward+yaw. Lateral motion is
            # reserved for the temporary visual clearance correction.
            lateral_mps=0.0,
            yaw_rate_rps=yaw_rate,
            walk_enabled=True,
        )
