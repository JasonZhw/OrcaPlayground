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
    """Follow a straight segment using body-forward speed and heading error."""

    def __init__(
        self,
        navigation: NavigationConfig | None = None,
        limits: CommandLimits | None = None,
    ) -> None:
        self.navigation = navigation or NavigationConfig()
        self.limits = limits or CommandLimits()
        self._aligning_after_waypoint = False

    def reset(self) -> None:
        """Clear waypoint-local heading alignment state."""
        self._aligning_after_waypoint = False

    def on_waypoint_changed(self) -> None:
        """Use a slower forward arc until the new segment heading is acquired."""
        self._aligning_after_waypoint = True

    def recovery_command(
        self,
        requested: VelocityCommand,
        forward_mps: float,
    ) -> VelocityCommand:
        """Restart the gait while preserving point-goal steering."""
        if forward_mps <= 0.0:
            raise ValueError("recovery forward speed must be positive")
        if not requested.walk_enabled:
            return requested
        return VelocityCommand(
            forward_mps=forward_mps,
            lateral_mps=requested.lateral_mps,
            yaw_rate_rps=requested.yaw_rate_rps,
            walk_enabled=True,
        )

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Return a body-frame command for the current relative goal."""
        distance = observation.goal_distance_m
        if distance <= self.navigation.goal_tolerance_m + 1e-6:
            return VelocityCommand.stopped()

        bearing = observation.goal_bearing_rad
        if (
            self._aligning_after_waypoint
            and abs(bearing) <= self.navigation.waypoint_heading_release_rad
        ):
            self._aligning_after_waypoint = False
        yaw_rate = float(
            np.clip(
                self.navigation.bearing_gain * bearing,
                -self.limits.max_yaw_rate_rps,
                self.limits.max_yaw_rate_rps,
            )
        )
        if (
            abs(bearing) >= self.navigation.turn_in_place_bearing_rad
            or self._aligning_after_waypoint
        ):
            # The locomotion policy turns poorly at exactly vx=0, so retain a
            # small forward arc.  Crucially, decay that arc toward the minimum
            # as the target moves behind G1; the old fixed 0.5 m/s command
            # could make the goal distance grow without bound after overshoot.
            forward_cap = (
                self.navigation.waypoint_turning_forward_mps
                if self._aligning_after_waypoint
                else self.navigation.turning_forward_mps
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
                forward_cap = self.navigation.minimum_forward_mps + heading_fraction * (
                    forward_cap - self.navigation.minimum_forward_mps
                )
            return VelocityCommand(
                forward_mps=min(forward_cap, self.limits.max_forward_mps),
                yaw_rate_rps=yaw_rate,
                walk_enabled=True,
            )

        x_forward = observation.goal_xy_robot_m[0]
        vx = float(
            np.clip(
                self.navigation.forward_gain * max(0.0, float(x_forward)),
                self.navigation.minimum_forward_mps,
                self.limits.max_forward_mps,
            )
        )
        if distance < self.navigation.slowdown_radius_m:
            slowdown = float(
                np.clip(
                    (distance - self.navigation.goal_tolerance_m)
                    / (self.navigation.slowdown_radius_m - self.navigation.goal_tolerance_m),
                    0.0,
                    1.0,
                )
            )
            vx = self.navigation.minimum_forward_mps + slowdown * (
                vx - self.navigation.minimum_forward_mps
            )
        vx *= max(0.0, float(np.cos(bearing)))
        return VelocityCommand(
            forward_mps=float(vx),
            # Route following is deliberately forward+yaw only.  Lateral
            # motion is reserved for the temporary visual safety correction.
            lateral_mps=0.0,
            yaw_rate_rps=yaw_rate,
            walk_enabled=True,
        )
