"""Low-speed point-goal baseline for the hierarchical G1 demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.euler.g1_vision_nav.command_bridge import NavigationObservation, VelocityCommand
from envs.euler.g1_vision_nav.config import CommandLimits, NavigationConfig


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

    # 平面导航只使用 yaw。俯仰/横滚不应改变平面距离或到点判断。
    yaw = float(np.arctan2(body_xmat[1, 0], body_xmat[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    delta = goal_xy - robot_xy
    return np.asarray([c * delta[0] + s * delta[1],
                       -s * delta[0] + c * delta[1]], dtype=np.float32)


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


@dataclass(frozen=True)
class RouteUpdate:
    advanced: bool
    completed: bool


class WaypointRoute:
    """Advance through world-frame points using separate intermediate/final tolerances."""

    def __init__(
        self,
        waypoints_xy: Sequence[Sequence[float]],
        *,
        waypoint_tolerance_m: float,
        passage_tolerance_m: float,
        goal_tolerance_m: float,
    ) -> None:
        points = np.asarray(waypoints_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
            raise ValueError("waypoints_xy must have shape (N, 2) with N >= 1")
        if not np.all(np.isfinite(points)):
            raise ValueError("waypoints_xy must be finite")
        if min(waypoint_tolerance_m, passage_tolerance_m, goal_tolerance_m) <= 0.0:
            raise ValueError("route tolerances must be positive")
        if passage_tolerance_m < waypoint_tolerance_m:
            raise ValueError("passage_tolerance_m must cover waypoint_tolerance_m")
        self.waypoints = points.copy()
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.passage_tolerance_m = float(passage_tolerance_m)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.current_index = 0
        self.completed = False
        self._segment_start: np.ndarray | None = None

    @property
    def current_goal(self) -> np.ndarray:
        return self.waypoints[self.current_index].copy()

    @property
    def final_goal(self) -> np.ndarray:
        return self.waypoints[-1].copy()

    def reset(self) -> None:
        self.current_index = 0
        self.completed = False
        self._segment_start = None

    def update(self, robot_xy: Sequence[float]) -> RouteUpdate:
        """Advance when G1 reaches a waypoint or safely crosses its corridor."""
        # Intermediate points accept either entry into the arrival circle or
        # crossing the endpoint plane inside a narrow corridor. The latter
        # prevents a small overshoot from leaving the robot stuck forever.
        # The final inspection point never uses this fallback: G1 must enter
        # the stricter final arrival radius before the photo may be saved.
        position = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(position)):
            raise ValueError("robot_xy must be finite")
        if self._segment_start is None:
            self._segment_start = position.copy()
        advanced = False
        while not self.completed:
            distance = float(np.linalg.norm(self.current_goal - position))
            final = self.current_index == len(self.waypoints) - 1
            tolerance = self.goal_tolerance_m if final else self.waypoint_tolerance_m
            reached = distance <= tolerance + 1e-6
            passed = not final and self._passed_intermediate_waypoint(position)
            if not reached and not passed:
                break
            if final:
                self.completed = True
                break
            previous_goal = self.current_goal
            self.current_index += 1
            self._segment_start = previous_goal
            advanced = True
        return RouteUpdate(advanced=advanced, completed=self.completed)

    def _passed_intermediate_waypoint(self, position: np.ndarray) -> bool:
        if self._segment_start is None:
            return False
        segment = self.current_goal - self._segment_start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-12:
            return False
        relative = position - self._segment_start
        progress = float(np.dot(relative, segment) / length_squared)
        if progress < 1.0:
            return False
        cross_product = segment[0] * relative[1] - segment[1] * relative[0]
        cross_track = abs(float(cross_product)) / math.sqrt(length_squared)
        return cross_track <= self.passage_tolerance_m

    def remaining_distance_m(self, robot_xy: Sequence[float]) -> float:
        position = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        if self.completed:
            return float(np.linalg.norm(self.final_goal - position))
        remaining = float(np.linalg.norm(self.current_goal - position))
        for index in range(self.current_index, len(self.waypoints) - 1):
            remaining += float(
                np.linalg.norm(self.waypoints[index + 1] - self.waypoints[index])
            )
        return remaining
