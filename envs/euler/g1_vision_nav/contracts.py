"""Contracts between perception, navigation, and G1 locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    """High-level command in the G1 body frame.

    Positive x moves forward, positive y moves left, and positive yaw turns left.
    ``walk_enabled=False`` maps to the locomotion policy's stand mode.
    """

    forward_mps: float = 0.0
    lateral_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    walk_enabled: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.forward_mps, self.lateral_mps, self.yaw_rate_rps],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("velocity command values must be finite")
        if not self.walk_enabled and np.any(np.abs(values) > 1e-9):
            raise ValueError("a disabled walking command must contain zero velocities")

    @classmethod
    def stopped(cls) -> "VelocityCommand":
        return cls()

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.forward_mps, self.lateral_mps, self.yaw_rate_rps],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class NavigationObservation:
    """Observation consumed by a high-level visual navigator."""

    rgb: np.ndarray
    goal_xy_robot_m: np.ndarray
    base_velocity_xy_mps: np.ndarray
    base_yaw_rate_rps: float
    previous_command: VelocityCommand
    frame_index: int
    sim_time_s: float

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        goal = np.asarray(self.goal_xy_robot_m, dtype=np.float32).reshape(-1)
        velocity = np.asarray(self.base_velocity_xy_mps, dtype=np.float32).reshape(-1)

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_xy_robot_m must contain two finite values")
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("base_velocity_xy_mps must contain two finite values")
        if not np.isfinite(self.base_yaw_rate_rps) or not np.isfinite(self.sim_time_s):
            raise ValueError("yaw rate and simulation time must be finite")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")

        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "goal_xy_robot_m", goal.copy())
        object.__setattr__(self, "base_velocity_xy_mps", velocity.copy())

    @property
    def goal_distance_m(self) -> float:
        return float(np.linalg.norm(self.goal_xy_robot_m))

    @property
    def goal_bearing_rad(self) -> float:
        return float(np.arctan2(self.goal_xy_robot_m[1], self.goal_xy_robot_m[0]))


class VisualNavigator(Protocol):
    """Interface implemented by scripted, learned, or remote navigators."""

    def reset(self) -> None:
        """Reset recurrent policy state at the beginning of an episode."""
        ...

    def on_waypoint_changed(self) -> None:
        """Discard local steering state that belongs to the previous waypoint."""
        ...

    def recovery_command(
        self,
        requested: VelocityCommand,
        forward_mps: float,
    ) -> VelocityCommand:
        """Build a post-stop command that can escape a persistent contact."""
        ...

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Produce one body-frame velocity command."""
        ...
