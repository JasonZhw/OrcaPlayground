"""Safe bridge from high-level navigation to the existing G1 policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from envs.euler.g1_vision_nav.config import CommandLimits


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


class G1LocomotionController(Protocol):
    """Public part of the existing ``G1Locomotion`` command API."""

    def set_commands(
        self,
        lin_vel: tuple[float, float] | None = None,
        ang_vel: float | None = None,
        base_height: float | None = None,
        stand: int | None = None,
    ) -> None:
        """Update the command consumed by the low-level ONNX policy."""
        ...


class CommandLimiter:
    """Clip velocity and apply a per-navigation-step slew-rate limit."""

    def __init__(self, limits: CommandLimits | None = None, navigation_hz: int = 10) -> None:
        if navigation_hz <= 0:
            raise ValueError("navigation_hz must be positive")
        self.limits = limits or CommandLimits()
        self.navigation_hz = navigation_hz
        self._previous = VelocityCommand.stopped()

    @property
    def previous(self) -> VelocityCommand:
        return self._previous

    def reset(self) -> None:
        self._previous = VelocityCommand.stopped()

    def prepare_waypoint_transition(self, max_forward_mps: float) -> None:
        """Drop stale lateral/yaw commands and cap momentum before a new segment."""
        if max_forward_mps <= 0.0:
            raise ValueError("waypoint transition speed must be positive")
        if not self._previous.walk_enabled:
            return
        self._previous = VelocityCommand(
            forward_mps=float(
                np.clip(self._previous.forward_mps, 0.0, max_forward_mps)
            ),
            walk_enabled=True,
        )

    def cap_forward_speed(self, max_forward_mps: float) -> None:
        """Limit corner-entry speed without discontinuously resetting steering."""
        if max_forward_mps <= 0.0:
            raise ValueError("forward speed cap must be positive")
        if not self._previous.walk_enabled:
            return
        self._previous = VelocityCommand(
            forward_mps=float(
                np.clip(self._previous.forward_mps, 0.0, max_forward_mps)
            ),
            lateral_mps=self._previous.lateral_mps,
            yaw_rate_rps=self._previous.yaw_rate_rps,
            walk_enabled=True,
        )

    def apply(self, requested: VelocityCommand, emergency_stop: bool = False) -> VelocityCommand:
        """Return a safe command; emergency or stand requests stop immediately."""
        if emergency_stop or not requested.walk_enabled:
            self.reset()
            return self._previous

        limits = self.limits
        clipped = np.asarray(
            [
                np.clip(requested.forward_mps, -limits.max_backward_mps, limits.max_forward_mps),
                np.clip(requested.lateral_mps, -limits.max_lateral_mps, limits.max_lateral_mps),
                np.clip(requested.yaw_rate_rps, -limits.max_yaw_rate_rps, limits.max_yaw_rate_rps),
            ],
            dtype=np.float64,
        )
        previous = self._previous.as_array().astype(np.float64)
        max_delta = np.asarray(
            [
                limits.max_forward_accel_mps2 / self.navigation_hz,
                limits.max_lateral_accel_mps2 / self.navigation_hz,
                limits.max_yaw_accel_rps2 / self.navigation_hz,
            ],
            dtype=np.float64,
        )
        # 遇障减速要比起步加速及时，但仍保留有限制动斜率，避免指令跳变。
        if 0.0 <= clipped[0] < previous[0]:
            max_delta[0] = limits.max_forward_decel_mps2 / self.navigation_hz
        limited = previous + np.clip(clipped - previous, -max_delta, max_delta)
        self._previous = VelocityCommand(
            forward_mps=float(limited[0]),
            lateral_mps=float(limited[1]),
            yaw_rate_rps=float(limited[2]),
            walk_enabled=True,
        )
        return self._previous


def apply_velocity_command(controller: G1LocomotionController, command: VelocityCommand) -> None:
    """Map the high-level command to ``G1Locomotion.set_commands`` semantics."""
    controller.set_commands(
        lin_vel=(command.forward_mps, command.lateral_mps),
        ang_vel=command.yaw_rate_rps,
        stand=int(command.walk_enabled),
    )


def recovery_velocity_command(
    requested: VelocityCommand,
    forward_mps: float,
) -> VelocityCommand:
    """Use a bounded forward launch speed while preserving steering."""
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
