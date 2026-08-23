"""Safe bridge from high-level navigation to the existing G1 policy."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from envs.euler.g1_vision_nav.config import CommandLimits
from envs.euler.g1_vision_nav.contracts import VelocityCommand


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
