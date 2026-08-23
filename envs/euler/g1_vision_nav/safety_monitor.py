"""Debounced and recoverable navigation safety state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationSafetyStatus:
    """Current result of the fall/contact safety monitor."""

    stop_active: bool
    terminal: bool
    reason: str | None
    fall_steps: int
    collision_steps: int
    collision_hold_steps: int
    clear_steps: int
    recovery_grace_steps: int
    recovered: bool


class NavigationSafetyMonitor:
    """Stop immediately on risk without permanently latching brief events.

    A suspected fall pauses immediately. Non-foot collisions must persist for
    several navigation samples before pausing. After any recoverable stop, a
    grace window prevents collision chatter from immediately stopping the
    resumed command; fall detection remains active throughout that window.
    """

    def __init__(
        self,
        *,
        fall_confirmation_steps: int,
        collision_confirmation_steps: int,
        collision_recovery_steps: int,
        collision_escape_steps: int,
        collision_recovery_grace_steps: int,
    ) -> None:
        values = (
            fall_confirmation_steps,
            collision_confirmation_steps,
            collision_recovery_steps,
            collision_escape_steps,
            collision_recovery_grace_steps,
        )
        if min(values) <= 0:
            raise ValueError("safety confirmation and recovery steps must be positive")
        self.fall_confirmation_steps = fall_confirmation_steps
        self.collision_confirmation_steps = collision_confirmation_steps
        self.collision_recovery_steps = collision_recovery_steps
        self.collision_escape_steps = collision_escape_steps
        self.collision_recovery_grace_steps = collision_recovery_grace_steps
        self.reset()

    def reset(self) -> None:
        self._fall_steps = 0
        self._collision_steps = 0
        self._collision_hold_steps = 0
        self._clear_steps = 0
        self._collision_hold = False
        self._recovery_grace_steps = 0
        self._terminal = False
        self._terminal_reason: str | None = None
        self._last_collision: str | None = None
        self._previous_stop_active = False

    def update(
        self,
        *,
        fallen: bool,
        collision: str | None,
    ) -> NavigationSafetyStatus:
        """Consume one 10 Hz safety sample and return the current stop state."""
        if not self._terminal:
            self._fall_steps = self._fall_steps + 1 if fallen else 0
            suppress_collision_stop = self._recovery_grace_steps > 0
            if collision is None or suppress_collision_stop:
                self._collision_steps = 0
            else:
                self._collision_steps += 1
                self._last_collision = collision

            if suppress_collision_stop:
                self._recovery_grace_steps -= 1

            if self._fall_steps >= self.fall_confirmation_steps:
                self._terminal = True
                self._terminal_reason = (
                    f"confirmed fall ({self._fall_steps} consecutive samples)"
                )
                self._collision_hold = False
                self._collision_hold_steps = 0
                self._clear_steps = 0
            elif (
                not suppress_collision_stop
                and self._collision_steps >= self.collision_confirmation_steps
                and not self._collision_hold
            ):
                self._collision_hold = True
                self._collision_hold_steps = 0

            if self._collision_hold:
                self._collision_hold_steps += 1
            if self._collision_hold and not fallen:
                if collision is None:
                    self._clear_steps += 1
                else:
                    self._clear_steps = 0
                if (
                    self._clear_steps >= self.collision_recovery_steps
                    or self._collision_hold_steps >= self.collision_escape_steps
                ):
                    self._collision_hold = False
                    self._collision_hold_steps = 0
                    self._collision_steps = 0
                    self._clear_steps = 0
                    self._last_collision = None
                    self._recovery_grace_steps = self.collision_recovery_grace_steps
            elif self._collision_hold:
                self._clear_steps = 0

        stop_active = bool(
            self._terminal
            or fallen
            or self._collision_hold
        )
        recovered = self._previous_stop_active and not stop_active
        self._previous_stop_active = stop_active
        if recovered and not self._terminal and self._recovery_grace_steps == 0:
            self._recovery_grace_steps = self.collision_recovery_grace_steps

        if self._terminal:
            reason = self._terminal_reason
        elif fallen:
            reason = (
                f"fall confirmation {self._fall_steps}/"
                f"{self.fall_confirmation_steps}"
            )
        elif self._collision_hold:
            if collision is not None:
                reason = (
                    f"collision escape hold {self._collision_hold_steps}/"
                    f"{self.collision_escape_steps}: {collision}"
                )
            else:
                reason = (
                    f"collision clear hold {self._clear_steps}/"
                    f"{self.collision_recovery_steps}: {self._last_collision}"
                )
        elif self._recovery_grace_steps > 0:
            reason = f"recovery grace {self._recovery_grace_steps} steps"
        elif collision is not None:
            reason = (
                f"collision confirmation {self._collision_steps}/"
                f"{self.collision_confirmation_steps}: {collision}"
            )
        else:
            reason = None

        return NavigationSafetyStatus(
            stop_active=stop_active,
            terminal=self._terminal,
            reason=reason,
            fall_steps=self._fall_steps,
            collision_steps=self._collision_steps,
            collision_hold_steps=self._collision_hold_steps,
            clear_steps=self._clear_steps,
            recovery_grace_steps=self._recovery_grace_steps,
            recovered=recovered,
        )
