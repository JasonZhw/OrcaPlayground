"""Deterministic gait phase driven by simulation time."""

from __future__ import annotations

import math


class SimulationGaitClock:
    """Accumulate walking time without depending on renderer wall-clock timing."""

    def __init__(self, gait_period_s: float) -> None:
        if not math.isfinite(gait_period_s) or gait_period_s <= 0.0:
            raise ValueError("gait_period_s must be finite and positive")
        self.gait_period_s = float(gait_period_s)
        self.reset()

    def reset(self) -> None:
        self._last_sim_time_s: float | None = None
        self._walking_elapsed_s = 0.0

    def update(self, sim_time_s: float, *, walking: bool) -> float:
        """Return phase in [0, 1), advancing only with simulated walking time."""
        if not math.isfinite(sim_time_s) or sim_time_s < 0.0:
            raise ValueError("sim_time_s must be finite and non-negative")
        if self._last_sim_time_s is None or sim_time_s < self._last_sim_time_s:
            if self._last_sim_time_s is not None:
                self._walking_elapsed_s = 0.0
            self._last_sim_time_s = float(sim_time_s)
            return 0.0

        elapsed_s = float(sim_time_s - self._last_sim_time_s)
        self._last_sim_time_s = float(sim_time_s)
        if walking:
            self._walking_elapsed_s += elapsed_s
        return (self._walking_elapsed_s % self.gait_period_s) / self.gait_period_s
