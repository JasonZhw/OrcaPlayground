"""RGB workbench perception and local avoidance for Demo 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.euler.g1_vision_nav.config import (
    CommandLimits,
    NavigationConfig,
    VisualAvoidanceConfig,
)
from envs.euler.g1_vision_nav.contracts import NavigationObservation, VelocityCommand
from envs.euler.g1_vision_nav.point_goal_navigator import PointGoalNavigator


@dataclass(frozen=True)
class VisualRiskEstimate:
    """Green-workbench occupancy measured in the navigable camera region."""

    risk_fraction: float
    left_fraction: float
    center_fraction: float
    right_fraction: float
    obstacle_visible: bool
    blocked: bool
    hard_stop: bool


class GreenWorkbenchDetector:
    """Detect current-layout green workbenches using only the 7072 RGB frame."""

    def __init__(self, config: VisualAvoidanceConfig | None = None) -> None:
        self.config = config or VisualAvoidanceConfig()

    def estimate(self, rgb: np.ndarray) -> VisualRiskEstimate:
        """Return per-region workbench occupancy, excluding the arm-heavy bottom."""
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        height, width, _ = image.shape
        roi_bottom = max(1, min(height, round(height * self.config.roi_bottom_fraction)))
        roi = image[:roi_bottom].astype(np.int16)
        red, green, blue = roi[..., 0], roi[..., 1], roi[..., 2]
        mask = (
            (green >= self.config.minimum_green)
            & (blue >= self.config.minimum_blue)
            & (green - red >= self.config.minimum_green_red_gap)
            & (blue - red >= self.config.minimum_blue_red_gap)
            & (green - blue >= self.config.minimum_green_blue_gap)
        )

        one_third = max(1, width // 3)
        two_thirds = min(width - 1, 2 * width // 3)
        left = self._fraction(mask[:, :one_third])
        center = self._fraction(mask[:, one_third:two_thirds])
        right = self._fraction(mask[:, two_thirds:])
        risk = self._fraction(mask)
        return VisualRiskEstimate(
            risk_fraction=risk,
            left_fraction=left,
            center_fraction=center,
            right_fraction=right,
            obstacle_visible=risk >= self.config.slow_risk_fraction,
            blocked=risk >= self.config.blocked_risk_fraction,
            hard_stop=risk >= self.config.hard_stop_risk_fraction,
        )

    @staticmethod
    def _fraction(mask: np.ndarray) -> float:
        return 0.0 if mask.size == 0 else float(np.count_nonzero(mask) / mask.size)


class VisualAvoidanceNavigator:
    """Wrap point-goal control with an RGB-driven workbench avoidance layer."""

    def __init__(
        self,
        *,
        navigation: NavigationConfig | None = None,
        limits: CommandLimits | None = None,
        visual: VisualAvoidanceConfig | None = None,
    ) -> None:
        self.navigation = navigation or NavigationConfig()
        self.limits = limits or CommandLimits()
        self.visual = visual or VisualAvoidanceConfig()
        self.goal_navigator = PointGoalNavigator(
            navigation=self.navigation,
            limits=self.limits,
        )
        self.detector = GreenWorkbenchDetector(self.visual)
        self.latest_risk: VisualRiskEstimate | None = None
        self.latest_base_command = VelocityCommand.stopped()
        self.intervention_count = 0
        self.maximum_risk_fraction = 0.0
        self._avoidance_side = 0
        self._clear_steps = 0

    @property
    def avoidance_side(self) -> int:
        """Current steering side: +1 is image/body left and -1 is right."""
        return self._avoidance_side

    def reset(self) -> None:
        self.goal_navigator.reset()
        self.latest_risk = None
        self.latest_base_command = VelocityCommand.stopped()
        self.intervention_count = 0
        self.maximum_risk_fraction = 0.0
        self._avoidance_side = 0
        self._clear_steps = 0

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Follow the goal on clear ground and steer toward the clearer image side."""
        base = self.goal_navigator.act(observation)
        risk = self.detector.estimate(observation.rgb)
        self.latest_base_command = base
        self.latest_risk = risk
        self.maximum_risk_fraction = max(self.maximum_risk_fraction, risk.risk_fraction)

        if not base.walk_enabled:
            self._avoidance_side = 0
            self._clear_steps = 0
            return base

        if risk.obstacle_visible:
            if self._avoidance_side == 0:
                self._avoidance_side = self._choose_clear_side(risk, observation.goal_bearing_rad)
            self._clear_steps = 0
        elif self._avoidance_side != 0:
            if risk.risk_fraction <= self.visual.clear_risk_fraction:
                self._clear_steps += 1
                if self._clear_steps >= self.visual.avoidance_hold_steps:
                    self._avoidance_side = 0
                    self._clear_steps = 0
            else:
                self._clear_steps = 0

        if self._avoidance_side == 0:
            return base

        self.intervention_count += 1
        severity = float(
            np.clip(
                (risk.risk_fraction - self.visual.slow_risk_fraction)
                / (self.visual.blocked_risk_fraction - self.visual.slow_risk_fraction),
                0.0,
                1.0,
            )
        )
        if risk.obstacle_visible:
            forward_limit = self.visual.approach_forward_mps + severity * (
                self.visual.crawl_forward_mps - self.visual.approach_forward_mps
            )
            forward = min(max(0.0, base.forward_mps), forward_limit)
            if risk.hard_stop:
                forward = 0.0
            lateral_scale = 0.70 + 0.30 * severity
            yaw_scale = 0.60 + 0.40 * severity
        else:
            # The workbench has just left the image. Continue diagonally past
            # its edge instead of immediately turning back toward the goal.
            forward = self.visual.approach_forward_mps
            lateral_scale = 1.0
            yaw_scale = 0.0
        return VelocityCommand(
            forward_mps=float(forward),
            lateral_mps=float(self._avoidance_side * self.visual.avoidance_lateral_mps * lateral_scale),
            yaw_rate_rps=float(self._avoidance_side * self.visual.avoidance_yaw_rate_rps * yaw_scale),
            walk_enabled=True,
        )

    def _choose_clear_side(self, risk: VisualRiskEstimate, goal_bearing_rad: float) -> int:
        difference = risk.left_fraction - risk.right_fraction
        if abs(difference) >= self.visual.clear_side_margin:
            return 1 if difference < 0.0 else -1
        return 1 if goal_bearing_rad >= 0.0 else -1
