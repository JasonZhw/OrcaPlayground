"""Green-table RGB perception and local navigation policy for Demo 1."""

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
class GreenTableRiskEstimate:
    """Green-workbench occupancy measured in the navigable camera region."""

    risk_fraction: float
    left_fraction: float
    center_fraction: float
    right_fraction: float
    obstacle_visible: bool
    blocked: bool
    hard_stop: bool


class GreenTableDetector:
    """Detect current-layout green workbenches using only the 7072 RGB frame."""

    def __init__(self, config: VisualAvoidanceConfig | None = None) -> None:
        self.config = config or VisualAvoidanceConfig()

    def estimate(self, rgb: np.ndarray) -> GreenTableRiskEstimate:
        """Return per-region workbench occupancy, excluding the arm-heavy bottom."""
        # --------------------------------------------------------------
        # Perception block
        # 1. Crop the arm-heavy lower image.
        # 2. Build a deterministic green/cyan mask in RGB space.
        # 3. Measure occupancy in left, center, and right image thirds.
        # Only center occupancy blocks the direct path; side occupancy is
        # retained to decide which turning direction is clearer.
        # --------------------------------------------------------------
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
        # Match demo-01: only green in the center third blocks the current
        # route. Once the table moves to an outer third, target tracking must
        # regain control even though the table is still visible at the side.
        risk = center
        return GreenTableRiskEstimate(
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


class GreenTableNavigator:
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
        self.detector = GreenTableDetector(self.visual)
        self.latest_risk: GreenTableRiskEstimate | None = None
        self.latest_base_command = VelocityCommand.stopped()
        self.intervention_count = 0
        self.maximum_risk_fraction = 0.0
        self._avoidance_side = 0
        self._avoidance_active = False
        self.latest_mode = "waiting_rgb"

    @property
    def avoidance_side(self) -> int:
        """Current steering side: +1 is image/body left and -1 is right."""
        return self._avoidance_side

    @property
    def avoidance_active(self) -> bool:
        return self._avoidance_active

    def reset(self) -> None:
        self.goal_navigator.reset()
        self.latest_risk = None
        self.latest_base_command = VelocityCommand.stopped()
        self.intervention_count = 0
        self.maximum_risk_fraction = 0.0
        self._avoidance_side = 0
        self._avoidance_active = False
        self.latest_mode = "waiting_rgb"

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Follow the goal on clear ground and steer toward the clearer image side."""
        # --------------------------------------------------------------
        # Decision block
        # - Clear center: use the pose-based command unchanged.
        # - Green enters center: latch the clearer side and turn with a small
        #   forward gait (the frozen policy turns poorly at exactly vx=0).
        # - Center clears: immediately return control to point-goal tracking,
        #   which recomputes the command from the robot's current pose.
        # --------------------------------------------------------------
        base = self.goal_navigator.act(observation)
        risk = self.detector.estimate(observation.rgb)
        self.latest_base_command = base
        self.latest_risk = risk
        self.maximum_risk_fraction = max(self.maximum_risk_fraction, risk.risk_fraction)

        if not base.walk_enabled:
            self._avoidance_side = 0
            self._avoidance_active = False
            self.latest_mode = "goal_stop"
            return base

        if self._avoidance_active:
            if risk.risk_fraction <= self.visual.clear_risk_fraction:
                self._avoidance_active = False
                self._avoidance_side = 0
                self.latest_mode = "follow_target"
                return base
        elif risk.obstacle_visible:
            self._avoidance_side = self._choose_clear_side(
                risk,
                observation.goal_bearing_rad,
            )
            self._avoidance_active = True
        else:
            self.latest_mode = "follow_target"
            return base

        # The frozen locomotion policy cannot rotate reliably at vx=0. Keep a
        # small forward gait while turning until the center view is clear.
        self.latest_mode = "turn_clear"
        self.intervention_count += 1
        return VelocityCommand(
            forward_mps=min(
                self.visual.turn_forward_mps,
                self.limits.max_forward_mps,
            ),
            lateral_mps=0.0,
            yaw_rate_rps=float(
                self._avoidance_side * self.visual.avoidance_yaw_rate_rps
            ),
            walk_enabled=True,
        )

    def _choose_clear_side(self, risk: GreenTableRiskEstimate, goal_bearing_rad: float) -> int:
        difference = risk.left_fraction - risk.right_fraction
        if abs(difference) >= self.visual.clear_side_margin:
            return 1 if difference < 0.0 else -1
        return 1 if goal_bearing_rad >= 0.0 else -1
