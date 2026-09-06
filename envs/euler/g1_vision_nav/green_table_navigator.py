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


class GreenTableDetector:
    """Detect current-layout green workbenches using only the 7072 RGB frame."""

    def __init__(self, config: VisualAvoidanceConfig | None = None) -> None:
        self.config = config or VisualAvoidanceConfig()

    def estimate(self, rgb: np.ndarray) -> GreenTableRiskEstimate:
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
        # Match demo-01: only green in the center third blocks the current
        # route. Side visibility alone does not block the route; the navigator
        # confirms clearance before gradually returning to target tracking.
        risk = center
        return GreenTableRiskEstimate(
            risk_fraction=risk,
            left_fraction=left,
            center_fraction=center,
            right_fraction=right,
            obstacle_visible=risk >= self.config.slow_risk_fraction,
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
        self._last_frame_index: int | None = None
        self._last_frame_time_s: float | None = None
        self._clear_elapsed_s = 0.0
        self._clear_seen = False
        self._return_elapsed_s: float | None = None
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
        self._last_frame_index = None
        self._last_frame_time_s = None
        self._reset_clearance()
        self.latest_mode = "waiting_rgb"

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Follow the goal on clear ground and steer toward the clearer image side."""
        base = self.goal_navigator.act(observation)
        risk = self.detector.estimate(observation.rgb)
        self.latest_base_command = base
        self.latest_risk = risk
        self.maximum_risk_fraction = max(self.maximum_risk_fraction, risk.risk_fraction)
        frame_dt = self._new_frame_interval(observation)

        if not base.walk_enabled:
            self._avoidance_side = 0
            self._avoidance_active = False
            self._reset_clearance()
            self.latest_mode = "goal_stop"
            return base

        if self._avoidance_active:
            if self._return_elapsed_s is not None and not risk.obstacle_visible:
                self._return_elapsed_s += frame_dt
                return self._return_to_target(base)
            if risk.risk_fraction <= self.visual.clear_risk_fraction:
                if self._clear_seen:
                    self._clear_elapsed_s += frame_dt
                self._clear_seen = True
                if self._clear_elapsed_s + 1e-9 >= self.visual.clear_confirm_s:
                    self._return_elapsed_s = 0.0
                    return self._return_to_target(base)
                self.latest_mode = "confirm_clear"
                self.intervention_count += 1
                return self._clearance_command(base)
            self._reset_clearance()
        elif risk.obstacle_visible:
            self._avoidance_side = self._choose_clear_side(
                risk,
                observation.goal_bearing_rad,
            )
            self._avoidance_active = True
            self._reset_clearance()
        else:
            self.latest_mode = (
                "align_goal" if self.goal_navigator.aligning_goal
                else "follow_target"
            )
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

    def _reset_clearance(self) -> None:
        self._clear_elapsed_s = 0.0
        self._clear_seen = False
        self._return_elapsed_s = None

    def _new_frame_interval(self, observation: NavigationObservation) -> float:
        """旧帧不累计时间；断流、帧号回退或仿真时钟回退后重新确认。"""
        now = observation.sim_time_s
        elapsed = 0.0 if self._last_frame_time_s is None else now - self._last_frame_time_s
        rewound = self._last_frame_index is not None and observation.frame_index < self._last_frame_index
        if elapsed < 0 or elapsed > self.visual.clear_frame_gap_s or rewound:
            self._reset_clearance()
            elapsed = 0.0
        if observation.frame_index == self._last_frame_index:
            return 0.0
        self._last_frame_index = observation.frame_index
        self._last_frame_time_s = now
        return elapsed

    def _clearance_command(self, base: VelocityCommand) -> VelocityCommand:
        # 清晰后先停止主动绕圈，而不是继续大角速度旋转；保持低速、不加速。
        return VelocityCommand(
            forward_mps=min(self.visual.turn_forward_mps, self.limits.max_forward_mps, base.forward_mps),
            lateral_mps=0.0,
            yaw_rate_rps=0.0,
            walk_enabled=True,
        )

    def _return_to_target(self, base: VelocityCommand) -> VelocityCommand:
        """逐渐恢复实时计算的目标指令；新障碍在 act 中优先打断此过程。"""
        assert self._return_elapsed_s is not None
        fraction = min(1.0, self._return_elapsed_s / self.visual.return_blend_s)
        if fraction >= 1.0 - 1e-9:
            self._avoidance_active = False
            self._avoidance_side = 0
            self._reset_clearance()
            self.latest_mode = "follow_target"
            return base
        self.latest_mode = "return_target"
        self.intervention_count += 1
        start = self._clearance_command(base)
        return VelocityCommand(
            forward_mps=(1.0 - fraction) * start.forward_mps + fraction * base.forward_mps,
            lateral_mps=fraction * base.lateral_mps,
            yaw_rate_rps=fraction * base.yaw_rate_rps,
            walk_enabled=True,
        )

    def _choose_clear_side(self, risk: GreenTableRiskEstimate, goal_bearing_rad: float) -> int:
        difference = risk.left_fraction - risk.right_fraction
        if abs(difference) >= self.visual.clear_side_margin:
            return 1 if difference < 0.0 else -1
        return 1 if goal_bearing_rad >= 0.0 else -1
