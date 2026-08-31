"""Factory RGB obstacle perception and local waypoint-command correction."""

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
class FactoryObstacleRisk:
    """Colored obstacle occupancy measured in the navigable camera region."""

    risk_fraction: float
    left_fraction: float
    center_fraction: float
    right_fraction: float
    obstacle_visible: bool
    blocked: bool
    hard_stop: bool
    dark_fraction: float
    green_fraction: float
    yellow_fraction: float


class FactoryColorObstacleDetector:
    """Detect dark, green, and yellow obstacle surfaces in camera_head RGB."""

    def __init__(self, config: VisualAvoidanceConfig | None = None) -> None:
        self.config = config or VisualAvoidanceConfig()

    def estimate(self, rgb: np.ndarray) -> FactoryObstacleRisk:
        """Return per-region occupancy, excluding sky and the arm-heavy bottom."""
        # --------------------------------------------------------------
        # Perception block
        # 1. Keep the middle near-field strip because the head camera points
        #    downward and the lower image contains G1's hands and feet.
        # 2. Combine dark, green, and yellow masks for this fixed factory.
        # 3. Use center occupancy as immediate route risk; retain left/right
        #    occupancy only to select the clearer avoidance direction.
        # --------------------------------------------------------------
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        height, width, _ = image.shape
        roi_top = max(0, min(height - 1, round(height * self.config.roi_top_fraction)))
        roi_bottom = max(
            roi_top + 1,
            min(height, round(height * self.config.roi_bottom_fraction)),
        )
        roi = image[roi_top:roi_bottom].astype(np.int16)
        red, green, blue = roi[..., 0], roi[..., 1], roi[..., 2]
        dark_mask = np.maximum.reduce((red, green, blue)) <= self.config.dark_value_max
        green_mask = (
            (green >= self.config.green_minimum)
            & (green - red >= self.config.green_red_gap)
            & (green - blue >= self.config.green_blue_gap)
        )
        yellow_mask = (
            (red >= self.config.yellow_red_minimum)
            & (green >= self.config.yellow_green_minimum)
            & (blue <= self.config.yellow_blue_maximum)
            & (np.minimum(red, green) - blue >= self.config.yellow_blue_gap)
        )
        mask = dark_mask | green_mask | yellow_mask

        one_third = max(1, width // 3)
        two_thirds = min(width - 1, 2 * width // 3)
        left = self._fraction(mask[:, :one_third])
        center = self._fraction(mask[:, one_third:two_thirds])
        right = self._fraction(mask[:, two_thirds:])
        # Only the center third represents the immediate path in front of G1.
        # Dark shelving at the image edges is common in the warehouse and must
        # not activate avoidance by itself.  Left/right occupancy is retained
        # to choose which way to pass a center obstacle.
        risk = center
        return FactoryObstacleRisk(
            risk_fraction=risk,
            left_fraction=left,
            center_fraction=center,
            right_fraction=right,
            obstacle_visible=risk >= self.config.slow_risk_fraction,
            blocked=risk >= self.config.blocked_risk_fraction,
            hard_stop=risk >= self.config.hard_stop_risk_fraction,
            dark_fraction=self._fraction(dark_mask),
            green_fraction=self._fraction(green_mask),
            yellow_fraction=self._fraction(yellow_mask),
        )

    @staticmethod
    def _fraction(mask: np.ndarray) -> float:
        return 0.0 if mask.size == 0 else float(np.count_nonzero(mask) / mask.size)


class FactoryInspectionNavigator:
    """Blend RGB obstacle steering into the point-goal path follower.

    The selected avoidance side is latched until the center view has remained
    clear long enough.  There is deliberately no turn/pass/rejoin state
    machine: vision contributes one bounded correction to the route command.
    """

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
        self.detector = FactoryColorObstacleDetector(self.visual)
        self.latest_risk: FactoryObstacleRisk | None = None
        self.latest_base_command = VelocityCommand.stopped()
        self.intervention_count = 0
        self.maximum_risk_fraction = 0.0
        self._avoidance_side = 0
        self._avoidance_active = False
        self._clear_steps = 0
        self._last_frame_index = -1
        self._stale_frame_steps = 0
        self.latest_mode = "waiting_rgb"

    @property
    def avoidance_side(self) -> int:
        """Current steering side: +1 is image/body left and -1 is right."""
        return self._avoidance_side

    @property
    def avoidance_active(self) -> bool:
        """Whether a visual steering correction is currently latched."""
        return self._avoidance_active

    def reset(self) -> None:
        self.goal_navigator.reset()
        self.latest_risk = None
        self.latest_base_command = VelocityCommand.stopped()
        self.intervention_count = 0
        self.maximum_risk_fraction = 0.0
        self._avoidance_side = 0
        self._avoidance_active = False
        self._clear_steps = 0
        self._last_frame_index = -1
        self._stale_frame_steps = 0
        self.latest_mode = "waiting_rgb"

    def on_waypoint_changed(self) -> None:
        """Update the path target without interrupting obstacle avoidance."""
        self.goal_navigator.on_waypoint_changed()
        if not self._avoidance_active:
            self.latest_mode = "follow_path"

    def recovery_command(
        self,
        requested: VelocityCommand,
        forward_mps: float,
    ) -> VelocityCommand:
        """Use peripheral RGB occupancy to steer away from contact."""
        if forward_mps <= 0.0:
            raise ValueError("recovery forward speed must be positive")
        if not requested.walk_enabled:
            return requested
        risk = self.latest_risk
        if risk is None:
            return self.goal_navigator.recovery_command(requested, forward_mps)
        side_difference = risk.left_fraction - risk.right_fraction
        if abs(side_difference) < self.visual.clear_side_margin:
            return self.goal_navigator.recovery_command(requested, forward_mps)
        escape_side = 1 if side_difference < 0.0 else -1
        self._avoidance_side = escape_side
        self._avoidance_active = True
        self._clear_steps = 0
        self.latest_mode = "collision_escape"
        return VelocityCommand(
            forward_mps=forward_mps,
            lateral_mps=float(escape_side * self.visual.avoidance_lateral_mps),
            yaw_rate_rps=float(escape_side * self.visual.avoidance_yaw_rate_rps),
            walk_enabled=True,
        )

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Follow the path and continuously blend steering away from RGB risk."""
        # --------------------------------------------------------------
        # Navigation decision block
        # - The pose controller always recomputes a command to the current
        #   waypoint from the live G1 position and yaw.
        # - Visible center risk temporarily adds one latched steering vector.
        # - After consecutive clear frames, the correction fades to zero and
        #   the live waypoint command regains full control.
        # --------------------------------------------------------------
        base = self.goal_navigator.act(observation)
        self.latest_base_command = base
        if not base.walk_enabled:
            self.latest_mode = "goal_stop"
            return base
        if observation.frame_index <= 0:
            self.latest_mode = "waiting_rgb"
            return VelocityCommand(walk_enabled=True)
        if observation.frame_index == self._last_frame_index:
            self._stale_frame_steps += 1
        else:
            self._last_frame_index = observation.frame_index
            self._stale_frame_steps = 0
        if self._stale_frame_steps >= self.visual.stale_frame_limit_steps:
            self.latest_mode = "stale_rgb_stop"
            return VelocityCommand(walk_enabled=True)

        risk = self.detector.estimate(observation.rgb)
        self.latest_risk = risk
        self.maximum_risk_fraction = max(self.maximum_risk_fraction, risk.risk_fraction)

        if not self._avoidance_active:
            if not risk.obstacle_visible:
                self.latest_mode = "follow_path"
                return base
            self._avoidance_side = self._choose_clear_side(
                risk,
                observation.goal_bearing_rad,
            )
            self._avoidance_active = True
            self._clear_steps = 0

        if risk.risk_fraction > self.visual.clear_risk_fraction:
            self._clear_steps = 0
            severity = float(
                np.clip(
                    (risk.risk_fraction - self.visual.slow_risk_fraction)
                    / (
                        self.visual.blocked_risk_fraction
                        - self.visual.slow_risk_fraction
                    ),
                    0.0,
                    1.0,
                )
            )
            correction_weight = 0.65 + 0.35 * severity
            forward_mps = (
                self.visual.obstacle_approach_forward_mps
                - severity
                * (
                    self.visual.obstacle_approach_forward_mps
                    - self.visual.near_obstacle_forward_mps
                )
            )
            self.latest_mode = "visual_correction"
            self.intervention_count += 1
            return self._blend_with_avoidance(
                base,
                correction_weight=correction_weight,
                forward_mps=forward_mps,
                lateral_scale=0.50 + 0.50 * severity,
            )

        # Fade the same steering side over consecutive clear frames.  This
        # prevents an immediate turn back toward the obstacle while allowing
        # the route bearing to take control continuously.
        self._clear_steps += 1
        if self._clear_steps >= self.visual.clear_confirmation_steps:
            self._avoidance_active = False
            self._avoidance_side = 0
            self._clear_steps = 0
            self.latest_mode = "follow_path"
            return base
        release_fraction = self._clear_steps / self.visual.clear_confirmation_steps
        correction_weight = 1.0 - release_fraction
        forward_mps = (
            correction_weight * self.visual.clear_forward_mps
            + release_fraction * base.forward_mps
        )
        self.latest_mode = "return_to_path"
        self.intervention_count += 1
        return self._blend_with_avoidance(
            base,
            correction_weight=correction_weight,
            forward_mps=forward_mps,
            lateral_scale=1.0,
        )

    def _blend_with_avoidance(
        self,
        base: VelocityCommand,
        *,
        correction_weight: float,
        forward_mps: float,
        lateral_scale: float,
    ) -> VelocityCommand:
        """Blend one latched avoidance vector with the route command."""
        weight = float(np.clip(correction_weight, 0.0, 1.0))
        avoidance_yaw = self._avoidance_side * self.visual.avoidance_yaw_rate_rps
        avoidance_lateral = (
            self._avoidance_side
            * self.visual.avoidance_lateral_mps
            * float(np.clip(lateral_scale, 0.0, 1.0))
        )
        return VelocityCommand(
            forward_mps=float(max(0.0, forward_mps)),
            lateral_mps=float(
                (1.0 - weight) * base.lateral_mps + weight * avoidance_lateral
            ),
            yaw_rate_rps=float(
                (1.0 - weight) * base.yaw_rate_rps + weight * avoidance_yaw
            ),
            walk_enabled=True,
        )

    def _choose_clear_side(self, risk: FactoryObstacleRisk, goal_bearing_rad: float) -> int:
        difference = risk.left_fraction - risk.right_fraction
        if abs(difference) >= self.visual.clear_side_margin:
            return 1 if difference < 0.0 else -1
        return 1 if goal_bearing_rad >= 0.0 else -1
