"""Factory RGB obstacle perception and local waypoint-command correction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from envs.euler.g1_vision_nav.command_bridge import NavigationObservation, VelocityCommand
from envs.euler.g1_vision_nav.config import (
    BluePanelInspectionConfig,
    CommandLimits,
    NavigationConfig,
    VisualAvoidanceConfig,
)
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
    red_fraction: float
    green_fraction: float
    yellow_fraction: float
    blue_fraction: float


class FactoryColorObstacleDetector:
    """Detect saturated red/yellow/green/blue regions, not gray floor shadows."""

    def __init__(self, config: VisualAvoidanceConfig | None = None) -> None:
        self.config = config or VisualAvoidanceConfig()

    def estimate(self, rgb: np.ndarray) -> FactoryObstacleRisk:
        """Return per-region occupancy, excluding sky and the arm-heavy bottom."""
        # --------------------------------------------------------------
        # Perception block
        # 1. Keep the middle near-field strip because the head camera points
        #    downward and the lower image contains G1's hands and feet.
        # 2. Gate by saturation/brightness, then combine four hue masks.
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
        # 输入流是 RGB，不是 OpenCV 默认的 BGR。H 为 0～179，S/V 为 0～255。
        hsv = cv2.cvtColor(image[roi_top:roi_bottom], cv2.COLOR_RGB2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        vivid = (saturation >= self.config.hsv_saturation_min) & (value >= self.config.hsv_value_min)
        red_mask = vivid & ((hue <= 10) | (hue >= 170))  # 红色色相跨越首尾。
        yellow_mask = vivid & (hue >= 15) & (hue < 35)
        green_mask = vivid & (hue >= 35) & (hue < 90)
        blue_mask = vivid & (hue >= 90) & (hue <= 135)
        mask = red_mask | yellow_mask | green_mask | blue_mask

        one_third = max(1, width // 3)
        two_thirds = min(width - 1, 2 * width // 3)
        left = self._fraction(mask[:, :one_third])
        center = self._fraction(mask[:, one_third:two_thirds])
        right = self._fraction(mask[:, two_thirds:])
        # Only the center third represents the immediate path in front of G1.
        # Colored shelving at the image edges is common in the warehouse and must
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
            red_fraction=self._fraction(red_mask),
            green_fraction=self._fraction(green_mask),
            yellow_fraction=self._fraction(yellow_mask),
            blue_fraction=self._fraction(blue_mask),
        )

    @staticmethod
    def _fraction(mask: np.ndarray) -> float:
        return 0.0 if mask.size == 0 else float(np.count_nonzero(mask) / mask.size)


class BluePanelInspection:
    """只在最后一点启动：先停稳 → 有蓝色就拍，无蓝色才旋转寻找。

    不驱动前进/侧移，不把蓝色占比当距离，也不替代途中避障。
    柜子坐标只用于选择无蓝色时的搜索方向，不限制照片的构图朝向。
    """

    def __init__(self, config: BluePanelInspectionConfig | None = None) -> None:
        self.config = config or BluePanelInspectionConfig()
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.ready = False
        self.failed = False
        self.mode = "等待到达终点"
        self.blue_fraction = 0.0
        self.center_error = 0.0
        self.photo_rgb: np.ndarray | None = None
        self.photo_frame_index = -1
        self._station = np.zeros(2)
        self._started_s = 0.0
        self._last_frame = -1
        self._last_frame_s: float | None = None
        self._stable_since: float | None = None
        self._arrival_stop_sent = False
        self._arrival_settled = False
        self._search_direction = 0
        self._search_stopped = False
        self._command = VelocityCommand.stopped()

    def start(self, station_xy, sim_time_s: float) -> None:
        self.reset()
        self.active = True
        self._station = np.asarray(station_xy, dtype=float)
        self._started_s = sim_time_s

    def invalidate_confirmation(self) -> None:
        """新的安全事件废弃候选照片，但不能重置总超时。"""
        self.ready = False
        self.photo_rgb = None
        self.photo_frame_index = -1
        self._stable_since = self._last_frame_s = None
        self._command = VelocityCommand.stopped()
        self.mode = "安全暂停，重新确认画面"

    def _blue_candidate(self, rgb: np.ndarray) -> bool:
        height, width = rgb.shape[:2]
        # 排除最下方手臂和左右边缘；统计连通块，避免零碎蓝点凑够比例。
        x0, x1 = int(width * .1), int(width * .9)
        y0, y1 = int(height * .05), int(height * .85)
        roi = rgb[y0:y1, x0:x1]
        self.blue_fraction = self.center_error = 0.0
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (95, 90, 45), (130, 255, 255))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return False
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[label, cv2.CC_STAT_AREA]
        self.blue_fraction = float(area / mask.size)
        self.center_error = float((centers[label, 0] + x0) / width - .5)
        # 以最大连通区域占比判定，不追加居中或形状要求；排除满屏近景。
        return bool(self.config.blue_min_fraction <= self.blue_fraction <= .60)

    def update(self, *, rgb, frame_index: int, sim_time_s: float, position_xy,
               heading_rad: float, linear_speed_mps: float, yaw_rate_rps: float,
               command_still: bool, safe: bool) -> VelocityCommand:
        stop = VelocityCommand.stopped()
        if not self.active or self.ready or self.failed:
            return stop
        position = np.asarray(position_xy, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(
            (sim_time_s, heading_rad, linear_speed_mps, yaw_rate_rps)
        ).all():
            self.failed, self.mode = True, "位姿或速度无效，停止寻找，未拍照"
            return stop
        if sim_time_s - self._started_s >= self.config.timeout_s:
            self.failed, self.mode = True, "未获得可用的电气柜画面，本次未拍照"
            return stop
        if np.linalg.norm(position - self._station) > self.config.station_radius_m:
            # 漂移限制管运动，不取消已到点的事实，也不禁止拍下当前画面。
            self._search_stopped = True
            self._command = stop
        if not self._arrival_stop_sent:
            self._arrival_stop_sent = True
            self.mode = "已到终点，先停止移动"
            return stop
        if not safe or rgb is None:
            self._stable_since = self._last_frame_s = None
            self.mode = "安全暂停" if not safe else "等待新鲜 RGB"
            self._command = stop
            return stop

        still = (linear_speed_mps <= self.config.max_linear_speed_mps
                 and abs(yaw_rate_rps) <= self.config.max_yaw_speed_rps and command_still)
        if not self._arrival_settled:
            if not still:
                self.mode = "等待停稳"
                return stop
            self._arrival_settled = True
        if not still:
            self._stable_since = None
        if frame_index <= self._last_frame:
            # 同一张图重复读取不能攒够确认时间；丢帧也不能跨越空档确认。
            if self._last_frame_s is not None and sim_time_s - self._last_frame_s > self.config.max_frame_gap_s:
                self._stable_since = None
            return self._command
        continuous = (self._last_frame_s is not None
                      and 0 < sim_time_s - self._last_frame_s <= self.config.max_frame_gap_s)
        self._last_frame, self._last_frame_s = frame_index, sim_time_s
        candidate = self._blue_candidate(rgb)
        delta = np.asarray(self.config.cabinet_xy) - position
        cabinet_heading = float(np.arctan2(delta[1], delta[0]))
        cabinet_error = self._angle(cabinet_heading - heading_rad)
        if candidate:
            # 蓝色在画面任意水平位置均可，不为了构图居中继续转动。
            self._command = stop
            self.mode = "检测到电气柜，保持站立确认照片"
            if still:
                if not continuous or self._stable_since is None:
                    self._stable_since = sim_time_s
                if sim_time_s - self._stable_since >= self.config.stable_s - 1e-9:
                    self.ready, self.mode = True, "电气柜画面已确认"
                    self.photo_rgb = rgb.copy()
                    self.photo_frame_index = frame_index
            return stop
        self._stable_since = None
        if self._search_stopped:
            self.mode = "保持站立，等待可用的电气柜画面"
            return stop
        # 只在无合格蓝色时旋转；锁定方向，避免经过柜子方向后反复左右切换。
        if not self._search_direction:
            self._search_direction = 1 if cabinet_error >= 0 else -1
        self.mode = "正在寻找电气柜"
        yaw = self._search_direction * self.config.yaw_rate_rps
        self._command = VelocityCommand(yaw_rate_rps=float(yaw), walk_enabled=True)
        return self._command

    @staticmethod
    def _angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


class FactoryInspectionNavigator:
    """Slow down for mild color risk; steer away only for a blocked center.

    The selected avoidance side is latched until the center view has remained
    clear long enough.  There is deliberately no turn/pass/rejoin state
    machine: strong avoidance releases after fresh clear-frame confirmation.
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
        self._reset_clear_confirmation()
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
        self._reset_clear_confirmation()
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
        self._reset_clear_confirmation()
        self.latest_mode = "collision_escape"
        return VelocityCommand(
            forward_mps=forward_mps,
            lateral_mps=float(escape_side * self.visual.avoidance_lateral_mps),
            yaw_rate_rps=float(escape_side * self.visual.avoidance_yaw_rate_rps),
            walk_enabled=True,
        )

    def act(self, observation: NavigationObservation) -> VelocityCommand:
        """Preserve waypoint steering for mild risk; override only when blocked."""
        # --------------------------------------------------------------
        # Navigation decision block
        # - The pose controller always recomputes a command to the current
        #   waypoint from the live G1 position and yaw.
        # - Mild center risk only limits forward speed; blocked risk latches steering.
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
        new_frame = observation.frame_index != self._last_frame_index
        if not new_frame:
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
            if risk.blocked:
                self._avoidance_side = self._choose_clear_side(
                    risk,
                    observation.goal_bearing_rad,
                )
                self._avoidance_active = True
                self._reset_clear_confirmation()

        # 新鲜清晰证据才允许使用释放缓冲区；长间隔或时钟回退后重新确认。
        clear_gap = 0.0
        if new_frame and self._clear_last_simtime is not None:
            clear_gap = observation.sim_time_s - self._clear_last_simtime
            if not 0.0 < clear_gap <= self.visual.clear_frame_gap_s:
                self._reset_clear_confirmation()
                clear_gap = 0.0
        release_threshold = (
            self.visual.clear_release_risk_fraction
            if self._clear_last_simtime is not None
            else self.visual.clear_risk_fraction
        )
        if risk.risk_fraction > release_threshold:
            self._reset_clear_confirmation()
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
            forward_mps = (
                (1.0 - severity) * self.visual.obstacle_approach_forward_mps
                + severity * self.visual.near_obstacle_forward_mps
            )
            self.intervention_count += 1
            if not risk.blocked:
                # 轻度风险只限前进速度，不能使第二个路径点所需的右转越来越慢。
                # 不新建避障方向锁；若此前强避障仍在恢复中，则保留其释放记录。
                self.latest_mode = "visual_caution"
                return VelocityCommand(
                    forward_mps=min(base.forward_mps, forward_mps),
                    lateral_mps=base.lateral_mps,
                    yaw_rate_rps=base.yaw_rate_rps,
                    walk_enabled=True,
                )
            self.latest_mode = "visual_correction"
            return VelocityCommand(
                forward_mps=min(base.forward_mps, forward_mps),
                lateral_mps=self._avoidance_side * self.visual.avoidance_lateral_mps,
                yaw_rate_rps=self._avoidance_side * self.visual.avoidance_yaw_rate_rps,
                walk_enabled=True,
            )

        # 只有连续到达的新帧才能累计清晰时间；卡帧、长间断不能充当安全证据。
        if new_frame:
            self._clear_elapsed_s += clear_gap
            self._clear_last_simtime = observation.sim_time_s

        fraction = float(np.clip(
            (self._clear_elapsed_s - self.visual.clear_confirm_s) / self.visual.return_blend_s,
            0.0, 1.0,
        ))
        if fraction >= 1.0:
            self._avoidance_active = False
            self._avoidance_side = 0
            self._reset_clear_confirmation()
            self.latest_mode = "follow_path"
            return base

        # 画面刚清晰时沿当前朝向通过，不继续侧移，也不立刻转回桌子。
        # 确认后逐渐恢复每周期重新计算的目标转向，避免恢复旧方向。
        self.latest_mode = "clear_confirm" if fraction == 0.0 else "return_to_path"
        self.intervention_count += 1
        passing_speed = min(base.forward_mps, self.visual.clear_forward_mps)
        return VelocityCommand(
            forward_mps=float((1.0 - fraction) * passing_speed + fraction * base.forward_mps),
            lateral_mps=float(fraction * base.lateral_mps),
            yaw_rate_rps=float(fraction * base.yaw_rate_rps),
            walk_enabled=True,
        )

    def _reset_clear_confirmation(self) -> None:
        self._clear_elapsed_s = 0.0
        self._clear_last_simtime: float | None = None

    def _choose_clear_side(self, risk: FactoryObstacleRisk, goal_bearing_rad: float) -> int:
        difference = risk.left_fraction - risk.right_fraction
        if abs(difference) >= self.visual.clear_side_margin:
            return 1 if difference < 0.0 else -1
        return 1 if goal_bearing_rad >= 0.0 else -1
