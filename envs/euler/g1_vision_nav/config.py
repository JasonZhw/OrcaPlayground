"""Scene-independent configuration for the G1 vision-navigation demo."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
G1_MODEL_XML = PROJECT_ROOT / "assets" / "g1" / "g1_29dof_camera.xml"
G1_LOCOMOTION_ONNX = PROJECT_ROOT / "assets" / "g1" / "models" / "dec_loco" / "model_6600.onnx"
G1_LOCOMOTION_CONFIG = PROJECT_ROOT / "assets" / "g1" / "config" / "g1_29dof_hist.yaml"


@dataclass(frozen=True)
class CameraConfig:
    """G1 head-camera stream settings.

    The user enables the head ColorCamera in Studio. These values describe the
    receiver; they do not overwrite UI camera settings. The factory entry uses
    RGB for navigation, inspection photos, and the browser preview.
    """

    entity_name: str = "camera_head"
    rgb_port: int = 7070
    left_rgb_port: int | None = None
    right_rgb_port: int | None = None
    depth_port: int | None = None
    width: int = 640
    height: int = 480
    vertical_fov_deg: float = 60.0
    first_frame_timeout_s: float = 10.0
    frame_timeout_s: float = 1.0
    enable_depth_preview: bool = False

    def __post_init__(self) -> None:
        if not self.entity_name:
            raise ValueError("entity_name must not be empty")
        rgb_ports = tuple(port for port in (self.rgb_port, self.left_rgb_port, self.right_rgb_port) if port is not None)
        if any(not 1 <= port <= 65535 for port in rgb_ports):
            raise ValueError("RGB camera ports must be in [1, 65535]")
        if len(set(rgb_ports)) != len(rgb_ports):
            raise ValueError("head, left, and right RGB ports must be different")
        if self.depth_port is not None:
            if not 1 <= self.depth_port <= 65535:
                raise ValueError("depth_port must be in [1, 65535]")
            if self.depth_port in rgb_ports:
                raise ValueError("depth_port must differ from all RGB ports")
        if self.enable_depth_preview and self.depth_port is None:
            raise ValueError("enable_depth_preview requires a known depth_port")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera resolution must be positive")
        if not 0.0 < self.vertical_fov_deg < 180.0:
            raise ValueError("vertical_fov_deg must be in (0, 180)")
        if self.first_frame_timeout_s <= 0.0:
            raise ValueError("first_frame_timeout_s must be positive")
        if self.frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")


@dataclass(frozen=True)
class TimingConfig:
    """Rates for physics, low-level locomotion, and high-level navigation."""

    physics_hz: int = 1000
    locomotion_hz: int = 50
    navigation_hz: int = 10

    def __post_init__(self) -> None:
        if min(self.physics_hz, self.locomotion_hz, self.navigation_hz) <= 0:
            raise ValueError("all control rates must be positive")
        if self.physics_hz % self.locomotion_hz != 0:
            raise ValueError("physics_hz must be divisible by locomotion_hz")
        if self.locomotion_hz % self.navigation_hz != 0:
            raise ValueError("locomotion_hz must be divisible by navigation_hz")

    @property
    def physics_steps_per_locomotion_step(self) -> int:
        return self.physics_hz // self.locomotion_hz

    @property
    def locomotion_steps_per_navigation_step(self) -> int:
        return self.locomotion_hz // self.navigation_hz


@dataclass(frozen=True)
class CommandLimits:
    """Limits for route commands sent to the frozen G1 policy."""

    max_forward_mps: float = 1.2
    max_backward_mps: float = 0.10
    max_lateral_mps: float = 0.15
    max_yaw_rate_rps: float = 0.65
    max_forward_accel_mps2: float = 0.80
    max_forward_decel_mps2: float = 1.20
    max_lateral_accel_mps2: float = 0.40
    max_yaw_accel_rps2: float = 1.30

    def __post_init__(self) -> None:
        values = (
            self.max_forward_mps,
            self.max_backward_mps,
            self.max_lateral_mps,
            self.max_yaw_rate_rps,
            self.max_forward_accel_mps2,
            self.max_forward_decel_mps2,
            self.max_lateral_accel_mps2,
            self.max_yaw_accel_rps2,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("all command limits must be positive")


@dataclass(frozen=True)
class NavigationConfig:
    """Point-goal controller and episode safety settings."""

    goal_tolerance_m: float = 0.38
    waypoint_tolerance_m: float = 0.40
    # Keep the crossed-waypoint fallback close to the actual path.  A wider
    # corridor can select the next segment while G1 is still visibly far from
    # the corner and make the requested yaw change abruptly.
    waypoint_passage_tolerance_m: float = 0.40
    slowdown_radius_m: float = 1.20
    turn_in_place_bearing_rad: float = 0.60
    # Keep a small forward arc because the frozen policy turns poorly at an
    # exact vx=0, but never let a large heading error retain cruise speed.
    turning_forward_mps: float = 0.25
    waypoint_turning_forward_mps: float = 0.20
    waypoint_heading_release_rad: float = 0.35
    minimum_forward_mps: float = 0.04
    forward_gain: float = 0.80
    bearing_gain: float = 1.20
    startup_stand_s: float = 0.0
    fall_height_m: float = 0.30
    fall_tilt_rad: float = 0.80
    collision_stop: bool = True
    fall_confirmation_steps: int = 3
    collision_confirmation_steps: int = 3
    collision_recovery_steps: int = 10
    # Do not deadlock while stopped against an obstacle. After one second,
    # release collision hold and let the route/vision command steer away.
    collision_escape_steps: int = 10
    # At 10 Hz this guarantees 5 seconds of uninterrupted recovery command.
    collision_recovery_grace_steps: int = 50
    recovery_forward_mps: float = 0.30

    def __post_init__(self) -> None:
        positive_values = (
            self.goal_tolerance_m,
            self.waypoint_tolerance_m,
            self.waypoint_passage_tolerance_m,
            self.slowdown_radius_m,
            self.turn_in_place_bearing_rad,
            self.turning_forward_mps,
            self.waypoint_turning_forward_mps,
            self.waypoint_heading_release_rad,
            self.minimum_forward_mps,
            self.forward_gain,
            self.bearing_gain,
            self.recovery_forward_mps,
            self.fall_height_m,
            self.fall_tilt_rad,
        )
        if min(positive_values) <= 0.0:
            raise ValueError("navigation settings must be positive")
        if self.startup_stand_s < 0.0:
            raise ValueError("startup_stand_s must be non-negative")
        if self.slowdown_radius_m <= self.goal_tolerance_m:
            raise ValueError("slowdown_radius_m must exceed goal_tolerance_m")
        if self.waypoint_tolerance_m < self.goal_tolerance_m:
            raise ValueError("waypoint_tolerance_m must not be smaller than goal_tolerance_m")
        if self.waypoint_passage_tolerance_m < self.waypoint_tolerance_m:
            raise ValueError("waypoint passage tolerance must cover waypoint tolerance")
        if self.waypoint_heading_release_rad >= self.turn_in_place_bearing_rad:
            raise ValueError("waypoint heading release must be below turn threshold")
        if self.waypoint_turning_forward_mps > self.turning_forward_mps:
            raise ValueError("waypoint turning speed must not exceed launch turning speed")
        safety_steps = (
            self.fall_confirmation_steps,
            self.collision_confirmation_steps,
            self.collision_recovery_steps,
            self.collision_escape_steps,
            self.collision_recovery_grace_steps,
        )
        if min(safety_steps) <= 0:
            raise ValueError("safety confirmation and recovery steps must be positive")


@dataclass(frozen=True)
class VisualAvoidanceConfig:
    """Simple RGB color occupancy used as a near-obstacle proxy."""

    # The head camera looks downward and includes G1's dark hands in the lower
    # image.  Use the middle near-field strip for obstacle activation; the
    # outer thirds remain useful only for choosing the clearer steering side.
    roi_top_fraction: float = 0.25
    roi_bottom_fraction: float = 0.55
    # OpenCV uint8 HSV：S/V 为 0～255。低饱和度灰地板、阴影不当作彩色障碍。
    hsv_saturation_min: int = 80
    hsv_value_min: int = 50
    slow_risk_fraction: float = 0.08
    blocked_risk_fraction: float = 0.20
    hard_stop_risk_fraction: float = 0.40
    clear_risk_fraction: float = 0.04
    # 先低于清晰阈值才开始确认；确认中允许小幅波动，避免反复归零。
    clear_release_risk_fraction: float = 0.06
    # 新鲜清晰帧连续确认后再恢复目标转向；不按重复读取的帧数计时。
    clear_confirm_s: float = 0.55
    return_blend_s: float = 1.0
    clear_frame_gap_s: float = 0.35
    stale_frame_limit_steps: int = 10
    clear_side_margin: float = 0.004
    obstacle_approach_forward_mps: float = 0.18
    near_obstacle_forward_mps: float = 0.05
    clear_forward_mps: float = 0.30
    avoidance_lateral_mps: float = 0.15
    avoidance_yaw_rate_rps: float = 0.65

    def __post_init__(self) -> None:
        if not 0.0 <= self.roi_top_fraction < self.roi_bottom_fraction <= 1.0:
            raise ValueError("expected 0 <= roi_top_fraction < roi_bottom_fraction <= 1")
        for duration in (self.clear_confirm_s, self.return_blend_s, self.clear_frame_gap_s):
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("clear confirmation/blend/frame gap durations must be finite and positive")
        for value in (self.hsv_saturation_min, self.hsv_value_min):
            if not math.isfinite(value) or not 1 <= value <= 255:
                raise ValueError("HSV saturation/value thresholds must be finite and in [1, 255]")
        fractions = (
            self.clear_risk_fraction,
            self.clear_release_risk_fraction,
            self.slow_risk_fraction,
            self.blocked_risk_fraction,
            self.hard_stop_risk_fraction,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("risk fractions must be in [0, 1]")
        if not (
            self.clear_risk_fraction
            < self.clear_release_risk_fraction
            < self.slow_risk_fraction
            < self.blocked_risk_fraction
            < self.hard_stop_risk_fraction
        ):
            raise ValueError("risk thresholds must be strictly increasing")
        if self.stale_frame_limit_steps <= 0:
            raise ValueError("stale-frame steps must be positive")
        if self.clear_side_margin < 0.0:
            raise ValueError("clear_side_margin must be non-negative")
        velocities = (
            self.obstacle_approach_forward_mps,
            self.near_obstacle_forward_mps,
            self.clear_forward_mps,
            self.avoidance_lateral_mps,
            self.avoidance_yaw_rate_rps,
        )
        if min(velocities) <= 0.0:
            raise ValueError("avoidance velocities must be positive")
        if not (
            self.near_obstacle_forward_mps
            <= self.obstacle_approach_forward_mps
            <= self.clear_forward_mps
        ):
            raise ValueError("expected near <= approach <= clear speeds")


@dataclass(frozen=True)
class BluePanelInspectionConfig:
    """柜前拍照参数；蓝色是场景约定的标记，不是仪表盘语义识别。"""

    cabinet_xy: tuple[float, float] = (21.59, 4.36)  # 柜子方向参考，不是停车点。
    blue_min_fraction: float = 0.05  # 最大蓝色连通块 / 中央 ROI 面积。
    stable_s: float = 0.0  # 停稳后有效帧合格即拍，不额外等待。
    timeout_s: float = 15.0
    station_radius_m: float = 0.43  # 超出则停止继续旋转，但仍允许静止拍照。
    yaw_rate_rps: float = 0.30
    max_linear_speed_mps: float = 0.08
    max_yaw_speed_rps: float = 0.10
    max_frame_gap_s: float = 0.35

    def __post_init__(self) -> None:
        if len(self.cabinet_xy) != 2 or not all(math.isfinite(v) for v in self.cabinet_xy):
            raise ValueError("cabinet_xy must contain two finite coordinates")
        if not 0 < self.blue_min_fraction < 0.60:
            raise ValueError("invalid blue area threshold")
        durations_and_limits = (
            self.timeout_s, self.station_radius_m, self.yaw_rate_rps,
            self.max_linear_speed_mps, self.max_yaw_speed_rps, self.max_frame_gap_s,
        )
        if not all(math.isfinite(v) and v > 0 for v in durations_and_limits):
            raise ValueError("inspection durations and limits must be finite and positive")
        if not math.isfinite(self.stable_s) or self.stable_s < 0:
            raise ValueError("inspection stable duration must be finite and nonnegative")
        if self.timeout_s <= self.stable_s:
            raise ValueError("inspection timeout must exceed stable duration")


@dataclass(frozen=True)
class G1VisionNavConfig:
    """Top-level configuration shared by the future Env and entry script."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    command_limits: CommandLimits = field(default_factory=CommandLimits)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    visual_avoidance: VisualAvoidanceConfig = field(default_factory=VisualAvoidanceConfig)


DEFAULT_CONFIG = G1VisionNavConfig()
