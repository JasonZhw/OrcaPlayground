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
    receiver; they do not overwrite UI camera settings. Depth stays disabled.
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
    navigation_hz: int = 50

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
    """Conservative initial limits for commands sent to the G1 policy."""

    max_forward_mps: float = 0.15
    max_backward_mps: float = 0.08
    max_lateral_mps: float = 0.08
    max_yaw_rate_rps: float = 0.50
    max_forward_accel_mps2: float = 0.30
    max_lateral_accel_mps2: float = 0.25
    max_yaw_accel_rps2: float = 1.00

    def __post_init__(self) -> None:
        values = (
            self.max_forward_mps,
            self.max_backward_mps,
            self.max_lateral_mps,
            self.max_yaw_rate_rps,
            self.max_forward_accel_mps2,
            self.max_lateral_accel_mps2,
            self.max_yaw_accel_rps2,
        )
        if min(values) <= 0.0:
            raise ValueError("all command limits must be positive")


@dataclass(frozen=True)
class NavigationConfig:
    """Point-goal controller and episode safety settings."""

    # 终点附近优先朝向：0.8 m 内开始对齐，允许转身偏移至 1 m。
    goal_alignment_radius_m: float = 0.80
    goal_tolerance_m: float = 1.00
    # 世界 Z 轴 yaw：+90° 朝世界 +Y，与绿色桌子 Demo 的出发朝向一致。
    goal_yaw_deg: float = 90.0
    goal_yaw_tolerance_deg: float = 10.0
    goal_turn_min_rate_rps: float = 0.35
    slowdown_radius_m: float = 1.50
    turn_in_place_bearing_rad: float = 0.60
    turning_forward_mps: float = 0.06
    minimum_forward_mps: float = 0.04
    forward_gain: float = 0.40
    bearing_gain: float = 1.20
    startup_stand_s: float = 2.0
    fall_height_m: float = 0.60
    fall_tilt_rad: float = 0.80
    # A brief hand/body touch against the workbench must not permanently zero
    # the locomotion command. Falling is still handled as an emergency stop.
    collision_stop: bool = False

    def __post_init__(self) -> None:
        positive_values = (
            self.goal_alignment_radius_m,
            self.goal_tolerance_m,
            self.goal_turn_min_rate_rps,
            self.slowdown_radius_m,
            self.turn_in_place_bearing_rad,
            self.turning_forward_mps,
            self.minimum_forward_mps,
            self.forward_gain,
            self.bearing_gain,
            self.startup_stand_s,
            self.fall_height_m,
            self.fall_tilt_rad,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive_values):
            raise ValueError("navigation settings must be positive")
        if not math.isfinite(self.goal_yaw_deg):
            raise ValueError("goal_yaw_deg must be finite")
        if not math.isfinite(self.goal_yaw_tolerance_deg) or not 0 < self.goal_yaw_tolerance_deg <= 180:
            raise ValueError("goal_yaw_tolerance_deg must be in (0, 180]")
        if self.slowdown_radius_m <= self.goal_tolerance_m:
            raise ValueError("slowdown_radius_m must exceed goal_tolerance_m")
        if self.goal_alignment_radius_m >= self.goal_tolerance_m:
            raise ValueError("goal_alignment_radius_m must be smaller than goal_tolerance_m")

    def heading_error_rad(self, base_yaw_world_rad: float) -> float:
        """最短有符号转角；处理 +180° / -180° 的环绕边界。"""
        delta = math.radians(self.goal_yaw_deg) - base_yaw_world_rad
        return math.atan2(math.sin(delta), math.cos(delta))

    def goal_reached(self, distance_m: float, base_yaw_world_rad: float) -> bool:
        """控制、结束判定和报告共同使用的位置 + 朝向到达条件。"""
        if not (math.isfinite(distance_m) and math.isfinite(base_yaw_world_rad)):
            return False
        return (
            0 <= distance_m <= self.goal_tolerance_m
            and abs(self.heading_error_rad(base_yaw_world_rad))
            <= math.radians(self.goal_yaw_tolerance_deg) + 1e-9
        )


@dataclass(frozen=True)
class VisualAvoidanceConfig:
    """RGB workbench detector and conservative local-avoidance settings.

    This first detector is deliberately scoped to the green workbenches in the
    current Demo 1 layout. It provides a deterministic visual baseline before a
    learned, category-independent perception model is introduced.
    """

    roi_bottom_fraction: float = 0.72
    minimum_green: int = 70
    minimum_blue: int = 55
    minimum_green_red_gap: int = 18
    minimum_blue_red_gap: int = 8
    minimum_green_blue_gap: int = -15
    # Start turning only when green occupies a substantial part of the center
    # view; stop turning after it falls below the lower hysteresis threshold.
    slow_risk_fraction: float = 0.10
    clear_risk_fraction: float = 0.08
    clear_side_margin: float = 0.005
    turn_forward_mps: float = 0.08
    avoidance_yaw_rate_rps: float = 0.50
    # 清晰确认与回归目标的时间均基于连续更新的 RGB 和仿真时间。
    clear_confirm_s: float = 0.55
    return_blend_s: float = 1.00
    clear_frame_gap_s: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.roi_bottom_fraction <= 1.0:
            raise ValueError("roi_bottom_fraction must be in (0, 1]")
        color_values = (
            self.minimum_green,
            self.minimum_blue,
            self.minimum_green_red_gap,
            self.minimum_blue_red_gap,
        )
        if min(color_values) < 0:
            raise ValueError("color thresholds must be non-negative")
        fractions = (
            self.clear_risk_fraction,
            self.slow_risk_fraction,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("risk fractions must be in [0, 1]")
        if not self.clear_risk_fraction < self.slow_risk_fraction:
            raise ValueError("risk thresholds must be strictly increasing")
        if self.clear_side_margin < 0.0:
            raise ValueError("clear_side_margin must be non-negative")
        velocities = (
            self.turn_forward_mps,
            self.avoidance_yaw_rate_rps,
        )
        if min(velocities) <= 0.0:
            raise ValueError("avoidance velocities must be positive")
        durations = (self.clear_confirm_s, self.return_blend_s, self.clear_frame_gap_s)
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("clearance timings must be positive and finite")


@dataclass(frozen=True)
class SceneConfig:
    """Asset identifiers; the environment path is filled after asset selection."""

    environment_asset_path: str | None = None
    environment_actor_name: str | None = None
    robot_asset_path: str = "assets/cae3c6559556dd4f/default_project/prefabs/g1_pick_usda"
    robot_actor_name: str = "g1"


@dataclass(frozen=True)
class G1VisionNavConfig:
    """Top-level configuration shared by the future Env and entry script."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    command_limits: CommandLimits = field(default_factory=CommandLimits)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    visual_avoidance: VisualAvoidanceConfig = field(default_factory=VisualAvoidanceConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)


DEFAULT_CONFIG = G1VisionNavConfig()
