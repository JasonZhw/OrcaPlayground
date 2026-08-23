"""Online G1 environment that requires RGB to affect navigation commands."""

from __future__ import annotations

from envs.euler.g1_vision_nav.camera_stream import CameraFrame
from envs.euler.g1_vision_nav.g1_vision_nav_env import G1VisionNavEnv
from envs.euler.g1_vision_nav.visual_avoidance import VisualAvoidanceNavigator


class G1VisualAvoidanceEnv(G1VisionNavEnv):
    """Add visual-intervention telemetry to the point-goal navigation Env."""

    def __init__(self, *args, navigator: VisualAvoidanceNavigator, **kwargs) -> None:
        self.visual_navigator = navigator
        super().__init__(*args, navigator=navigator, **kwargs)

    def before_loop(self, verifier) -> None:
        super().before_loop(verifier)
        verifier.observe(
            "visual_avoidance_mode",
            "7072 RGB 连续修正已启用：正常直线路径跟随；检测到障碍时锁定"
            "较空一侧并叠加转向，连续清空后平滑回到主路径，全程不倒车",
        )

    def verify_step(self, step: int, verifier) -> None:
        super().verify_step(step, verifier)
        if step % self.NAVIGATION_LOG_INTERVAL != 0:
            return
        risk = self.visual_navigator.latest_risk
        if risk is None:
            return
        base = self.visual_navigator.latest_base_command
        verifier.observe(
            f"visual_risk_{step}",
            f"risk={risk.risk_fraction:.3f}, regions=({risk.left_fraction:.3f}, "
            f"{risk.center_fraction:.3f}, {risk.right_fraction:.3f}), "
            f"colors=dark:{risk.dark_fraction:.3f}/green:{risk.green_fraction:.3f}/"
            f"yellow:{risk.yellow_fraction:.3f}, "
            f"mode={self.visual_navigator.latest_mode}, "
            f"active={self.visual_navigator.avoidance_active}, "
            f"side={self.visual_navigator.avoidance_side:+d}, "
            f"base_yaw={base.yaw_rate_rps:+.3f}, "
            f"interventions={self.visual_navigator.intervention_count}",
            step=step,
        )

    def verify_final(self, verifier) -> None:
        super().verify_final(verifier)
        verifier.check(
            "visual_obstacle_observed",
            self.visual_navigator.maximum_risk_fraction >= self.visual_navigator.visual.slow_risk_fraction,
            self.visual_navigator.maximum_risk_fraction,
            f">={self.visual_navigator.visual.slow_risk_fraction}",
            "7072 RGB 实际观察到深色、绿色或黄色障碍区域",
        )
        verifier.check(
            "visual_avoidance_intervened",
            self.visual_navigator.intervention_count > 0,
            self.visual_navigator.intervention_count,
            ">0",
            "视觉风险实际改变了点目标速度指令",
        )

    def _camera_preview_lines(self, frame: CameraFrame) -> list[str]:
        lines = super()._camera_preview_lines(frame)
        risk = self.visual_navigator.latest_risk
        if risk is None:
            lines.append(f"Avoidance: {self.visual_navigator.latest_mode} | risk: waiting")
        else:
            base = self.visual_navigator.latest_base_command
            lines.append(
                f"Avoidance: {self.visual_navigator.latest_mode}"
                f" | risk={risk.risk_fraction:.3f}"
                f" | side={self.visual_navigator.avoidance_side:+d}"
                f" | active={int(self.visual_navigator.avoidance_active)}"
                f" | base_yaw={base.yaw_rate_rps:+.2f}"
            )
        return lines
