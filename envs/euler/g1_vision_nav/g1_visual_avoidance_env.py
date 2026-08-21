"""Online G1 environment that requires RGB to affect navigation commands."""

from __future__ import annotations

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
            "7072 RGB 绿色工作台检测已启用；威胁出现时减速并向画面较空一侧绕行",
        )

    def verify_step(self, step: int, verifier) -> None:
        super().verify_step(step, verifier)
        if step % self.NAVIGATION_LOG_INTERVAL != 0:
            return
        risk = self.visual_navigator.latest_risk
        if risk is None:
            return
        verifier.observe(
            f"visual_risk_{step}",
            f"risk={risk.risk_fraction:.3f}, regions=({risk.left_fraction:.3f}, "
            f"{risk.center_fraction:.3f}, {risk.right_fraction:.3f}), "
            f"side={self.visual_navigator.avoidance_side:+d}, "
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
            "7072 RGB 实际观察到绿色工作台",
        )
        verifier.check(
            "visual_avoidance_intervened",
            self.visual_navigator.intervention_count > 0,
            self.visual_navigator.intervention_count,
            ">0",
            "视觉风险实际改变了点目标速度指令",
        )
