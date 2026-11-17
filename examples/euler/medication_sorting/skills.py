"""命名技能库（对齐 agentic robotics 的「技能 + 结构化失败码」范式）。

设计参考
--------
- BT-Agent：行为树技能，强制局部执行条件 + 返回结构化失败反馈
- ChunkAct：把运动原语组进可复用 chunk，失败交还 agent 决策
- Recova：可命名的恢复技能

每个技能返回 `SkillResult`，其中 `code` 是结构化失败码，
agent 把 `code` 放进下一轮观测，据此选择恢复动作（nudge / regrasp / skip）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 结构化失败码
# ---------------------------------------------------------------------------

#: 所有可能的失败码
FAILURE_CODES = {
    "ok",                # 成功
    "ik_no_solution",    # 无满足碰撞/限位的 IK 解（几何不可行）
    "plan_collision",    # 起点或目标在碰撞中 / 无可行 pickup
    "lost_grasp",        # 抬升或搬运中物体滑落
    "missed_bin",        # 释放后物体没进筐
    "disturbed",         # 放置后物体被扰动
    "tracking_error",    # 末端没跟上规划轨迹
    "servo_range",       # 伺服目标超出控制范围
    "nonfinite_state",   # 仿真状态发散
    "no_object",         # 物体已不在（可能已被拿走）
    "target_mismatch",   # 视觉目标与任务绑定物体不一致
    "grasp_geometry",    # 指端与物体尚不满足夹取几何条件
    "unknown",           # 其它未分类错误
}

#: 失败码 → 给 agent 看的中文解释（放进观测 prompt）
CODE_HINT = {
    "ik_no_solution": "这个角度下机械臂够不到或会碰撞，换角度或放弃",
    "plan_collision": "起点/目标被挡住，可能被其它物体遮挡",
    "lost_grasp": "夹住了但抬起来时滑落了",
    "missed_bin": "放下去但没进筐",
    "disturbed": "放好后被碰倒了",
    "tracking_error": "手臂没跟到目标位姿",
    "servo_range": "关节指令超限",
    "nonfinite_state": "仿真异常",
    "no_object": "物体已不在桌上",
    "target_mismatch": "视觉定位与任务物体不一致，停止并重新确认目标",
    "grasp_geometry": "指端与物体的对准、深度或开口宽度不满足夹取要求",
    "unknown": "未知错误",
}


@dataclass
class SkillResult:
    """技能执行的结构化结果。"""
    ok: bool
    code: str
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.code not in FAILURE_CODES:
            self.code = "unknown"

    @property
    def hint(self) -> str:
        return CODE_HINT.get(self.code, "")

    def as_dict(self) -> dict:
        return {"ok": self.ok, "code": self.code, "message": self.message, **self.detail}


def classify(exc: BaseException) -> str:
    """把执行层抛出的异常归入结构化失败码。"""
    name = type(exc).__name__
    msg = str(exc)

    if name == "IKError":
        return "ik_no_solution"
    if name == "PlanningError":
        return "plan_collision"
    if name == "ValueError" and msg.startswith("Carried planning tracking error"):
        return "tracking_error"
    if name == "ValueError" and msg.startswith("Lost grasp:"):
        return "lost_grasp"
    if name == "SceneBindingError" or name == "ValueError":
        if "servo" in msg.lower() or "control range" in msg:
            return "servo_range"
        return "unknown"
    if name == "RuntimeError":
        low = msg.lower()
        if "did not follow" in low or "lost grasp" in low:
            return "lost_grasp"
        if "did not enter" in low:
            return "missed_bin"
        if "disturbed" in low:
            return "disturbed"
        if "missed target" in low or "end pose" in low:
            return "tracking_error"
        if "non-finite" in low or "simulation state" in low:
            return "nonfinite_state"
        if "no feasible" in low:
            return "plan_collision"
    return "unknown"


# ---------------------------------------------------------------------------
# 技能实现
# ---------------------------------------------------------------------------


class Skill:
    """技能基类：name + 前置条件 + 执行。"""

    name = "skill"

    def precondition(self, ctx, **kw) -> SkillResult | None:
        """返回 None 表示通过；返回 SkillResult(ok=False, ...) 表示不满足。"""
        return None

    def run(self, ctx, **kw) -> SkillResult:
        raise NotImplementedError


class GraspSkill(Skill):
    """grasp(object, yaw, coarse_xy)：抓取并放入目标筐。

    - coarse_xy=None（L0）：用真值位置
    - coarse_xy=(x,y)（L1）：用视觉接地给出的粗位置，在它周围做局部搜索

    角度兜底：先试给定 yaw，再按候选序列搜。
    局部搜索：粗位置 ±search_radius 的 3×3 网格，每个位置先做 IK 可行性预检，
    可行才跑完整抓取（完整抓取慢，靠「物体是否跟随抬升」来确认真夹到了）。
    """

    name = "grasp"

    def run(self, ctx, key, arm, yaw, yaw_candidates, tried,
            coarse_xy=None, search_radius=0.05) -> SkillResult:
        import numpy as np

        from examples.euler.medication_sorting.planning import IKError, servo_bias_feasible, solve_ik
        from examples.euler.medication_sorting.workflow import grasp_target

        trial = ctx["trial"]
        scene = ctx["scene"]
        order = [float(yaw)] + [float(y) for y in yaw_candidates if float(y) != yaw]

        # 候选抓取中心：粗位置 + 5×5 局部网格（中心优先，步长 r/2，覆盖 ±r）
        # 视觉接地误差实测 5-11cm，物体宽 6.4cm，需要覆盖 ±r 且步长 < 物体半径
        if coarse_xy is not None:
            r = search_radius
            steps = [-r, -r / 2, 0.0, r / 2, r]
            offsets = [(dx, dy) for dx in steps for dy in steps]
            # 中心优先（先试粗位置，再按距离近的向外扩）
            offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
            centers = [(coarse_xy[0] + dx, coarse_xy[1] + dy) for dx, dy in offsets]
        else:
            centers = [None]

        def feasible(cx, cand) -> bool:
            t = grasp_target(scene, arm, key, yaw_deg=cand, center_xy=cx)

            def valid(qq):
                return scene.collision_free(qq, arm, target=key, allow_target_contact=True,
                                            penetration_tolerance=0.001) and servo_bias_feasible(scene, arm)

            try:
                solve_ik(scene, arm, t["position"], t["rotation"], scene.d.qpos.copy(),
                         valid=valid, attempts=40, random_seed=0)
                return True
            except IKError:
                return False

        last: SkillResult = SkillResult(False, "unknown", "no attempt")
        for cx in centers:
            for cand in order:
                if cand in tried:
                    continue
                tried.add(cand)
                # IK 可行性预检（所有模式）：先把场景同步到当前状态，再用当前姿态检查。
                # 不可行的角度不调 pick_place，手臂不动 → 不会爬升，也不需要复位。
                trial.sync()
                if not feasible(cx, cand):
                    continue
                try:
                    # place ≠ release：抓取+搬运+降到筐内，但不释放。
                    # 直接调 pick_place：pick_available 内部又会调一次 pick_place，
                    # 且永远用 yaw=0，会让角度搜索失效。prepared=False 让它自己做 prepare_pick。
                    trial.pick_place(arm, key, prepared=False, yaw_deg=cand,
                                     center_xy=cx, release=False)
                    return SkillResult(True, "ok", detail={
                        "used_yaw": cand, "center": cx,
                        "pending_release": (arm, key),
                    })
                except Exception as exc:  # noqa: BLE001
                    last = SkillResult(False, classify(exc), f"{type(exc).__name__}: {exc}")
        return last


class NudgeSkill(Skill):
    """nudge(object, direction)：沿给定方向轻推物体，改变其位姿（恢复技能）。

    用 `allow=True` 让夹爪/末端接触物体，做一小段直线位移把物体推离当前位置。
    """

    name = "nudge"

    def run(self, ctx, key, arm, direction="+y", dist=0.05) -> SkillResult:
        import numpy as np

        from examples.euler.medication_sorting.workflow import grasp_target

        trial = ctx["trial"]
        scene = ctx["scene"]
        try:
            t = grasp_target(scene, arm, key)
            p = np.asarray(t["position"], dtype=float)
            step = {"+x": [1, 0, 0], "-x": [-1, 0, 0],
                    "+y": [0, 1, 0], "-y": [0, -1, 0]}[direction]
            goal = p + np.asarray(step) * dist
            trial.move(arm, goal, t["rotation"], key + "_nudge",
                       target=key, allow=True, max_velocity=0.05, max_acceleration=0.10)
            return SkillResult(True, "ok", detail={"direction": direction, "dist": dist})
        except Exception as exc:  # noqa: BLE001
            return SkillResult(False, classify(exc), f"{type(exc).__name__}: {exc}")


class SkipSkill(Skill):
    """skip(object)：放弃该物体（暂缓，等挡路的移走后重试）。"""

    name = "skip"

    def run(self, ctx, key) -> SkillResult:
        return SkillResult(True, "ok", detail={"object": key})


class FinishSkill(Skill):
    """finish：宣布任务结束。"""

    name = "finish"

    def run(self, ctx) -> SkillResult:
        return SkillResult(True, "ok")


#: 技能注册表
SKILLS: dict[str, Skill] = {
    "grasp": GraspSkill(),
    "nudge": NudgeSkill(),
    "skip": SkipSkill(),
    "finish": FinishSkill(),
}


def get_skill(name: str) -> Skill:
    if name not in SKILLS:
        raise KeyError(f"unknown skill: {name} (available: {sorted(SKILLS)})")
    return SKILLS[name]
