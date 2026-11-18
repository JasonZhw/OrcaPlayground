"""结构化失败码与结果对象。

每个工具返回 `SkillResult(ok, code, message, detail)`，`code` 是机器可读的失败码，
`CODE_HINT` 把它翻译成给大模型看的中文提示——这是「失败反馈闭环」的词汇表。
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


