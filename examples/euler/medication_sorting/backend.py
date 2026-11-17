"""Backend 协议：把我们的执行层适配成 RobotUse 风格的接口。

对齐 RobotUse 的 contracts.Backend：模型只通过像素点（point）和引用（ref）
与后端交互，后端绝不从任务定义/真值里推导目标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """一次观测：多视角图像。"""
    observation_id: str
    views: dict[str, Any]  # view_id -> RGB ndarray


@dataclass(frozen=True)
class PointRef:
    """模型在某个视图里点出的像素目标。u/v 为归一化坐标 [0, 1000]。"""
    ref: str
    observation_id: str
    view_id: str
    u: float
    v: float
    world_xy: tuple[float, float]  # 后端用单应矩阵接地出的桌面 (x,y)


@dataclass
class Candidate:
    """围绕一个点生成的候选抓取。"""
    candidate_ref: str
    key: str
    arm: str
    yaw: float
    center_xy: tuple[float, float] | None


@dataclass
class Validation:
    validation_ref: str
    candidate_ref: str
    accepted: bool
    code: str = "ok"


@dataclass
class Execution:
    execution_ref: str
    status: str  # succeeded | failed | unknown
    code: str = "ok"
    message: str = ""


class Backend:
    """RobotUse 风格后端适配器。

    内部复用 Trial / Scene / skills，只暴露语义化接口。
    """

    def __init__(self, trial, scene, env, bridge, loop, grabber, grounder, playbook):
        self.trial = trial
        self.scene = scene
        self.env = env
        self.bridge = bridge
        self.loop = loop
        self.grabber = grabber
        self.grounder = grounder      # 单应矩阵 H 或 None（L0）
        self.playbook = playbook
        self._seq = 0
        self._last_views: dict = {}

    # ---- 观测 ----
    def observe(self) -> Observation:
        """抓三路相机帧，返回 Observation。"""
        import time

        frames = {}
        if self.grabber is not None:
            for i in range(2):
                try:
                    self.loop.run_until_complete(self.bridge.render(
                        self.env.data.qpos.ravel(), float(self.env.data.time),
                        simulate_index=700000 + self._seq * 10 + i, request_idr=True))
                except Exception:
                    break
                time.sleep(0.08)
            frames = self.grabber.grab(max_wait=8.0)
        self._last_views = frames
        obs = Observation(observation_id=f"obs{self._seq}", views=frames)
        self._seq += 1
        return obs

    # ---- 接地 ----
    def point(self, observation_id: str, view_id: str, u: float, v: float) -> PointRef:
        """模型给归一化像素 → 单应矩阵 → 桌面 (x,y)。"""
        from .visual_grounding import pixel_to_world

        if self.grounder is None:
            world = (float("nan"), float("nan"))  # L0：无接地，用真值兜底
        else:
            # 归一化 [0,1000] -> 实际像素
            img = None
            obs_views = self._last_views
            if observation_id in ("obs0",) and view_id in obs_views:
                img = obs_views[view_id]
            w = img.shape[1] if img is not None else 640
            h = img.shape[0] if img is not None else 480
            px_u, px_v = u / 1000.0 * w, v / 1000.0 * h
            world = pixel_to_world(self.grounder, px_u, px_v)
        return PointRef(ref=f"pt{self._seq}", observation_id=observation_id,
                        view_id=view_id, u=u, v=v, world_xy=world)

    # ---- 候选抓取 ----
    def grasp_candidates(self, point_ref: PointRef, key: str, arm: str) -> list[Candidate]:
        """围绕接地点生成候选：13 角度 × 3×3 局部网格（中心优先）。"""
        from .agent_policy import YAW_CANDIDATES

        best = self.playbook.best_yaw(key)
        order = ([best] if best is not None else []) + [float(y) for y in YAW_CANDIDATES if y != best]
        cands = []
        r = 0.05
        centers = [(0.0, 0.0)]
        if point_ref.world_xy[0] == point_ref.world_xy[0]:  # 非 NaN（L1）
            centers += [(dx, dy) for dx in (-r, 0, r) for dy in (-r, 0, r) if (dx, dy) != (0, 0)]
        for (dx, dy) in centers:
            cx = (point_ref.world_xy[0] + dx, point_ref.world_xy[1] + dy)
            for yaw in order:
                cands.append(Candidate(candidate_ref=f"c{self._seq}_{len(cands)}",
                                       key=key, arm=arm, yaw=yaw, center_xy=cx))
        return cands

    # ---- 校验候选（IK 可行性 + 碰撞）----
    def validate_grasp(self, candidate: Candidate) -> Validation:
        from .planning import IKError, servo_bias_feasible, solve_ik
        from .workflow import grasp_target

        t = grasp_target(self.scene, candidate.arm, candidate.key,
                         yaw_deg=candidate.yaw, center_xy=candidate.center_xy)

        def valid(qq):
            return self.scene.collision_free(qq, candidate.arm, target=candidate.key,
                                             allow_target_contact=True,
                                             penetration_tolerance=0.001) and \
                servo_bias_feasible(self.scene, candidate.arm)

        try:
            solve_ik(self.scene, candidate.arm, t["position"], t["rotation"],
                     self.scene.d.qpos.copy(), valid=valid, attempts=18, random_seed=0)
            return Validation(validation_ref=f"v{self._seq}", candidate_ref=candidate.candidate_ref,
                              accepted=True)
        except IKError:
            return Validation(validation_ref=f"v{self._seq}", candidate_ref=candidate.candidate_ref,
                              accepted=False, code="ik_no_solution")

    # ---- 执行（grasp + place，不释放）----
    def execute_grasp(self, candidate: Candidate) -> Execution:
        from .skills import classify

        try:
            self.trial.pick_place(candidate.arm, candidate.key, prepared=False,
                                  yaw_deg=candidate.yaw, center_xy=candidate.center_xy,
                                  release=False)
            return Execution(execution_ref=f"e{self._seq}", status="succeeded",
                             code="ok", message="")
        except Exception as exc:  # noqa: BLE001
            return Execution(execution_ref=f"e{self._seq}", status="failed",
                             code=classify(exc), message=f"{type(exc).__name__}: {exc}")

    # ---- 释放 ----
    def release(self, arm: str, key: str) -> Execution:
        from .skills import classify

        try:
            self.trial.release_and_return(arm, key)
            return Execution(execution_ref=f"e{self._seq}", status="succeeded", code="ok")
        except Exception as exc:  # noqa: BLE001
            return Execution(execution_ref=f"e{self._seq}", status="failed",
                             code=classify(exc), message=f"{type(exc).__name__}: {exc}")
