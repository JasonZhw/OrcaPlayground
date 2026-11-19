"""细粒度工具集：让 agent 逐步控制机械臂（RobotUse 风格）。

与旧的「一个大 grasp 包办一切」不同，这里每个工具是一个**小步、可逆**的动作，
agent 逐步调用，并决定：
  - 要运动到哪个坐标（move_to）
  - 现在手在哪、关节什么状态（where）
  - 到了没有（judge）
  - 要不要微调（micro_adjust）
  - 什么时候闭合/释放（grasp / release）

坐标由「视觉判断 + 机械臂状态」共同决定，几何计算（IK/轨迹）仍由后端完成。
"""

from __future__ import annotations

import time

from examples.euler.medication_sorting.failures import SkillResult, classify


class ArmTools:
    """机械臂细粒度工具。内部复用 Trial 的运动/夹爪原语，只暴露语义化接口。"""

    def __init__(self, trial, scene, env, bridge, loop, grabber, grounder, backend):
        self.trial = trial
        self.scene = scene
        self.env = env
        self.bridge = bridge
        self.loop = loop
        self.grabber = grabber
        self.grounder = grounder  # 单应矩阵 H 或 None
        self.backend = backend  # 用于 judge 的视觉模型
        self.observation_index = 1_000_000 + int(time.monotonic() * 1000) % 1_000_000_000

    # ------------------------------------------------------------------
    # 感知
    # ------------------------------------------------------------------
    def observe(self) -> dict:
        """拍三路相机，返回 {view_id: ndarray}。"""
        if self.grabber is None:
            return {}
        self.observation_index += 1
        for _i in range(2):
            try:
                self.loop.run_until_complete(
                    self.bridge.render(
                        self.env.data.qpos.ravel(),
                        float(self.env.data.time),
                        simulate_index=self.observation_index,
                        request_idr=True,
                    )
                )
            except Exception:
                return {}  # A failed render must not make old frames look current.
            time.sleep(0.08)
        # 不要求帧索引精确匹配（OrcaLab GUI 的帧计数偶尔对不上会导致取不到帧）；
        # 直接取最新帧，渲染刚刷新过，画面足够新。
        return self.grabber.grab(max_wait=8.0, required_simulate_index=None)


    def point(self, images, object_desc: str):
        """让模型看图报出物体中心像素 → 单应矩阵 → 世界坐标 (x, y)。

        object_desc：自然语言描述，如 "白色药瓶" / "左侧紫色筐"。
        返回 (x, y) 或 None（模型没报出）。
        """
        import json
        import math
        import re

        if self.grounder is None or not images or not hasattr(self.backend, "ask"):
            return None
        from examples.euler.medication_sorting.grounding import pixel_to_world

        head = images.get("head_cam")
        if head is None:
            return None
        # 单点：报物体中心（比包围盒更稳）；误差主要来自物体混淆，靠描述区分
        ph, pw = head.shape[:2]
        prompt = (
            f"这是机器人头部相机俯视桌面（{pw}x{ph}）。"
            f"找出「{object_desc}」的中心。"
            f"只能用 0~{pw-1} 的 u 和 0~{ph-1} 的 v。"
            '只输出一行 JSON，不要代码块：{"u":整数,"v":整数}'
        )
        try:
            raw = self.backend.ask([_to_jpg(head)], prompt)
        except Exception:
            return None
        # 兼容代码块 / 前后说明文字 / 浮点数
        u = v = None
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
            u, v = obj.get("u"), obj.get("v")
            if u is not None and v is not None:
                u, v = int(round(float(u))), int(round(float(v)))
        except (TypeError, ValueError, KeyError):
            u = v = None
        if u is None or v is None:
            return None
        h, w = head.shape[:2]
        if not 0 <= u < w or not 0 <= v < h:
            return None
        x, y = pixel_to_world(self.grounder, u, v)
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return (float(x), float(y))

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 运动
    # ------------------------------------------------------------------
    def move_to(
        self,
        arm,
        x,
        y,
        z,
        rotation=None,
        stage="move",
        target=None,
        allow=False,
        carried=None,
        max_velocity=0.3,
        max_acceleration=0.25,
    ) -> SkillResult:
        """手臂运动到世界坐标 (x,y,z)。rotation 缺省用当前手朝向。

        target：要抓/接触的物体 key，规划时允许接近它；allow：是否允许接触 target。
        carried：已抓持的物体（抬升/搬运时传入，碰撞检查会忽略它与手臂的接触）。
        返回结构化结果；调用者必须检查 ok，不可将失败当成到位。
        """
        import numpy as np

        self.trial.sync()
        if rotation is None:
            _, rotation = self.scene.end_pose(arm)
        try:
            self.trial.move(
                arm,
                np.array([x, y, z]),
                rotation,
                stage,
                target=target,
                allow=allow,
                carried=carried,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
            )
            return SkillResult(True, "ok")
        except Exception as exc:  # noqa: BLE001
            return SkillResult(False, classify(exc), f"{type(exc).__name__}: {exc}")


    def step_move_to(
        self,
        arm,
        target,
        stage="move",
        target_key=None,
        carried=None,
        step=0.04,
        max_steps=40,
        rotation=None,
        allow=None,
    ) -> SkillResult:
        """小步移动到 target：每步只走 step（默认 4cm），每步重新解一次 IK。

        保留目标朝向和接触约束；分段并不保证可达或无碰撞。
        """
        import numpy as np

        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(step) or step <= 0:
            return SkillResult(False, "unknown", "Invalid step target or step size")
        if allow is None:
            allow = target_key is not None or carried is not None
        self.trial.sync()
        if rotation is None:
            _, rotation = self.scene.end_pose(arm)
        rotation = np.array(rotation, copy=True)
        for _ in range(max_steps):
            self.trial.sync()
            p, _ = self.scene.end_pose(arm)
            d = target - p
            dist = float(np.linalg.norm(d))
            if dist <= step * 1.01:
                m = self.move_to(
                    arm,
                    target[0],
                    target[1],
                    target[2],
                    rotation,
                    stage,
                    target=target_key,
                    allow=allow,
                    carried=carried,
                )
                if m.ok:
                    return m
                # 精确到目标的最后一步 IK 失败，但已经离目标很近 → 接受当前位形，
                # 后续前伸/微调会补上这点偏差，不需要因为最后一小步就整个失败。
                if dist <= 0.03:
                    return SkillResult(True, "ok", detail={"approx": True, "dist": float(dist)})
                return m
            nxt = p + d / dist * step
            m = self.move_to(
                arm, nxt[0], nxt[1], nxt[2], rotation, stage + "_mid", target=target_key, allow=allow, carried=carried
            )
            if not m.ok:
                return m
        return SkillResult(False, "unknown", "step_move_to: step budget exhausted")

    def micro_adjust(self, arm, dx=0.0, dy=0.0, dz=0.0, target=None, carried=None) -> SkillResult:
        """微调当前手位置。先同步场景到当前实际状态，再读当前手位置。

        target=要抓的物体时允许接触它；carried=已抓持物体（搬运微调时传）。
        """
        self.trial.sync()
        p, r = self.scene.end_pose(arm)
        return self.move_to(
            arm,
            p[0] + dx,
            p[1] + dy,
            p[2] + dz,
            r,
            "micro_adjust",
            target=target,
            allow=target is not None or carried is not None,
            carried=carried,
        )

    # ------------------------------------------------------------------
    # 夹爪
    # ------------------------------------------------------------------
    def validate_pick_point(self, key, xy) -> SkillResult:
        """Simulation identity guard; no substitution of true XY for visual XY."""
        import numpy as np

        self.trial.sync()
        xy = np.asarray(xy, dtype=float)
        p, _ = self.scene.body_pose(key)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.linalg.norm(xy - p[:2]) > 0.025:
            return SkillResult(False, "target_mismatch", "Visual point is not within 25 mm of the task box")
        return SkillResult(True, "ok")


    def grasp(self, arm, closure=1.0, seconds=1.2) -> SkillResult:
        """闭合夹爪到 closure（0-1）。"""
        try:
            self.trial.ramp_gripper(arm, closure, seconds, "grasp")
            return SkillResult(True, "ok")
        except Exception as exc:  # noqa: BLE001
            return SkillResult(False, classify(exc), f"{type(exc).__name__}: {exc}")

    def release(self, arm, seconds=1.2) -> SkillResult:
        """张开夹爪。"""
        return self.grasp(arm, 0.0, seconds)

    def check_grasp(self, key, initial_z, rise=0.02) -> bool:
        """物体是否跟着试抬上升；接触判定之后仍需通过此检查。

        2cm 是保守阈值——只要物体被稳定带动就认为夹持成立，
        不要求抬升幅度很大（避免把「能夹住」误判成「没夹住」）。
        """
        self.trial.sync()
        p, _ = self.scene.body_pose(key)
        return p[2] - initial_z >= rise

    def check_contact(self, arm, key) -> bool:
        """接触确认：夹爪手指是否与物体有实际接触（基于接触对）。

        返回 True 只表示手指碰到物体，不代表夹稳或能够搬运。
        """
        import mujoco

        self.trial.sync()
        mujoco.mj_forward(self.scene.m, self.scene.d)  # 刷新接触数据
        binding = self.scene.arms[arm]
        obj_geoms = self.scene.body_geoms(key, descendants=True)
        finger_geoms = binding.finger_geom_ids
        for contact in self.scene.d.contact:
            if contact.dist >= 0:  # dist<0 才是穿透接触
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in finger_geoms and b in obj_geoms) or (b in finger_geoms and a in obj_geoms):
                return True
        return False

    def release_ready(self, key, bin_key) -> bool:
        """Simulation geometry gate: object must be inside or directly above its bin.

        This is a simulator oracle, not a real-world visual measurement. It does
        not replace the independent visual check or post-release verification.
        """
        import numpy as np

        from examples.euler.medication_sorting.scene import vertices

        self.trial.sync()
        s = self.scene
        floors = [g for g in s.body_geoms(bin_key) if int(s.m.geom_type[g]) == 6 and s.m.geom_size[g, 2] < 0.015]
        floor = min(floors, key=lambda g: s.m.geom_pos[g, 2])
        origin = s.d.geom_xpos[floor]
        rotation = s.d.geom_xmat[floor].reshape(3, 3)
        item = (vertices(s, key) - origin) @ rotation
        rim = ((vertices(s, bin_key) - origin) @ rotation)[:, 2].max()
        bottom = item[:, 2].min()
        # Release can occur inside the bin, unlike the stricter hover predicate;
        # unlike landed placement it must contain the entire object footprint.
        return bool(
            np.isfinite(item).all()
            and np.all(np.abs(item[:, :2]) <= s.m.geom_size[floor, :2] - 0.003)
            and s.m.geom_size[floor, 2] - 0.002 <= bottom <= rim + 0.12
        )


    def grasp_staged(self, arm) -> SkillResult:
        """三段式闭合 + 握持：0.4→0.7→1.0，给物体时间被夹稳（对齐 pick_place）。"""
        try:
            for closure, seconds in [(0.4, 1.2), (0.7, 1.5), (1.0, 2.0)]:
                self.trial.ramp_gripper(arm, closure, seconds, "grasp_staged")
            self.trial.hold(30, "grip_hold")
            return SkillResult(True, "ok")
        except Exception as exc:  # noqa: BLE001
            return SkillResult(False, classify(exc), f"{type(exc).__name__}: {exc}")

    def grasp_preset(self, arm, key):
        """夹爪闭合预设：三段式闭合 + 接触确认。

        成功结果中 carried 仅供试抬；必须验证物体跟随之后才允许搬运。
        """
        try:
            closed = self.grasp_staged(arm)
            if not closed.ok:
                return closed
            if not self.check_contact(arm, key):
                return SkillResult(False, "lost_grasp", "No finger contact after closure")
            # 建立抓持关系：之后抬升/搬运时碰撞检查会忽略「手-物体」接触
            carried = self.scene.attach_object(arm, key, self.env.data.qpos.ravel().copy())
            return SkillResult(True, "ok", detail={"carried": carried})
        except Exception as exc:  # noqa: BLE001
            return SkillResult(False, classify(exc), f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # 判断
    # ------------------------------------------------------------------
    def judge(self, images, question: str):
        """视觉判断：让模型回答一个 yes/no 问题。

        返回 (bool|None, raw)。None 表示无法判断。
        """
        import re

        if not images or not hasattr(self.backend, "ask"):
            return None, "无图/无模型"
        head = images.get("head_cam")
        if head is None:
            return None, "无 head_cam"
        prompt = question + ' 只输出 JSON：{"answer": true} 或 {"answer": false}'
        try:
            raw = self.backend.ask([_to_jpg(head)], prompt)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        m = re.search(r'"answer"\s*:\s*(true|false)', raw, re.IGNORECASE)
        if not m:
            return None, raw[:60]
        return m.group(1).lower() == "true", raw[:60]



    def decide_grasp_action(self, arm, object_desc, images, hint=None, lesson=None):
        """到抓取点后，让大模型分析当前情况，统一给出下一步动作 + 自适应距离。

        返回 (action, reason, seen, dist_m)：
          action ∈ {"grasp","left","right","forward","backward","retreat","skip"}
          dist_m 是模型估算的偏移/还差多少（米，0 表示没给）
        hint：上一轮失败码的中文提示（如「夹住了但抬起来时滑落了」），
              喂回给大模型让它据此选择恢复动作——这是 agentic 的失败反馈闭环。
        """

        if not images or not hasattr(self.backend, "ask"):
            return "unknown", "无法判断", "（无图）", 0.0
        wrist = images.get("camera_left" if arm == "left" else "camera_right")
        head = images.get("head_cam")
        views = [v for v in (wrist, head) if v is not None]
        if not views:
            return "unknown", "无相机", "（无相机）", 0.0
        # 方向语义按握法区分：侧握 forward 是水平前伸，顶握 forward 是竖直下降
        if arm == "left":
            fwd_desc = "forward=夹爪还没伸到物体两侧，需要水平前伸（往物体方向）"
            back_desc = "backward=伸过头了，需要后退"
        else:
            fwd_desc = "forward=夹爪还在物体上方、没降到能夹住的高度，需要竖直下降"
            back_desc = "backward=降得太低/插进物体，需要抬高"
        hint_line = f"\n【上一轮结果】{hint}\n" if hint else ""
        lesson_line = f"【历史经验】同类失败以前用这些动作奏效过：{lesson}，这次可优先考虑。\n" if lesson else ""
        prompt = (
            f"夹爪已经到「{object_desc}」的抓取点。下面两张图："
            f"第一张是{arm}腕部摄像头（近景，看夹爪两指之间），第二张是头部俯视图（看夹爪和物体的整体位置关系）。"
            f"{hint_line}{lesson_line}"
            f"请把两张图结合起来判断——腕部图看两指是否夹在物体两侧、物体离手指远近，俯视图看夹爪整体有没有对准：\n"
            f"① action：能夹住=grasp；物体偏左=left；偏右=right；{fwd_desc}=forward；"
            f"{back_desc}=backward；需退后=retreat；抓不了=skip；\n"
            f"② dist_cm：估算还需要移动多少厘米（数字，例如 3.5；已对齐/grasp 则填 0）；\n"
            f"③ seen：一句话（≤30字）描述物体相对夹爪的位置；\n"
            f"④ reason：一句话（≤20字）说明为什么。\n"
            "只输出 JSON（action 放第一个），不要代码块："
            '{"action":"grasp","dist_cm":0,"seen":"...","reason":"..."}'
        )
        # 依次尝试多种图组合，确保一定有大模型分析：双图 → 只腕部 → 只头部
        raw = ""
        for combo in (views, [wrist] if wrist is not None else None, [head] if head is not None else None):
            if not combo:
                continue
            try:
                raw = self.backend.ask([_to_jpg(v) for v in combo], prompt)
            except Exception:
                raw = ""
            if raw.strip():
                break
        if not raw.strip():
            return "unknown", "模型无响应", "（模型未返回内容）", 0.0
        return parse_action(raw, {"grasp", "left", "right", "forward", "backward", "retreat", "skip"})

    def decide_place_action(self, arm, object_desc, bin_desc, images, hint=None):
        """放置阶段：大模型看画面判断物体是否在筐口正上方，决定微调方向。

        返回 (action, reason, seen, dist_m)：action ∈ {"aligned","left","right","forward","backward"}。
        hint：上一轮失败码的中文提示，喂回给大模型做恢复决策。
        """

        if not images or not hasattr(self.backend, "ask"):
            return "unknown", "无法判断", "（无图）", 0.0
        # 放置判断只用头部俯视图：它同时看得到物体和筐口，能直接判断「在不在筐口范围内」
        head = images.get("head_cam")
        views = [head] if head is not None else []
        if not views:
            return "unknown", "无头部相机", "（无 head_cam）", 0.0
        hint_line = f"\n【上一轮结果】{hint}\n" if hint else ""
        prompt = (
            f"机器人正抓着「{object_desc}」，准备放进「{bin_desc}」。"
            f"这是头部俯视图，能同时看到夹爪抓着的物体和目标筐口。"
            f"{hint_line}"
            f"请判断这个物体现在是否已经落在筐口正上方（下放就能掉进筐里）：\n"
            f"① action：aligned（在筐口范围内，可以下放）/ left / right / forward / backward（需要往哪移）；\n"
            f"② dist_cm：还需移动多少厘米（aligned 填 0）；\n"
            f"③ seen：一句话（≤30字）描述物体相对筐口的位置。\n"
            "只输出 JSON（action 放第一个），不要代码块："
            '{"action":"aligned","dist_cm":0,"seen":"..."}'
        )
        raw = ""
        for combo in (views,):
            try:
                raw = self.backend.ask([_to_jpg(v) for v in combo], prompt)
            except Exception:
                raw = ""
            if raw.strip():
                break
        if not raw.strip():
            return "unknown", "模型无响应", "（无响应）", 0.0
        return parse_action(raw, {"aligned", "left", "right", "forward", "backward"})


    def decide_carry_recovery(self, arm, object_desc, bin_desc, images, hint=None):
        """搬运被下方物体挡住时，让大模型看画面决定怎么绕过去。

        可选项不止"抬高"：如果筐口某一侧空着，横着偏过去往往更省事。
        返回 (raise_cm, lateral_cm, reason)；lateral_cm 正数=画面右，负数=画面左。
        """
        import json
        import re

        if not images or not hasattr(self.backend, "ask"):
            return 6.0, 0.0, "模型无响应，默认抬高"
        head = images.get("head_cam")
        if head is None:
            return 6.0, 0.0, "无头部相机，默认抬高"
        hint_line = f"\n【上一轮结果】{hint}\n" if hint else ""
        prompt = (
            f"机器人抓着「{object_desc}」要搬到「{bin_desc}」，但水平搬运时下方被东西挡住、过不去。"
            f"这是头部俯视图，能看到筐口以及周围哪些方向是空的。{hint_line}"
            f"请像人一样决定怎么绕过去最省事：可以抬高从上方过，也可以从空着的那一侧横向绕过去（左右哪边空就往哪边偏）。\n"
            f"要求：优先横向绕（如果左右有空位），实在没空位才抬高。\n"
            '只输出 JSON，不要代码块：{"raise_cm":0,"lateral_cm":6,"reason":"右边空可以右移绕过"}'
        )
        try:
            raw = self.backend.ask([_to_jpg(head)], prompt)
        except Exception:
            return 6.0, 0.0, "模型无响应，默认抬高"
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return 6.0, 0.0, "输出无法解析，默认抬高"
        try:
            obj = json.loads(m.group(0))
        except (TypeError, ValueError):
            return 6.0, 0.0, "输出无法解析，默认抬高"
        try:
            raise_cm = float(min(max(float(obj.get("raise_cm", 0)), 0.0), 12.0))
            lateral_cm = float(min(max(float(obj.get("lateral_cm", 0)), -12.0), 12.0))
        except (TypeError, ValueError):
            return 6.0, 0.0, "数值非法，默认抬高"
        reason = str(obj.get("reason", "") or "").strip()[:40]
        return raise_cm, lateral_cm, reason

    def plan_placement(self, arm, object_desc, bin_desc, images):
        """放置规划：大模型看筐内情况，决定放筐里的哪个位置（相对筐中心的偏移）。

        返回 (ready, dx_m, dy_m, seen)：ready=能否下放，dx/dy=相对筐中心的偏移（米）。
        """
        import json
        import re

        if not images or not hasattr(self.backend, "ask"):
            return True, 0.0, 0.0, "（无图，放筐中心）"
        head = images.get("head_cam")
        if head is None:
            return True, 0.0, 0.0, "（无 head_cam，放筐中心）"
        prompt = (
            f"机器人抓着「{object_desc}」，准备放进「{bin_desc}」。"
            f"这是头部俯视图，能同时看到：被抓的物体、目标筐口、以及筐里已经放进去的其他物体。"
            f"请像人一样决定这一件该放在筐里的哪个位置：\n"
            f"① ready：现在能不能下放（物体是否已大致在筐口上方）；\n"
            f"② dx_cm / dy_cm：相对筐中心往哪偏（厘米，正数=画面右/远，负数=画面左/近）；\n"
            f"③ seen：一句话描述筐内已有什么、物体在哪。\n"
            f"重要：筐里后面还要放同类物体，所以【不要放正中间】，应偏向一侧（如 -4~-2cm），"
            f"给下一件留出空位；如果筐里已有物体，就挑空的那一侧放，避免叠在一起。\n"
            "只输出 JSON：{\"ready\":true,\"dx_cm\":0,\"dy_cm\":0,\"seen\":\"...\"}"
        )
        try:
            raw = self.backend.ask([_to_jpg(head)], prompt)
        except Exception:
            return True, 0.0, 0.0, "（模型无响应，放筐中心）"
        obj = {}
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
        except Exception:
            obj = {}

        def num(name):
            try:
                return float(obj.get(name, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        ready = bool(obj.get("ready", True)) if isinstance(obj, dict) else True
        dx = max(-0.06, min(0.06, num("dx_cm") / 100.0))
        dy = max(-0.06, min(0.06, num("dy_cm") / 100.0))
        return ready, dx, dy, str(obj.get("seen", "") or "")

    def survey_scene(self, images, goal=None, keys=None):
        """开局：让大模型观察场景，说出看到什么、打算怎么做、按什么顺序处理。

        goal：任务目标（自然语言）；keys：候选物体 key，用来让模型给出处理顺序。
        返回 (seen, plan, order)：order 是模型建议的顺序（可能是 None 或非法排列）。
        顺序只作建议：调用方必须校验它是 keys 的合法排列，否则回退原顺序。
        """
        import json
        import re

        if not images or not hasattr(self.backend, "ask"):
            return "无法观察场景", "按固定顺序处理", None
        head = images.get("head_cam")
        if head is None:
            return "无法观察场景", "按固定顺序处理", None
        task_line = f"任务目标：{goal}。\n" if goal else ""
        order_line = ""
        if keys:
            order_line = (
                f"③ 按什么顺序处理最省事。候选物体固定为这些 key：{list(keys)}，"
                "每个都要处理一次、不重复不遗漏。\n"
            )
        prompt = (
            "这是机器人工作台的头部俯视图。"
            f"{task_line}"
            "请只根据画面里真正看到的东西，用简洁自然的中文说出："
            "① 桌面上有哪些物体（形状 + 大致方位）；"
            "② 你打算先做什么。"
            f"{order_line}"
            '输出 JSON：{"seen":"...","plan":"...","order":["key1","key2"]}'
        )
        try:
            raw = self.backend.ask([_to_jpg(head)], prompt)
        except Exception:
            return "无法观察场景", "按固定顺序处理", None
        seen = re.search(r'"seen"\s*:\s*"([^"]*)"', raw)
        plan = re.search(r'"plan"\s*:\s*"([^"]*)"', raw)
        order = None
        m = re.search(r'"order"\s*:\s*\[([^\]]*)\]', raw)
        if m:
            try:
                parsed = json.loads("[" + m.group(1) + "]")
                if all(isinstance(item, str) for item in parsed):
                    order = parsed
            except (TypeError, ValueError):
                order = None
        return (
            seen.group(1) if seen else "（模型未描述）",
            plan.group(1) if plan else "按固定顺序处理",
            order,
        )


def parse_action(raw, allowed):
    """Only complete, valid JSON may authorize a robot action."""
    import json
    import math

    try:
        text = raw.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        obj = json.loads(text)
        if not isinstance(obj, dict) or obj.get("action") not in allowed:
            raise ValueError("Invalid action")
        dist = obj.get("dist_cm", 0)
        if isinstance(dist, bool) or not isinstance(dist, (int, float)) or not math.isfinite(dist) or dist < 0:
            raise ValueError("Invalid distance")
        # 距离转米，上限 12cm：保留大模型的自适应估计（侧握前伸需要 6~9cm，2cm 会永远够不到）
        return obj["action"], str(obj.get("reason", "")), str(obj.get("seen", "")), min(0.12, dist / 100)
    except (TypeError, ValueError, AttributeError):
        return "unknown", "模型响应不完整或不合法", "", 0.0


_TMP_DIR = None


def _to_jpg(arr):
    """ndarray → jpg 路径（模型后端需要文件路径）。

    复用同一个临时目录 + 同一个文件名：每次调用覆盖写，避免每次观测都新建目录。
    """
    import pathlib
    import tempfile

    from PIL import Image

    global _TMP_DIR
    if _TMP_DIR is None:
        _TMP_DIR = pathlib.Path(tempfile.mkdtemp(prefix="med_agent_"))
    p = _TMP_DIR / "view.jpg"
    Image.fromarray(arr).save(p, "JPEG", quality=85)
    return str(p)
