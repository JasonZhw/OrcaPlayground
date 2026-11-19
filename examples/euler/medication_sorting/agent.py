"""Agent-as-Policy 执行框架（对齐 agentic robotics 的「感知→决策→执行→校验」闭环）。

设计思路
--------
大模型是决策大脑，不直接碰底层运动。每个动作都返回结构化 SkillResult（含失败码），
失败码会翻译成中文提示喂回给大模型，让它据此选择恢复动作——这就是 agentic 的
「失败反馈闭环」，而不是脚本式的一次失败就停。

每个物体走两个阶段，两阶段共享同一套循环：

  抓取阶段 Grasp
    locate_object   大模型看俯视图定位物体（单应矩阵接地）
    approach/descend 到预抓取点（带标定偏移），张开夹爪
    align_grasp      大模型看腕部+俯视图逐轮决策 left/right/forward/backward/grasp
                     （药瓶侧握 = Y 方向前伸，药盒顶握 = Z 方向下压，同一套循环）
    close            三段式闭合 + 接触确认 → 建立抓持关系 carried
    test_lift        抬 3cm，物体跟着起来且仍有接触才算「夹稳」
    verify_grasp     失败（lost_grasp）→ 喂回失败码，有界重抓 ≤3 次

  放置阶段 Place
    locate_bin       大模型定位目标筐
    plan_placement   大模型看筐内已有几件，决定这一件放筐里哪个位置（相对筐中心偏移）
    lift_safe/carry  先抬到筐口上方（释放点 +8cm），再水平搬运（补偿物体相对手腕的偏移）
                     · 搬运被挡时让大模型看画面决定怎么绕：优先横向绕空位、其次抬高，
                       最多 6 次，每轮把上一轮尝试的参数喂回去，避免重复
                     · 搬运用抓取时的固定朝向，全程不转手腕
    align_place      【几何是唯一放行条件】物体在筐口范围内才下放；
                     模型说 aligned 但几何没过时，按几何误差朝筐心微调一小步
                     （模型偶尔会把左右筐看混，不能只信它）
    release          当前位置直接释放（不再做 Z 下放——筐壁总会挡住，纯属多余动作）
    settle           释放后 step 200 帧让物体自由落进筐底，再校验
    verify_placement 视觉 + 几何双重校验是否入筐

经验库 lessons（case-based memory，不训练模型）
  agent_out/lessons.json 累积「失败码 → 成功恢复动作」。下次遇到同类失败时，
  把「历史上哪些动作救回来过」塞进 prompt。跑得越多经验越丰富。

工程 trick（保留）
  - arm-only 模型：裁掉躯干/腿/头，且关闭它们的碰撞体（保留视觉），避免手臂撞躯干
  - 手指全链接触豁免：侧握时手指中段碰物体不算碰撞
  - 抓持关系 carried：抬升/搬运时碰撞检查忽略「手-物体」接触
  - 大跳移动失败时降级小步（step_move_to），每步重解 IK
  - 释放前接触检查放在下放【之前】（下放到筐底后物体被托住、接触自然松，那不是掉）
  - 物体相对手腕的偏移补偿：侧握物体挂在手腕旁 20cm，投放点要让【物体】对准筐口
  - 到位判定容差 1.5cm + 二次保持：高位置伺服有重力残差，多保持一会儿再判
  - 不做人为高度上限：够不够得到交给 IK 决定，失败由恢复逻辑接手
  - 模型输出非法 / 微调失败都不终止任务，一律反馈给模型重新规划
  - 药盒顶握步长自适应（2~6cm），药瓶侧握保持固定 6cm

安全规则
  - 所有运动都经过碰撞/限位/伺服余量检查
  - 失败路径绝不打开「可能抓着东西」的夹爪
  - 未知感知 / 不可执行的运动一律阻止，不猜测
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from examples.euler.medication_sorting import lessons as lesson_store
from examples.euler.medication_sorting.task import TaskSpec
from examples.euler.medication_sorting.failures import CODE_HINT, SkillResult, classify

class StageFailure(RuntimeError):
    def __init__(self, result):
        super().__init__(result.message)
        self.result = result


def execute_tasks(tools, spec, home_poses, out, *, grasp_builder=None, placement_builder=None, verify=None):
    """跑完整个任务：对每件物体依次执行 定位 → 抓取 → 搬运 → 入框，最后复位。

    这是一条线性的状态机，四个阶段在函数内用横幅标出：

        阶段 1  定位与接近   模型报像素 → 抓取点几何 → 分步靠近
        阶段 2  抓取         对齐 → 闭合 → 接触确认 → 试抬校验
        阶段 3  搬运         计算释放点 → 抬安全高度 → 水平移动
        阶段 4  入框         对齐 → 几何放行 → 释放 → 落地校验

    设计约束：
    - 每次运动都在 `Trial.move` 内校验实际到位姿态，不通过就返回失败，不会"假装到了"；
    - 只有 IK / 路径类失败允许一次分段重试，其余失败直接反馈给模型重新决策；
    - 一旦判定 epblocked，本回合不再发送任何指令（包括释放与复位）；
    - `grasp_builder / placement_builder / verify` 可注入，便于无仿真单元测试。
    """
    if grasp_builder is None or placement_builder is None or verify is None:
        from examples.euler.medication_sorting.geometry import (
            grasp_target, placement_satisfied, placement_targets,
        )
        # placement_satisfied 既作默认 verify，也用作「是否已入筐」的几何判据

        grasp_builder = grasp_builder or grasp_target
        placement_builder = placement_builder or placement_targets
        verify = verify or placement_satisfied

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keys = spec.keys
    scene = tools.scene
    placed = dict.fromkeys(keys, False)
    report = {"status": "running", "placed": placed, "object": None, "stage": "start", "events": []}

    def record(stage, result=None, **detail):
        report["stage"] = stage
        event = {"object": report["object"], "stage": stage, "time": datetime.now(timezone.utc).isoformat(), **detail}
        if result is not None:
            event.update(ok=result.ok, code=result.code, message=result.message)
        report["events"].append(event)
        temporary = out / "episode_result.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(out / "episode_result.json")

    def require(result):
        if not isinstance(result, SkillResult):
            raise StageFailure(SkillResult(False, "unknown", "Tool returned an invalid result"))
        record(report["stage"], result)
        if not result.ok:
            raise StageFailure(result)
        return result

    def check(condition, code, message):
        require(SkillResult(bool(condition), "ok" if condition else code, message))

    def move(arm, position, rotation, stage, *, target=None, allow=False, carried=None):
        position, rotation = np.asarray(position), np.asarray(rotation)
        record(stage, position=position.tolist(), rotation=rotation.tolist())
        # 搬运（carried 非空）用更慢的加减速，避免物体因惯性从夹爪滑落
        mv = 0.12 if carried is not None else 0.3
        ma = 0.06 if carried is not None else 0.25
        result = tools.move_to(
            arm,
            *position,
            rotation=rotation,
            stage=stage,
            target=target,
            allow=allow,
            carried=carried,
            max_velocity=mv,
            max_acceleration=ma,
        )
        if result.code in ("ik_no_solution", "plan_collision") and not result.ok:
            record(stage, result)
            # Explicit rotation AND contact permission survive the retry.
            record(stage + "_retry", position=position.tolist(), rotation=rotation.tolist())
            result = tools.step_move_to(
                arm,
                position,
                rotation=rotation,
                stage=stage + "_retry",
                target_key=target,
                allow=allow,
                carried=carried,
                step=0.03,
                max_steps=12,
            )
        require(result)

    def locate(description, stage):
        record(stage)
        for attempt in range(3):
            xy = tools.point(tools.observe(), description)
            if xy is not None and np.shape(xy) == (2,) and np.isfinite(xy).all():
                record(stage, SkillResult(True, "ok"), xy=list(xy), attempt=attempt + 1)
                print("  [工具] 视觉定位完成", flush=True)
                return np.asarray(xy)
        raise StageFailure(SkillResult(False, "unknown", "No valid visual location after 3 observations"))

    try:
        record("start")
        if not keys or len(set(keys)) != len(keys) or any(k not in spec.objects for k in keys):
            raise StageFailure(SkillResult(False, "unknown", "Expected a nonempty set of known task objects"))
        # 开局：大模型观察场景，说出看到什么、打算怎么做
        print("\n  🔭 观察场景 → 决定任务", flush=True)
        try:
            observed = tools.survey_scene(tools.observe(), keys=keys)
            _seen, _plan = observed[0], observed[1]
            _order = observed[2] if len(observed) > 2 else None
            print(f"  [推理] {_seen}", flush=True)
            print(f"  [执行] {_plan}", flush=True)
            # 顺序只作建议：必须是合法排列，且同一条手臂上"先近后远"，才采纳。
            if _order and len(_order) == len(keys) and set(_order) == set(keys):
                if _order_feasible(spec, scene, _order):
                    keys = tuple(_order)
                    placed.clear()
                    placed.update({k: False for k in keys})
                    print(f"  [执行] 按模型建议排序：{' → '.join(keys)}", flush=True)
                else:
                    print("  [执行] 模型建议的顺序会让远处物体先抓（近处还挡着），按配置顺序处理", flush=True)
            elif _order:
                print(f"  [警告] 模型给的顺序不是合法排列，按配置顺序处理：{_order}", flush=True)
        except Exception as _exc:  # noqa: BLE001
            print(f"  ⚠️  开局观察失败：{_exc}", flush=True)

        # ==================================================================
        # 阶段 1  定位与接近：模型报像素 → 抓取点几何 → 分步靠近
        # ==================================================================
        for key in keys:
            report["object"] = key
            arm = spec.objects[key].arm
            bin_key = spec.objects[key].dest
            print(
                f"\n  🎯 {spec.objects[key].desc} → {spec.bins[bin_key]}（{arm}臂{'侧握' if arm == 'left' else '顶握'}）",
                flush=True,
            )
            xy = locate(spec.objects[key].desc, "locate_object")
            if arm == "right":
                # 右臂药盒：确认大模型指点确实落在目标药盒上（防抓错物体），
                # 只做身份校验，不做几何测量对齐（对齐交给大模型判断）。
                require(tools.validate_pick_point(key, xy))
            tools.trial.sync()
            record("build_grasp")
            gt = grasp_builder(scene, arm, key, center_xy=xy)
            # 把「为什么是这个坐标」说出来：物体在哪 → 抓取点怎么推出来 → 打算怎么接近
            obj_xy = np.asarray(xy, dtype=float)
            grip_xy = np.asarray(gt["position"][:2], dtype=float)
            gap_cm = float(np.linalg.norm(grip_xy - obj_xy)) * 100.0
            lift_cm = (float(gt["approach"][2]) - float(gt["position"][2])) * 100.0
            grip_mode = "侧握（水平前伸）" if arm == "left" else "顶握（竖直下压）"
            offset_dir = "正后方" if arm == "left" else "正上方"
            print(
                f"  [推理] {grip_mode}，抓取点取{offset_dir} "
                f"{gap_cm:.0f}cm，预备点比抓取点再高 {lift_cm:.0f}cm；接近方式=先水平对齐、再垂直下压",
                flush=True,
            )
            # 安全接近：先保持当前高度【水平】平移到预备抓取点的 XY（不碰物体），
            # 再垂直下降到预备抓取点，最后下降到抓取点。避免斜向直接扑过去扫到物体。
            cur_p, cur_r = scene.end_pose(arm)
            move(
                arm,
                [gt["approach"][0], gt["approach"][1], cur_p[2]],
                cur_r,
                key + "_approach_xy",
                target=key,
            )
            move(arm, gt["approach"], gt["rotation"], key + "_approach", target=key)
            move(arm, gt["position"], gt["rotation"], key + "_descend", target=key, allow=True)
            record("open")
            require(tools.release(arm))

            carried = None
            skipped = False
            direction = {
                "left": (1, 0, 0),
                "right": (-1, 0, 0),
                "forward": (0, -1, 0) if arm == "left" else (0, 0, -1),
                "backward": (0, 1, 0) if arm == "left" else (0, 0, 1),
                "retreat": (0, 1, 0) if arm == "left" else (0, 0, 1),
            }
            # 抓取阶段：对齐 → 闭合 → 试抬验证。试抬失败（lost_grasp）时把失败码喂回大模型，
            # 做有界重抓（agentic 的失败反馈闭环），而不是一次失败就停。
            grasp_hint = None
            failure_code = None  # 最近一次失败码：用于经验库「失败->成功恢复动作」

        # ==================================================================
        # 阶段 2  抓取：对齐 → 闭合 → 接触确认 → 试抬校验
        # ==================================================================
            for regrasp in range(3):
                for attempt in range(6):
                    record("align_grasp", regrasp=regrasp + 1, attempt=attempt + 1)
                    action, reason, seen, distance = tools.decide_grasp_action(
                        arm,
                        spec.objects[key].desc,
                        tools.observe(),
                        hint=grasp_hint,
                        lesson=lesson_store.suggest(failure_code) if failure_code else None,
                    )
                    record("align_grasp", action=action, reason=reason, seen=seen)
                    print(f"  [推理] {seen or '（模型未描述）'}", flush=True)
                    print(f"  [执行] {action}{(' — ' + reason) if reason else ''}", flush=True)
                    if action == "skip":
                        skipped = True
                        break
                    if action not in (*direction, "grasp"):
                        # 大模型这次输出无法解析成有效动作（unknown）→ 不终止，
                        # 反馈格式要求后重新判断；连续失败才由预算兜底。
                        grasp_hint = (
                            "上一次输出无法解析。请严格只输出 JSON，action 必须是 "
                            "grasp/left/right/forward/backward 之一。"
                        )
                        print(f"  [执行] 输出无法解析（{action}）→ 重新判断", flush=True)
                        continue
                    if action in direction:
                        check(np.isfinite(distance) and distance >= 0, "unknown", "Invalid adjustment distance")
                        if action == "forward":
                            if arm == "left":
                                # 药瓶侧握：水平前伸，保持原来的固定 6cm
                                step = 0.06
                            else:
                                # 药盒顶握：竖直下压，步长跟随模型估计（2~6cm），
                                # 接近盒底时自动收敛，避免一步压到够不到的位置。
                                step = min(max(distance if distance > 0 else 0.06, 0.02), 0.06)
                            delta = np.asarray(direction["forward"]) * step
                        else:
                            # 左右/后退微调 2~4cm，交给大模型判断是否使用
                            step = min(max(distance, 0.02), 0.04) if distance > 0 else 0.03
                            delta = np.asarray(direction[action]) * step
                        record("adjust_grasp", delta=delta.tolist())
                        tools.trial.sync()
                        print(f"  [工具] micro_adjust({action} {step * 100:.1f}cm)", flush=True)
                        moved = tools.micro_adjust(arm, dx=delta[0], dy=delta[1], dz=delta[2], target=key)
                        record(report["stage"], moved)
                        if not moved.ok:
                            # 微调失败（到手臂极限/碰撞）→ 不终止，把失败反馈给大模型，
                            # 让它重新规划：换方向、或改判直接夹紧（也许当前位置已经够得着）。
                            if failure_code:
                                lesson_store.record(failure_code, action, worked=False)
                            failure_code = moved.code
                            if action == "forward" and arm == "right":
                                grasp_hint = (
                                    "已经降到能到达的最深位置（再往前就超出关节/臂展极限了），"
                                    "不要再 forward/继续下降。请直接 grasp——顶握时指尖通常已经到物体底部，"
                                    "再降没有意义；如果确实夹不住，就改 left/right/backward。"
                                )
                            else:
                                grasp_hint = (
                                    f"{action} 微调失败（{moved.code}: 到手臂极限或碰到东西）。"
                                    "不要再沿这个方向硬走，请改判：直接 grasp，或换左右/后退方向。"
                                )
                            print(f"  [工具] micro_adjust({action}) ✗ {moved.code} → 反馈大模型重新规划", flush=True)
                            continue
                        if failure_code:
                            # 上一个失败之后这个动作成功了 → 记一条成功经验
                            lesson_store.record(failure_code, action, worked=True)
                            failure_code = None
                        # 微调成功 → 回到循环顶部重新观察、重新判断（关键：不能掉到 grasp 分支）
                        continue
                    check(action == "grasp", "unknown", "Grasp perception is unknown; no closure authorized")
                    record("close")
                    result = tools.grasp_preset(arm, key)
                    record(report["stage"], result)
                    if not result.ok:
                        # 没夹到物体 → 张开夹爪、反馈给大模型，【继续本轮循环】重新前伸再判断，
                        # 而不是直接停：一个 TOOL 执行完，根据它的结果再去判断下一步。
                        if failure_code:
                            lesson_store.record(failure_code, "grasp", worked=False)
                        failure_code = result.code
                        grasp_hint = CODE_HINT["lost_grasp"]
                        print(f"  [工具] grasp_preset ✗ {result.message} → 张开重试", flush=True)
                        require(tools.release(arm))
                        continue
                    if failure_code:
                        lesson_store.record(failure_code, "grasp", worked=True)
                        failure_code = None
                    carried = result.detail.get("carried")
                    break
                if skipped or carried is None:
                    break
                # 试抬验证：接触到物体不代表夹稳，抬一下看物体是否跟着。
                tools.trial.sync()
                initial_z = float(scene.body_pose(key)[0][2])
                p, rotation = scene.end_pose(arm)
                move(arm, p + [0, 0, 0.03], rotation, key + "_test_lift", carried=carried)
                record("verify_grasp")
                if tools.check_grasp(key, initial_z, rise=0.02) and tools.check_contact(arm, key):
                    break
                # 试抬失败：张开、反馈、回预抓取点重抓
                record("verify_grasp", SkillResult(False, "lost_grasp", "Object did not follow the test lift"))
                require(tools.release(arm))
                grasp_hint = CODE_HINT["lost_grasp"]
                carried = None
                move(arm, gt["position"], gt["rotation"], key + "_regrasp", target=key, allow=True)
            if skipped:
                hp, hr = home_poses[arm]
                move(arm, hp, hr, key + "_home")
                record("skipped")
                continue
            check(carried is not None, "lost_grasp", "Grasp budget exhausted after regrasp attempts")

            # 已夹住 → 切换到入框状态
            print("  [执行] 夹持确认通过 → 切换入框状态", flush=True)
            # 确认模型此刻还能看见筐（找不到会抛 StageFailure）；落点坐标用仿真真值更准
            locate(spec.bins[bin_key], "locate_bin")
            tools.trial.sync()

        # ==================================================================
        # 阶段 3  搬运：计算释放点 → 抬安全高度 → 水平移动（被挡则由模型决定绕行）
        # ==================================================================
            record("build_placement")
            p, rotation = scene.end_pose(arm)
            # 夹住物品后手腕朝向锁定 = 抓取时的朝向，搬运全程不转（否则手腕晃动很别扭）
            rotation = np.asarray(gt["rotation"], dtype=float)
            targets = placement_builder(scene, key, carried, rotation, bin_key)
            # XY 目标 = 筐中心 减去「物体相对手腕的偏移」，让【物体】而不是手腕对准筐口。
            # 侧握时物体挂在手腕旁 20cm，直接用手腕对准会让物体偏出筐口。
            bin_p = scene.body_pose(bin_key)[0]
            obj_offset = np.asarray(rotation) @ np.asarray(carried.local_position)
            # 投放位置交给大模型决定：看筐里已经有什么、这件该放哪
            plan_ok, pdx, pdy, pseen = tools.plan_placement(
                arm, spec.objects[key].desc, spec.bins[bin_key], tools.observe())
            print(f"  [推理] {pseen or '（模型未描述）'}", flush=True)
            # 大模型的投放偏移限制在筐内安全范围（±4cm），避免模型给出过大偏移
            # 导致物体实际落点超出筐口（几何验证会判定释放区外）。
            pdx = float(np.clip(pdx, -0.04, 0.04))
            pdy = float(np.clip(pdy, -0.04, 0.04))
            object_xy = np.array([bin_p[0] + pdx, bin_p[1] + pdy])
            hand_xy = object_xy - obj_offset[:2]
            # 搬运悬停高度 = 释放点上方 8cm（比贴筐口再高 3cm，避免物体底部擦到筐沿）。
            above = np.array([hand_xy[0], hand_xy[1], targets["release"][2] + 0.08])
            # 安全高度：抬到能越过筐沿即可。搬运时如果物体下方碰到东西（IK/碰撞失败），
            # 不直接停、也不写死抬升量——交给大模型看画面判断"再抬高多少厘米"，循环推理执行。
            # 不做人为高度上限：真够不到 IK 会失败，由恢复逻辑处理
            safe = max(float(p[2]), float(above[2]))
            # 只在开始时抬一次安全高度；之后的重试都是"从当前位置再抬高"，不回原位
            move(arm, [p[0], p[1], safe], rotation, key + "_lift_safe", carried=carried)
            check(tools.check_contact(arm, key), "lost_grasp", "Contact lost after lift")
            carry_hint = None
            for carry_try in range(6):
                carry_stage = key + ("_carry" if carry_try == 0 else f"_carry_{carry_try}")
                try:
                    move(arm, [above[0], above[1], safe], rotation, carry_stage, carried=carried)
                    check(tools.check_contact(arm, key), "lost_grasp", "Contact lost during transport")
                    break
                except StageFailure as exc:
                    blocked = exc.result.code in ("ik_no_solution", "plan_collision")
                    if not blocked or carry_try >= 5:
                        raise
                    # 交给大模型看画面决定怎么绕：优先从空着的一侧横移，其次才抬高
                    raise_cm, lateral_cm, why = tools.decide_carry_recovery(
                        arm,
                        spec.objects[key].desc,
                        spec.bins[bin_key],
                        tools.observe(),
                        hint=carry_hint or exc.result.code,
                    )
                    print(f"  [推理] 搬运被挡，模型决定：{why or '（模型未说明）'}", flush=True)
                    if abs(lateral_cm) > 0.1:
                        above = above + np.array([lateral_cm / 100.0, 0.0, 0.0])
                    if raise_cm > 0.1:
                        # 从【当前实际高度】往上抬（单纯的抬高动作），不回原位、不重复 lift_safe
                        tools.trial.sync()
                        p_now, r_now = scene.end_pose(arm)
                        new_z = float(p_now[2]) + raise_cm / 100.0
                        move(arm, [p_now[0], p_now[1], new_z], r_now, key + f"_raise_{carry_try}", carried=carried)
                        safe = new_z
                    print(
                        f"  [工具] carry ✗ → 抬高 {raise_cm:.0f}cm、横移 {lateral_cm:+.0f}cm 后重试",
                        flush=True,
                    )
                    carry_hint = (
                        f"上一轮已尝试：抬高 {raise_cm:.0f}cm、横移 {lateral_cm:+.0f}cm，仍然过不去。"
                        "这次请换一个不同的绕法（更大的横移、换另一侧、或抬更高），不要重复同样的参数。"
                    )
                    record(
                        "carry_recovery",
                        message=f"Carry blocked; raise {raise_cm:.0f}cm lateral {lateral_cm:+.0f}cm",
                        raise_cm=raise_cm,
                        lateral_cm=lateral_cm,
                    )

            for attempt in range(6):
                record("align_place", attempt=attempt + 1)
                # 先看【几何】：物体已经在筐口范围内就直接可下放，不依赖大模型主观判断
                # （大模型偶尔会把左右筐看混，导致反复往反方向微调）。
                if tools.release_ready(key, bin_key):
                    print("  [执行] 几何检查通过（物体已在筐口范围）→ 可以下放", flush=True)
                    break
                action, reason, seen, distance = tools.decide_place_action(
                    arm,
                    spec.objects[key].desc,
                    spec.bins[bin_key],
                    tools.observe(),
                    hint="几何检查显示物体还不在筐口范围内（可能偏在一侧），请据此给出该往哪移。" if attempt > 0 else None,
                )
                record("align_place", action=action, reason=reason, seen=seen)
                print(f"  [推理] {seen or '（模型未描述）'}", flush=True)
                print(f"  [执行] {action}{(' — ' + reason) if reason else ''}", flush=True)
                if action == "aligned":
                    # 模型主观认为对齐、但几何检查没过 → 不放行，按【几何误差】朝筐心微调一小步
                    tools.trial.sync()
                    obj_now, _ = scene.body_pose(key)
                    err = np.asarray(bin_p[:2], dtype=float) - np.asarray(obj_now[:2], dtype=float)
                    step = np.clip(err, -0.02, 0.02)
                    if float(np.linalg.norm(step)) < 0.003:
                        step = np.array([0.0, 0.0])  # 误差已很小，别抖
                    print("  [执行] 模型认为已对齐，但几何未通过 → 朝筐心微调", flush=True)
                    require(tools.micro_adjust(arm, dx=step[0], dy=step[1], dz=0.0, carried=carried))
                    continue
                directions = {"left": (1, 0, 0), "right": (-1, 0, 0), "forward": (0, -1, 0), "backward": (0, 1, 0)}
                check(action in directions, "unknown", "Placement perception is unknown; release blocked")
                check(np.isfinite(distance) and distance >= 0, "unknown", "Invalid adjustment distance")
                delta = np.asarray(directions[action]) * min(0.02, distance if distance > 0 else 0.015)
                record("adjust_place", delta=delta.tolist())
                print(f"  [工具] micro_adjust({action} {min(0.02, distance) * 100:.1f}cm)", flush=True)
                require(tools.micro_adjust(arm, dx=delta[0], dy=delta[1], dz=delta[2], carried=carried))
            else:
                raise StageFailure(SkillResult(False, "missed_bin", "Alignment budget exhausted; release blocked"))

            # 接触检查放在【下放之前】：下放到筐底后物体被地面托住，手指接触自然松开，
            # 那不是掉，是正常落底。所以先确认「下放前还抓着」。
            check(tools.check_contact(arm, key), "lost_grasp", "Contact lost before lowering")
            tools.trial.sync()
            p, rotation = scene.end_pose(arm)
            # 不再做向下的 Z 下放（筐壁总会挡住，白白多一个动作）——直接在当前位置释放。
            # 同时把物体的实际位置打出来，方便排查落点。
            tools.trial.sync()
            obj_p, _ = scene.body_pose(key)
            ready = tools.release_ready(key, bin_key)
            print(f"  [执行] 释放前几何检查：{'通过' if ready else '未通过'}", flush=True)
            record("verify_release", object_position=obj_p.tolist(), hand_position=p.tolist())
            # 不再重复问大模型“能不能下放”——align_place 循环里已经确认过 aligned，
            # 而且这中间没有任何运动（不做 Z 下放），重复判定只会因模型抖动误否决释放。
            # 只保留几何检查（物体确实在筐口范围内）作为最后一道保险。
            check(
                tools.release_ready(key, bin_key), "missed_bin", "Actual object geometry is outside the release region"
            )

        # ==================================================================
        # 阶段 4  入框：对齐 → 几何放行 → 释放 → 落地校验
        # ==================================================================
            record("release")
            require(tools.release(arm))
            # 释放后让物体自由落进筐里再检查：不 step 的话它会停在释放高度，
            # 几何检查会误判成"还在筐口上方、没进筐"（实际只是还没掉下去）。
            tools.trial.hold(200, "settle_after_release")
            tools.trial.sync()
            record("verify_placement")
            visual, description = tools.judge(tools.observe(), f"{spec.objects[key].desc}是否已放进{spec.bins[bin_key]}？")
            tools.trial.sync()
            check(
                visual is True and bool(verify(scene, key, bin_key)),
                "missed_bin",
                f"Placement not confirmed by vision and geometry: {description}",
            )
            placed[key] = True
            hp, hr = home_poses[arm]
            # 回家：直接回家如果被挡住（手臂在筐上方，回家路径可能擦到筐），
            # 先原地抬高到安全高度，再回家，不要因此终止整个任务。
            try:
                move(arm, hp, hr, key + "_home")
            except StageFailure:
                tools.trial.sync()
                p_now, r_now = scene.end_pose(arm)
                record("home_raise", message="Direct home blocked; raise then retry")
                print("  [工具] home ✗ 被挡 → 先抬高再回家", flush=True)
                move(
                    arm,
                    [p_now[0], p_now[1], float(p_now[2]) + 0.10],
                    r_now,
                    key + "_home_lift",
                )
                move(arm, hp, hr, key + "_home_retry")

        record("verify_all")
        tools.trial.sync()
        for key in keys:
            placed[key] = bool(placed[key] and verify(scene, key, spec.objects[key].dest))
        report["status"] = "complete" if all(placed.values()) else "incomplete"
        record("finished")
        return report["status"] == "complete"
    except Exception as exc:
        result = exc.result if isinstance(exc, StageFailure) else SkillResult(False, classify(exc), str(exc))
        report["status"] = "blocked"
        record(report["stage"], result)
        print(f"停止于 {report['object']}/{report['stage']}: [{result.code}] {result.message}", flush=True)
        return False


def _order_feasible(spec, scene, order):
    """校验处理顺序是否"同一条手臂上先近后远"。

    远处物体先抓时，近处物体还挡在路上，IK 常常只能找到碰撞解，
    表现为 `ik_no_solution ... best position error 0`（位置够了但解被拒）。
    这个规则比模型看图判断更可靠，所以模型的顺序必须过这一关。
    """
    for arm in ("left", "right"):
        on_arm = [k for k in order if spec.objects[k].arm == arm]
        if len(on_arm) < 2:
            continue
        hand, _ = scene.end_pose(arm)
        dist = {}
        for k in on_arm:
            obj, _ = scene.body_pose(k)
            dist[k] = float(((obj[0] - hand[0]) ** 2 + (obj[1] - hand[1]) ** 2) ** 0.5)
        for near, far in zip(on_arm, on_arm[1:], strict=False):
            if dist[near] > dist[far] + 0.02:  # 2cm 容差
                return False
    return True


def _open_scene(args):
    """打开仿真、加载场景模型，返回 (loop, channel, bridge, env, scene, work)。调用方负责关闭。"""
    import asyncio
    import grpc
    from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
    from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

    from examples.euler.medication_sorting.execution import SortingEnv
    from examples.euler.medication_sorting.scene import Scene, prepare_model

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    channel = grpc.aio.insecure_channel(
        args.endpoint, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)]
    )
    bridge = OrcaStudioBridge(GrpcServiceStub(channel))
    source = loop.run_until_complete(asyncio.wait_for(bridge.load_model_xml(), 30))
    work = Path(args.output) / "calibration"
    work.mkdir(parents=True, exist_ok=True)
    scene_xml = prepare_model(source, work / "scene.xml")
    env = SortingEnv(str(scene_xml), endpoint=args.endpoint)
    scene = Scene(str(scene_xml))
    loop.run_until_complete(bridge.pause_simulation())
    return loop, channel, bridge, env, scene, work


def _close_scene(loop, channel, env):

    try:
        env.close()
    finally:
        loop.run_until_complete(channel.close())
        loop.close()


def print_movable_bodies(args) -> bool:
    """列出场景里可移动 body 的名字、位置、包围盒尺寸，方便写 YAML / 选标定参考物体。"""
    import numpy as np

    from examples.euler.medication_sorting.scene import vertices

    loop, channel, bridge, env, scene, _ = _open_scene(args)
    try:
        print(f"场景可移动 body（共 {len(scene.movable_bodies())} 个）：", flush=True)
        for name in scene.movable_bodies():
            try:
                p, _ = scene.body_pose(name)
                v = vertices(scene, name)
            except (ValueError, KeyError):
                print(f"  {name}  （无碰撞几何）", flush=True)
                continue
            ext = (v.max(axis=0) - v.min(axis=0)) * 100
            vol = float(np.prod(ext))
            print(f"  {name}", flush=True)
            print(f"      pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  {ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} cm  体积 {vol:.0f} cm³", flush=True)
    finally:
        _close_scene(loop, channel, env)
    return True


def run_calibration(args, backend) -> bool:
    """为当前场景标定单应矩阵，写到 --homography。"""
    import numpy as np

    from examples.euler.medication_sorting.grounding import calibrate

    if not args.ref_body:
        print("缺少 --ref-body（参考物体的 body 名）；先用 --list-bodies 查看可选名字", flush=True)
        return False
    loop, channel, bridge, env, scene, _ = _open_scene(args)
    try:
        env.set_joint_qvel(np.zeros(env.model.nv))
        env.mj_forward()
        from examples.euler.medication_sorting.scene import vertices

        p, _ = scene.body_pose(args.ref_body)
        v = vertices(scene, args.ref_body)
        ext = (v.max(axis=0) - v.min(axis=0))
        span = float(min(ext[0], ext[1]))
        # 网格半宽按参考物尺寸定：小物体 3cm，大物体到 10cm。
        # 大物体网格太小的话，像素几乎不动，拟合条件很差。
        rng = min(0.10, max(0.03, span * 0.3))
        print(f"  参考物 {args.ref_body} 尺寸 {span*100:.1f} cm → 标定网格 ±{rng*100:.1f} cm", flush=True)
        res = calibrate(
            bridge, scene, env, getattr(args, "ref_arm", "left"), backend,
            object_key=args.ref_body,
            ref_desc=args.ref_desc or args.ref_body,
            x_range=(float(p[0]) - rng, float(p[0]) + rng),
            y_range=(float(p[1]) - rng, float(p[1]) + rng),
            table_z=float(p[2]),
            out_path=str(args.homography),
            loop=loop,
        )
        print(f"\n标定完成：残差 {res['residual'] * 1000:.1f} mm → {args.homography}", flush=True)
    finally:
        _close_scene(loop, channel, env)
    return True


def run_agentic_episode(args, backend) -> bool:
    """Create an isolated run directory and close resources on every exit path."""
    import asyncio
    import time
    from contextlib import ExitStack

    import grpc
    from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
    from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

    from examples.euler.medication_sorting.main import FrameGrabber
    from examples.euler.medication_sorting.execution import Executor, SortingEnv
    from examples.euler.medication_sorting.tools import ArmTools
    from examples.euler.medication_sorting.grounding import load_homography
    from examples.euler.medication_sorting.scene import Scene, prepare_model

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    out = Path(args.output) / run_id
    out.mkdir(parents=True, exist_ok=False)
    print(f"本轮记录：{out}", flush=True)
    with ExitStack() as cleanup:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cleanup.callback(loop.close)
        channel = grpc.aio.insecure_channel(
            args.endpoint, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)]
        )
        cleanup.callback(lambda: loop.run_until_complete(channel.close()))
        bridge = OrcaStudioBridge(GrpcServiceStub(channel))
        source = loop.run_until_complete(asyncio.wait_for(bridge.load_model_xml(), 30))
        if getattr(args, "arm_only", True):
            from examples.euler.medication_sorting import arm_only_model

            prepared = prepare_model(source, out / "prepared.xml")
            ao = arm_only_model.build(prepared)
            scene_xml = str(ao.out_path)
            env = arm_only_model.make_arm_only_env(ao, args.endpoint)
            original_render = bridge.render

            def render_padded(qpos, *a, **kw):
                return original_render(ao.pad(qpos), *a, **kw)

            bridge.render = render_padded
        else:
            scene_xml = prepare_model(source, out / "scene.xml")
            env = SortingEnv(scene_xml, endpoint=args.endpoint)
        cleanup.callback(env.close)
        scene = Scene(scene_xml)
        loop.run_until_complete(bridge.pause_simulation())
        env.set_joint_qvel(np.zeros(env.model.nv))
        env.mj_forward()
        env.step(env.command)
        trial = Executor(env, scene, bridge, loop, out)
        trial.hold(75, "initial_settle")
        trial.align_initial_command()
        trial.home = env.command.copy()
        trial.sync()
        home_poses = {a: scene.end_pose(a) for a in ("left", "right")}
        grabber = FrameGrabber(bridge=bridge, loop=loop) if args.capture_images else None
        if grabber:
            cleanup.callback(grabber.close)
        path = Path(getattr(args, "homography", None) or "layouts/homography.json")
        grounder = load_homography(path) if path.exists() else None
        if grounder is None:
            print(f"  [警告] 找不到单应矩阵 {path}，视觉定位会失败；先跑 --calibrate 生成", flush=True)
        tools = ArmTools(trial, scene, env, bridge, loop, grabber, grounder, backend)
        # 相机暖机：刚开流时前几帧可能没就绪，先空拍几张让三路相机稳定，
        # 避免第一次 observe 取不到帧导致视觉定位直接失败。
        if grabber is not None:
            for _ in range(4):
                tools.observe()
                time.sleep(0.2)
        spec = TaskSpec.load(getattr(args, "task", None) or "tasks/medication_sorting.yaml")
        return execute_tasks(tools, spec, home_poses, out)
