"""药品回合的只读离线验收与使用说明。

使用项目唯一解释器（可从任意目录执行）：
    /home/pc/ZXJ/SY/.venv/bin/python /home/pc/ZXJ/SY/tools/medication_sorting/verify_episodes.py --input /home/pc/ZXJ/SY/data/medication_sorting/raw --expert-root /home/pc/ZXJ/SY/data/medication_sorting/experiments

只向 stdout 输出 JSON，不连接 MCP/gRPC、不创建机器人、不加载 GPU、
不写/修补图像、timestamp、成功标记或数据划分。测试中的人工样例只用于
检验拒绝规则；本工具不会创建示范。退出码 1 表示数据拒绝、无完整回合、
回合正在写入或不足两个不同内容的回合，无法提出独立验证集。

对仍在采集的批次，只读验收已提交最终 metadata 的回合：
    /home/pc/ZXJ/SY/.venv/bin/python /home/pc/ZXJ/SY/tools/medication_sorting/verify_episodes.py --input /absolute/raw_session --completed-only --summary
未提交 metadata 的目录单列 pending，不读取其逐帧产物。此模式的退出码
只表示已完成回合的质量；actual_split_ready 表示至少 21 个独立内容组，
足够保留 20 组训练及至少 1 组验证，尚未实际转换或应用到 loader。
重试拒绝 PNG 单独核对和计数，不参与最终 RGB 帧数或重复统计。

频率严格依据 metadata.fps：5 Hz 对应 0.2 s，20 Hz 对应 0.05 s。
核对 metadata/NPZ/report/trace 帧数和成功证据；观测在动作之前，trace
在动作之后，因此 trace.time=NPZ.sim_time+1/fps，NPZ.state[1:] 应等于
trace.state[:-1]。state/action 均为七关节绝对 rad 加绝对 gripper closure，
0=open、1=closed。按主线程共用的 1.2 rad/s、1.6 closure/s 指令限速
校验每帧目标变化：5 Hz 为 0.24 rad/0.32 closure；旧函数的默认
0.06/0.08 是 20 Hz 单步预算，不用于推断当前 5 Hz 部署配置。

viewport 核对 frames/序号/{base_rgb,wrist_rgb}.png（原始 RGBA）、
先转 RGB 再 LANCZOS resize 和 gamma 后的 NPZ，以及独立 *_224.png。
核对 camera_transforms.json（兼容 cameraTransforms.json）的 frame、
sim_time/simTime、physics_paused 和请求位姿。virtual_hand_follower 是
实际 EE 加世界偏移 [-.035,.16,.28]、固定 Euler [-65,0,180] 的虚拟观察
相机，并非物理腕部传感器；同步前 ee_position（兼容
actual_sourceEE_position）用于核对
跟随偏移，旧回合可对照上一条 trace EE。旧 virtual_right_wrist 和 sensor 文件格式
亦可检查。固定 overhead 为 [.58,-3.62,5.35]、Euler [-90,0,180]。

报告抬升/释放/筐底几何结论须与轨迹一致；离线只能核对已保存证据，
不能重演物理接触或独立证明渲染来源。手端相机若未保存来源 EE 位姿，
工具会明确提示缺失；不会把请求位姿当成独立实测证明。sensor 的相机
位姿检查未落盘也会提示。紫筐验收读取实际 placement.local_bounds 和
floor_height，不假定原始筐尺寸，兼容实验副本放大。

统计两路内容、变化、相邻重复、最长重复段；拒绝整段冻帧和复制视角。
短时静止的重复只警告。划分以完整回合为单位，内容完全相同的副本
归同一组，避免跨训练/验证泄漏；仅输出确定性建议，现有 OpenPI loader
未消费该建议，必须由主线程另行落实独立数据集及训练集统计量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.medication_sorting.convert_episodes import episode_directories, load_episode
from tools.medication_sorting.task_specs import TARGETS, EXPERT_PROFILES


CAMERAS = ("base_rgb", "wrist_rgb")
JOINTS = ["unitree_g1_1_right_" + name + "_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
PHASES = ("settle_initial", "approach", "descend", "close", "lift", "carry", "lower", "release", "retreat", "settle_final")
ALIGNMENTS = {"paused_physics_snapshot_with_camera_pose_check": "sensor",
              "paused_physics_snapshot_with_explicit_camera_transform": "viewport"}
TOLERANCE = 1e-6


def _reject_constant(value):
    raise ValueError(f"JSON 不允许非有限值：{value}")


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        raise ValueError(f"{path.name} 缺失或损坏：{error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须为 JSON 对象")
    return value


def _trace(path):
    try:
        rows = [json.loads(line, parse_constant=_reject_constant) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as error:
        raise ValueError(f"trace.jsonl 缺失或损坏：{error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("trace 必须有非空完整的逐步记录")
    return rows


def _number(value):
    return type(value) in (int, float) and np.isfinite(value)


def _period(times, fps):
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(times < 0)
            or np.any(np.abs(np.diff(times) - 1 / fps) > TOLERANCE)):
        raise ValueError(f"sim_time 必须严格以 metadata fps={fps} 等间隔采样，允许误差 {TOLERANCE} s")


def _report_errors(report):
    errors = []
    for key in ("success", "lift_verified", "released", "no_object_teleport_during_episode"):
        if report.get(key) is not True:
            errors.append(f"report.{key} 必须为 true")
    if report.get("error"):
        errors.append(f"report 执行失败：{report['error']}")
    lift, speed = report.get("lift_height"), report.get("final_linear_speed")
    if not _number(lift) or lift <= 0.06:
        errors.append("report.lift_height 必须真实抬升超过 0.06 m")
    if not _number(speed) or not 0 <= speed < 0.02:
        errors.append("report.final_linear_speed 必须位于 [0,0.02) m/s")
    placement = report.get("placement")
    if not isinstance(placement, dict) or placement.get("inside") is not True:
        errors.append("report.placement.inside 必须证实药盒已进入真实筐底几何")
    else:
        try:
            bounds = np.asarray(placement["local_bounds"], dtype=float)
            half = np.asarray(placement["floor_half_size"], dtype=float)
            height = placement["floor_height"]
            if (bounds.shape != (2, 3) or half.shape != (2,) or not np.isfinite(bounds).all()
                    or not np.isfinite(half).all() or np.any(half <= 0) or not _number(height)
                    or np.any(bounds[0] > bounds[1]) or np.any(bounds[1, :2] - bounds[0, :2] >= 2 * half)
                    or not height - 0.003 <= bounds[0, 2] < height + 0.03):
                raise ValueError("筐底几何数值不一致")
            if 'floor_center' in placement:
                center=np.asarray(placement['floor_center'],dtype=float)
                if (center.shape!=(3,) or not np.isfinite(center).all()
                    or np.any(bounds[0,:2]<=center[:2]-half)
                    or np.any(bounds[1,:2]>=center[:2]+half)):
                    raise ValueError('药盒外形不在已记录的筐底平面范围内')
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"report.placement 几何证据无效：{error}")
    return errors


def _trace_checks(report, rows, fps):
    arrays = {key: np.asarray([row[key] for row in rows], dtype=float)
              for key in ("state", "action", "box_pos")}
    times = np.asarray([row["time"] for row in rows], dtype=float)
    _period(times, fps)
    n = len(rows)
    if type(report.get("frames")) is not int or report["frames"] != n:
        raise ValueError("report.frames 与 trace 帧数不一致")
    for key, width in (("state", 8), ("action", 8), ("box_pos", 3)):
        if arrays[key].shape != (n, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"trace.{key} 必须为有限 ({n},{width})")
    for key in ("state", "action"):
        if np.any((arrays[key][:, 7] < 0) | (arrays[key][:, 7] > 1)):
            raise ValueError(f"trace.{key} closure 超出 [0,1]")
    phases = [row["phase"] for row in rows]
    sequence = [phase for index, phase in enumerate(phases) if index == 0 or phase != phases[index - 1]]
    if sequence != list(PHASES):
        raise ValueError(f"trace 阶段不完整或顺序异常：{sequence}")
    initial = arrays["box_pos"][[i for i, p in enumerate(phases) if p == "settle_initial"][-1], 2]
    lift_rows = [i for i, p in enumerate(phases) if p == "lift"]
    height = arrays["box_pos"][lift_rows[-1], 2] - initial
    if height <= 0.06 or not _number(report.get("lift_height")) or abs(height - report["lift_height"]) > 0.003:
        raise ValueError("trace 药盒抬升与 report.lift_height 不一致")
    contacts = [row["contact_sides"] for row in rows]
    if not all(isinstance(sides, list) and all(side in ("inner", "outer") for side in sides) for sides in contacts):
        raise ValueError("trace 缺少有效的两指接触记录")
    if not any(set(contacts[i]) == {"inner", "outer"} and arrays["box_pos"][i, 2] - initial > 0.06 for i in lift_rows):
        raise ValueError("trace 没有双指真实夹持抬升证据")
    final = [i for i, p in enumerate(phases) if p == "settle_final"]
    tail = final[-fps:]
    if len(tail) < fps or any(contacts[i] for i in tail) or np.any(arrays["action"][tail, 7] > 0.05):
        raise ValueError("trace 最后一秒不足或仍有夹爪接触/闭合目标")
    motion = np.linalg.norm(np.diff(arrays["box_pos"][tail], axis=0), axis=1) * fps
    if len(motion) and np.max(motion) >= 0.02:
        raise ValueError("trace 最后一秒物体仍运动超过 0.02 m/s")
    arrays.update(sim_time=times, phase=np.asarray(phases))
    delta = np.abs(np.diff(arrays["action"][:, :7], axis=0))
    stats = {"trace_frames": n, "lift_height_from_trace_m": float(height),
             "max_joint_command_step_rad": float(delta.max()),
             "command_transitions_above_deployment_rate": int(np.count_nonzero((delta > 1.2 / fps + TOLERANCE).any(axis=1))),
             "closure_transitions_above_deployment_rate": int(np.count_nonzero(np.abs(np.diff(arrays['action'][:, 7])) > 1.6 / fps + TOLERANCE)),
             "deployment_joint_command_speed_rad_s": 1.2,
             "deployment_closure_command_speed_s": 1.6,
             "deployment_joint_step_budget_rad": 1.2 / fps,
             "deployment_closure_step_budget": 1.6 / fps}
    return arrays, stats


def _rgb_stats(images):
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3 or not len(images):
        raise ValueError("RGB 必须是非空 uint8 (T,H,W,3)")
    hashes = [hashlib.sha256(image.tobytes()).digest() for image in images]
    repeats = [first == second for first, second in zip(hashes[:-1], hashes[1:], strict=True)]
    longest = run = 1
    for repeat in repeats:
        run = run + 1 if repeat else 1
        longest = max(longest, run)
    temporal_change = [float(np.abs(first.astype(np.int16) - second.astype(np.int16)).mean())
                       for first, second in zip(images[:-1], images[1:], strict=True)]
    return {"shape": list(images.shape), "dtype": str(images.dtype),
            "std_min": float(min(image.std() for image in images)),
            "nonzero_fraction_min": float(min(np.mean(image != 0) for image in images)),
            "effective_fraction_min": float(min(np.mean(image > 8) for image in images)),
            "spatial_range_min": int(np.ptp(images, axis=(1, 2)).max(axis=1).min()),
            "mean_temporal_pixel_change": float(np.mean(temporal_change)) if temporal_change else 0.0,
            "unique_frames": len(set(hashes)), "changed_transitions": len(repeats) - sum(repeats),
            "adjacent_duplicate_fraction": sum(repeats) / max(1, len(repeats)), "max_duplicate_run_frames": longest}


def _png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _snapshot(path):
    result = []
    for name in ("metadata.json", "episode.npz", "report.json", "trace.jsonl"):
        file = path / name
        try:
            stat = file.stat()
            result.append((name, stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            result.append((name, None, None))
    return result


def _camera_checks(path, metadata, arrays, rows, stats, warnings):
    backend = ALIGNMENTS.get(metadata.get("camera_alignment"))
    if backend is None:
        raise ValueError("camera_alignment 未声明可审计的 sensor/viewport 同步方式")
    stats["camera_backend"] = backend
    n = len(arrays["sim_time"])
    pair_count, transform_count = 0, 0
    ee_checks, direct_ee_checks, retries, rejected_checked = 0, 0, 0, 0
    for i in range(n):
        folder = path / "frames" / f"{i:06d}"
        transforms = None
        if backend == "viewport":
            files = [file for file in (folder / "camera_transforms.json", folder / "cameraTransforms.json") if file.exists()]
            if len(files) != 1:
                raise ValueError(f"帧 {i} 必须有唯一 camera_transforms.json/cameraTransforms.json")
            transforms = _json(files[0])
            sim_time = transforms.get("sim_time", transforms.get("simTime"))
            if (type(transforms.get("frame")) is not int or transforms["frame"] != i
                    or not _number(sim_time) or abs(sim_time - arrays["sim_time"][i]) > TOLERANCE
                    or transforms.get("physics_paused", transforms.get("physicsPaused")) is not True):
                raise ValueError(f"帧 {i} 相机 transform 的 frame/sim_time/physics_paused 与观测不一致")
            poses = transforms.get("poses")
            if not isinstance(poses, dict) or set(poses) != set(CAMERAS):
                raise ValueError(f"帧 {i} 缺少双相机 poses")
            for key in CAMERAS:
                pose = poses[key]
                if not isinstance(pose, dict):
                    raise ValueError(f"帧 {i} {key} pose 无效")
                for field in ("position", "rotation_degrees"):
                    value = np.asarray(pose.get(field), dtype=float)
                    if value.shape != (3,) or not np.isfinite(value).all():
                        raise ValueError(f"帧 {i} {key}.{field} 必须为有限三维值")
            base = poses["base_rgb"]
            if (not np.allclose(base["position"], [0.58, -3.62, 5.35], atol=TOLERANCE, rtol=0)
                    or not np.allclose(base["rotation_degrees"], [-90, 0, 180], atol=TOLERANCE, rtol=0)):
                raise ValueError(f"帧 {i} 固定 overhead 位姿偏离采集/推理共同配置")
            wrist_name = metadata["camera_metadata"]["wrist_rgb"].get("name")
            if wrist_name not in ("virtual_right_wrist", "virtual_hand_follower"):
                raise ValueError("viewport wrist_rgb 必须声明实际使用的虚拟观察相机名称")
            if wrist_name == "virtual_hand_follower":
                hand = poses["wrist_rgb"]
                if not np.allclose(hand["rotation_degrees"], [-65, 0, 180], atol=TOLERANCE, rtol=0):
                    raise ValueError(f"帧 {i} hand follower 世界朝向异常")
                # trace 保存动作后 EE，因此上一条 EE 对应当前暂停观测。
                source = transforms.get("ee_position", transforms.get("actual_sourceEE_position", transforms.get("source_ee_position")))
                direct_source = source is not None
                if source is None and i > 0 and len(rows) >= i:
                    source = rows[i - 1].get("ee")
                if source is not None:
                    ee = np.asarray(source, dtype=float)
                    if (ee.shape != (3,) or not np.isfinite(ee).all()
                            or not np.allclose(hand["position"], ee + [-0.035, 0.16, 0.28], atol=0.003, rtol=0)):
                        raise ValueError(f"帧 {i} hand follower 并非实际 EE 加约定世界偏移")
                    ee_checks += 1
                    direct_ee_checks += int(direct_source)
            attempts = transforms.get("render_attempts")
            if attempts is not None:
                if (not isinstance(attempts, dict) or set(attempts) != set(CAMERAS)
                        or any(type(value) is not int or not 1 <= value <= 3 for value in attempts.values())):
                    raise ValueError(f"帧 {i} render_attempts 必须为双相机各 1–3 次")
                for slot, count in attempts.items():
                    retries += count - 1
                    for attempt in range(1, count):
                        rejected = _png(folder / f"rejected_{slot}_{attempt:02d}.png")
                        rejected = np.asarray(Image.fromarray(rejected).resize((224, 224), Image.Resampling.LANCZOS))
                        rejected = np.rint(255 * (rejected.astype(np.float32) / 255) ** metadata["camera_metadata"][slot]["gamma"]).astype(np.uint8)
                        if rejected.std() >= 2 and np.mean(rejected > 8) >= 0.05:
                            raise ValueError(f"帧 {i} {slot} rejected PNG 与空白重绘记录不一致")
                        rejected_checked += 1
            transform_count += 1
        for key in CAMERAS:
            camera = metadata["camera_metadata"][key]
            gamma = camera.get("gamma")
            if not _number(gamma) or gamma <= 0:
                raise ValueError(f"{key} 缺少有效 gamma")
            image = arrays[key][i]
            if list(image.shape[:2]) != camera.get("resolution"):
                raise ValueError(f"{key} metadata 分辨率与 NPZ 不一致")
            if backend == "viewport":
                raw_path = folder / f"{key}.png"
            else:
                name = camera.get("name")
                if name not in ("head_cam", "camera_right"):
                    raise ValueError(f"{key} 未声明真实 sensor 相机名称")
                candidates = [file for file in (folder / "color").glob(f"{name}_color_*.png")
                              if re.fullmatch(re.escape(name) + r"_color_\d+\.png", file.name)]
                if len(candidates) != 1:
                    raise ValueError(f"帧 {i} {key} 缺少唯一原生 sensor PNG")
                raw_path = candidates[0]
            raw = _png(raw_path)
            if backend == "viewport":
                raw = np.asarray(Image.fromarray(raw).resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS))
            elif raw.shape != image.shape:
                raise ValueError(f"帧 {i} {key} sensor 原生 PNG 分辨率不一致")
            if raw.std() < 1:
                raise ValueError(f"帧 {i} {key} 原始 RGB 近乎空白；alpha 不能充当有效内容")
            exposed = np.rint(255 * (raw.astype(np.float32) / 255) ** gamma).astype(np.uint8)
            if not np.array_equal(exposed, image):
                raise ValueError(f"帧 {i} {key} 原始 PNG resize/gamma 与 NPZ 不一致")
            if backend == "viewport" and not np.array_equal(_png(folder / f"{key}_224.png"), image):
                raise ValueError(f"帧 {i} {key} 独立处理后 PNG 与 NPZ 不一致")
        pair_count += 1
    stats.update(native_png_pairs_checked=pair_count, camera_transforms_checked=transform_count,
                 hand_follower_ee_checks=ee_checks, hand_follower_direct_ee_checks=direct_ee_checks,
                 render_retries=retries, retained_rejected_pngs_checked=rejected_checked)
    if backend == "sensor":
        warnings.append("sensor 的物理相机位姿检查未落盘，离线只能对照原始 PNG")
    elif metadata["camera_metadata"]["wrist_rgb"].get("name") == "virtual_hand_follower" and ee_checks < n:
        warnings.append(f"hand follower 只有 {ee_checks}/{n} 帧有来源 EE 对照；首帧需额外保存来源位姿")


def verify_episode(path):
    """验收已有单个回合；拒绝理由与图像统计同时保留，不改变产物。"""
    path = Path(path).resolve()
    before = _snapshot(path)
    result = {"path": str(path), "accepted": False, "status": "rejected", "errors": [], "warnings": [], "stats": {}}
    errors, warnings, stats = result["errors"], result["warnings"], result["stats"]
    metadata, report, rows, arrays = {}, {}, [], None
    for file, target in (("metadata.json", metadata), ("report.json", report)):
        try:
            target.update(_json(path / file))
        except ValueError as error:
            errors.append(str(error))
    errors.extend(_report_errors(report))
    if 'expert_profile' in metadata or 'expert_profile' in report:
        profile=metadata.get('expert_profile')
        if profile not in EXPERT_PROFILES or report.get('expert_profile') != profile:
            errors.append('实验配置缺失、未知或与执行报告不一致')
    if 'object_name' in metadata:
        target=metadata['object_name']
        spec=TARGETS.get(target) if isinstance(target,str) else None
        if spec is None:
            errors.append('metadata.object_name 不是四物品任务目标')
        elif (report.get('object_name')!=target or report.get('task')!=spec['task']
            or metadata.get('task')!=spec['task']
            or metadata.get('basket_suffix')!=spec['basket_suffix']
            or not str(report.get('box','')).endswith(spec['body_suffix'])
            or not str(report.get('basket','')).endswith(spec['basket_suffix'])):
            errors.append('物品、任务指令或目标筐与真实执行报告不一致')
    fps = metadata.get("fps")
    if type(fps) is not int or fps <= 0:
        errors.append("严格验收要求 metadata.fps 为实际正整数采样频率")
        fps = None
    else:
        stats["fps"] = fps
    for key in ("success", "complete", "synchronized"):
        if metadata.get(key) is not True:
            errors.append(f"metadata.{key} 必须为 true")
    if metadata.get("joint_names") != JOINTS:
        errors.append("metadata 七个右臂关节顺序与控制/推理契约不同")
    try:
        with np.load(path / "episode.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("state", "action", *CAMERAS, "sim_time", "phase")}
        stats["frames"] = len(arrays["state"])
        for key in CAMERAS:
            stats[key] = _rgb_stats(arrays[key])
            if stats[key]["unique_frames"] == 1:
                errors.append(f"{key} 整段冻帧")
            if stats[key]["std_min"] < 1:
                errors.append(f"{key} 含近乎空白的 RGB")
            if stats[key]["spatial_range_min"] == 0:
                errors.append(f"{key} 含无空间内容的单色 RGB")
            if metadata.get("camera_alignment") == "paused_physics_snapshot_with_explicit_camera_transform" and (
                    stats[key]["std_min"] < 2 or stats[key]["effective_fraction_min"] < 0.05):
                errors.append(f"{key} 未达到 viewport 的 std>=2、有效分量>=5% 内容门槛")
            if stats[key]["adjacent_duplicate_fraction"] > 0.25:
                warnings.append(f"{key} 相邻重复比例较高：{stats[key]['adjacent_duplicate_fraction']:.3f}，需复查静止阶段")
        same = np.all(arrays["base_rgb"] == arrays["wrist_rgb"], axis=(1, 2, 3))
        stats["identical_view_pairs"] = int(same.sum())
        if np.any(same):
            errors.append("出现完全相同的双视角帧，疑似复制画面")
        if fps is not None:
            _period(arrays["sim_time"], fps)
            stats["sim_time_interval_s"] = [float(np.diff(arrays["sim_time"]).min()), float(np.diff(arrays["sim_time"]).max())]
            for key in ("state", "action"):
                delta = np.abs(np.diff(arrays[key][:, :7], axis=0))
                stats[f"max_{key}_joint_step_rad"] = float(delta.max())
            if metadata.get("frames") != len(arrays["state"]) or type(metadata.get("frames")) is not int:
                errors.append("metadata.frames 必须与 NPZ 一致")
            # 默认 0.5 rad 跳变门槛不因已经落筐而放宽。
            load_episode(path, fps=fps, terminal_phase="settle_final", tolerance_s=TOLERANCE)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, zipfile.BadZipFile) as error:
        errors.append(f"NPZ/metadata 质量不合格：{error}")
    try:
        rows = _trace(path / "trace.jsonl")
        if fps is not None:
            traced, traced_stats = _trace_checks(report, rows, fps)
            stats.update(traced_stats)
            if traced_stats["command_transitions_above_deployment_rate"]:
                errors.append("实际关节指令超过共用的 1.2 rad/s 限速，不能用于最终训练")
            if traced_stats["closure_transitions_above_deployment_rate"]:
                errors.append("实际夹爪指令超过共用的 1.6 closure/s 限速，不能用于最终训练")
            if arrays is not None:
                n = len(arrays["state"])
                if len(rows) != n:
                    raise ValueError("trace 与 NPZ 帧数不一致，可能仍在写入或采集中断")
                if not np.allclose(traced["sim_time"], arrays["sim_time"] + 1 / fps, atol=TOLERANCE, rtol=0):
                    raise ValueError("trace 必须在动作之后，NPZ 观测必须在动作之前，不能移位伪造同步")
                if not np.array_equal(traced["phase"], arrays["phase"].astype(str)):
                    raise ValueError("NPZ phase 与 trace 不一致")
                if not np.allclose(traced["action"], arrays["action"], atol=TOLERANCE, rtol=0):
                    raise ValueError("NPZ.action 与实际 trace 指令不一致")
                if not np.allclose(traced["state"][:-1], arrays["state"][1:], atol=TOLERANCE, rtol=0):
                    raise ValueError("动作后 trace.state 不等于下一帧观测 state，数据动作语义错位")
    except (ValueError, TypeError, KeyError, IndexError) as error:
        errors.append(f"trace 质量不合格：{error}")
    if arrays is not None:
        try:
            _camera_checks(path, metadata, arrays, rows, stats, warnings)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            errors.append(f"PNG/相机同步证据不合格：{error}")
    if before != _snapshot(path):
        errors.append("校验期间关键文件变化，回合可能仍在写入，须落盘结束后重新验收")
        result["status"] = "in_progress"
    elif not (path / "metadata.json").exists() and report and not report.get("error"):
        result["status"] = "in_progress"
    if not errors:
        placement = report["placement"]
        if "floor_center" not in placement:
            warnings.append("placement 未保存 floor 的局部中心，离线只复查底板高度和尺寸，inside 依赖真实控制报告")
        result.update(accepted=True, status="accepted")
        digest = hashlib.sha256()
        for key in ("state", "action", *CAMERAS, "sim_time", "phase"):
            digest.update(key.encode())
            digest.update(str(arrays[key].dtype).encode())
            digest.update(str(arrays[key].shape).encode())
            digest.update(arrays[key].tobytes())
        result["content_sha256"] = digest.hexdigest()
    return result


def _split(accepted, fraction):
    groups = {}
    for episode in accepted:
        groups.setdefault(episode["content_sha256"], []).append(episode["path"])
    result = {"status": "blocked", "train": [], "validation": [], "distinct_content_groups": len(groups),
              "applied_to_training_loader": False, "unit": "complete_episode_content_group"}
    if len(groups) < 2:
        result["reason"] = "至少两个不同内容的合格完整回合才能建立独立验证集；不能随机拆相邻帧"
        return result
    keys = sorted(groups)
    count = min(len(keys) - 1, max(1, round(len(keys) * fraction)))
    validation = set(keys[:count])
    result.update(status="suggested_only",
                  train=sorted(path for key, paths in groups.items() if key not in validation for path in paths),
                  validation=sorted(path for key, paths in groups.items() if key in validation for path in paths))
    return result


def verify_dataset(source, *, validation_fraction=0.2, completed_only=False):
    """逐回合验收，隔离失败和内容重复；划分建议不写入训练入口。"""
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f"输入目录不存在：{source}")
    if not _number(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须位于 (0,1)")
    candidates = episode_directories(source, include_reports=True)
    if completed_only:
        # 采集中只有 frames 的目录也需列入 pending；不打开这些实时帧。
        candidates = sorted(set(candidates) | {path for path in source.rglob("episode_*") if path.is_dir()})
    accepted, rejected, pending = [], [], []
    reference = None
    for path in candidates:
        if completed_only and not (path / "metadata.json").exists():
            pending.append({"path": str(path.resolve()), "status": "pending", "reason": "尚未提交最终 metadata，不验收实时产物"})
            continue
        episode = verify_episode(path)
        if episode["accepted"]:
            metadata = _json(path / "metadata.json")
            signature = {"fps": metadata["fps"], "joint_names": metadata["joint_names"],
                         "camera_alignment": metadata["camera_alignment"], "camera_metadata": metadata["camera_metadata"]}
            if reference is None:
                reference = signature
            elif signature != reference:
                episode.update(accepted=False, status="rejected")
                episode["errors"].append("跨回合 fps/相机处理/关节顺序不同，应使用独立数据集")
        (accepted if episode["accepted"] else rejected).append(episode)
    split = _split(accepted, validation_fraction)
    quality_ok = bool(accepted) and not rejected
    return {"ok": quality_ok and (completed_only or split["status"] == "suggested_only"),
            "quality_ok": quality_ok, "completed_only": completed_only,
            "source": str(source), "accepted": accepted, "rejected": rejected, "pending": pending, "split": split,
            "read_only": True, "gpu_used": False, "mcp_grpc_used": False,
            "training_loader_split_applied": False,
            "evidence_limit": "只核对已保存的真实产物；不能独立重演接触或证明渲染来源。"}


def summarize_dataset(result):
    """简洁质量报告；划分就绪与质量结论分别陈述，不伪称已完成转换。"""
    rows = []
    for episode in sorted(result["accepted"] + result["rejected"], key=lambda row: row["path"]):
        stats = episode["stats"]
        rows.append({"path": episode["path"], "accepted": episode["accepted"], "status": episode["status"],
                     "frames": stats.get("frames"), "fps": stats.get("fps"),
                     "max_state_joint_step_rad": stats.get("max_state_joint_step_rad"),
                     "max_action_joint_step_rad": stats.get("max_action_joint_step_rad"),
                     "rgb": {key: {field: stats.get(key, {}).get(field) for field in (
                         "unique_frames", "mean_temporal_pixel_change", "adjacent_duplicate_fraction")}
                             for key in CAMERAS},
                     "render_retries": stats.get("render_retries", 0),
                     "retained_rejected_pngs_checked": stats.get("retained_rejected_pngs_checked", 0),
                     "direct_ee_checks": stats.get("hand_follower_direct_ee_checks", 0),
                     "errors": episode["errors"], "warnings": episode["warnings"]})
    summary = {"source": result["source"], "ok": result["ok"], "quality_ok": result["quality_ok"],
               "accepted_count": len(result["accepted"]), "rejected_count": len(result["rejected"]),
               "pending_count": len(result["pending"]), "pending": result["pending"],
               "distinct_content_groups": result["split"]["distinct_content_groups"],
               "actual_split_ready": result["split"]["distinct_content_groups"] >= 21,
               "training_loader_split_applied": False, "episodes": rows,
               "warnings": sorted({warning for row in rows for warning in row["warnings"]}),
               "read_only": True, "gpu_used": False, "mcp_grpc_used": False}
    for field in ("max_state_joint_step_rad", "max_action_joint_step_rad"):
        values = [row[field] for row in rows if row[field] is not None]
        summary[field] = max(values) if values else None
    for field in ("render_retries", "retained_rejected_pngs_checked"):
        summary[field] = sum(row[field] for row in rows)
    if "expert_evidence" in result:
        summary["expert_evidence"] = result["expert_evidence"]
    return summary


def expert_evidence(root):
    """单列专家真实成功证据，不把无双 RGB 的专家实验当训练回合。"""
    records = []
    for path in sorted(Path(root).rglob("report.json")):
        try:
            report = _json(path)
            if report.get("success") is not True:
                continue
            errors = _report_errors(report)
            rows = _trace(path.with_name("trace.jsonl"))
            times = np.asarray([row["time"] for row in rows], dtype=float)
            fps = round(1 / np.median(np.diff(times)))
            _, stats = _trace_checks(report, rows, fps)
            records.append({"path": str(path.resolve()), "report_and_trace_consistent": not errors,
                            "errors": errors, "stats": {"fps": fps, **stats},
                            "training_episode": False, "reason": "专家实验本身不包含同步双 RGB/NPZ 示范"})
        except (ValueError, TypeError, KeyError, IndexError) as error:
            records.append({"path": str(path.resolve()), "report_and_trace_consistent": False, "errors": [str(error)]})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读离线药品回合验收：按 metadata fps 检查物理报告、双 RGB、相机位姿、动作和完整回合划分。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, help="可选：单独核对专家报告和轨迹，不计入训练集")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--completed-only", action="store_true", help="只验收已提交 metadata 的回合，其余目录单列 pending")
    parser.add_argument("--summary", action="store_true", help="输出 accepted/rejected、maxstep、RGB、重试和划分就绪摘要")
    args = parser.parse_args(argv)
    try:
        report = verify_dataset(args.input, validation_fraction=args.validation_fraction, completed_only=args.completed_only)
        if args.expert_root:
            report["expert_evidence"] = expert_evidence(args.expert_root)
        if args.summary:
            report = summarize_dataset(report)
    except (ValueError, OSError) as error:
        report = {"ok": False, "read_only": True, "error": str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return int(not report["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
