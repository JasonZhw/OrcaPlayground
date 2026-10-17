"""将成功且结构完整的 G1 episode.npz 转为锁定版本的 LeRobot 数据集。

原始回合目录必须包含 episode.npz 和最终 metadata.json；失败或不完整
回合仅列入 rejected，不进入训练。采集器必须在任务成功、NPZ 写完之后
提交 success:true 的 metadata；离线转换不能证明药盒确实已进入紫色筐。

NPZ：state/action (T,8) float32，base_rgb/wrist_rgb (T,H,W,3) uint8 RGB，
sim_time (T,) 浮点数，phase (T,) Unicode/bytes 文本；禁用 pickle。
metadata：success bool、task 英文、joint_names 七个不重复名称、
gripper_semantics {"open":0,"closed":1}、camera_metadata（或 cameras）
含 base_rgb/wrist_rgb，各有 color_space="RGB"、layout="HWC"。
可选 complete 必须为 true，可选 synchronized 必须为 true。
可选 fps/frames 必须与调用参数及 NPZ 一致；存在 report.json 时必须
同时证实成功、真实抬升、已释放、落筐和低末速度。无 report 的旧格式
仅作结构兼容；严格物理证据及 PNG/trace 验收请使用 verify_episodes。
提供 fps 的实际采集回合还必须遵守部署共用的 1.2 rad/s 关节指令和
1.6 closure/s 夹爪指令限速；默认 0.5 rad 跳变门槛不能代替限速检查。
--terminal-phase 可额外要求末阶段标签；不预设主线程的阶段命名。

只按同一行配对 state、指令、两路图像；不移位、不插值、不做 delta。
sim_time 必须以 1/fps 均匀递增；不以伪造的均匀 timestamp 掩盖漏帧。
关节单步跳变默认上限 0.5 rad，可按采集器控制频率调整。
LeRobot image feature 声明 CHW，add_frame 仍传 HWC；保留 sim_time/phase。
采用 image 存储，避免视频编码对颜色、时刻和像素的额外影响。

先检查（仅 NumPy，无 LeRobot/GPU/输出文件）：
    .venv/bin/python -m tools.medication_sorting.convert_episodes --input /absolute/raw --fps 20 --validate-only
依赖就绪后转换：
    .venv/bin/python -m tools.medication_sorting.convert_episodes --input /absolute/raw --output /absolute/cache/local/g1_medication --repo-id local/g1_medication --fps 20
运行 OpenPI 前须将 HF_LEROBOT_HOME 设为 /absolute/cache。锁定版 loader
按 HF_LEROBOT_HOME/repo_id 读取，不接受任意 root 配置参数。
不覆盖已有目录，不上传 Hub，不安装依赖，也不启动训练。

官方 API 依据：
https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py
同 commit 的 utils.py::validate_feature_image_or_video / hf_transform_to_torch。
该版本 save_episode 即落盘，没有 consolidate/finalize API。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np

from tools.medication_sorting.pi05_adapter import LEROBOT_COMMIT, OPENPI_COMMIT, validate_repo_id


@dataclasses.dataclass(frozen=True)
class Episode:
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict

    @property
    def length(self) -> int:
        return len(self.arrays["state"])


def _metadata(path: Path) -> dict:
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"metadata.json 缺失或损坏：{error}") from error
    if not isinstance(metadata, dict) or metadata.get("success") is not True:
        raise ValueError("只接受最终 success:true 的成功回合")
    if "complete" in metadata and metadata["complete"] is not True:
        raise ValueError("complete 必须为 true，不能训练中断回合")
    task = metadata.get("task")
    if not isinstance(task, str) or not task.strip() or not task.isascii() or not any(char.isalpha() for char in task):
        raise ValueError("task 必须是非空英文 prompt")
    names = metadata.get("joint_names")
    if not isinstance(names, list) or len(names) != 7 or not all(isinstance(name, str) and name.strip() for name in names) or len(set(names)) != 7:
        raise ValueError("joint_names 必须含七个不重复的右臂关节名，并与数据列顺序一致")
    semantics = metadata.get("gripper_semantics")
    if (not isinstance(semantics, dict)
            or type(semantics.get("open")) not in (int, float)
            or type(semantics.get("closed")) not in (int, float)
            or semantics["open"] != 0 or semantics["closed"] != 1):
        raise ValueError("gripper_semantics 必须是 0=open、1=closed 的归一化 closure")
    cameras = metadata.get("camera_metadata", metadata.get("cameras"))
    if not isinstance(cameras, dict):
        raise ValueError("缺少 camera_metadata/cameras")
    if cameras.get("synchronized", True) is not True or metadata.get("synchronized", True) is not True:
        raise ValueError("双相机必须 synchronized；同一 sim_time 对应同一行观测")
    for key in ("base_rgb", "wrist_rgb"):
        camera = cameras.get(key)
        if not isinstance(camera, dict) or camera.get("color_space") != "RGB" or camera.get("layout") != "HWC":
            raise ValueError(f"{key} camera metadata 必须标明 RGB/HWC")
        if camera.get("dtype", "uint8") != "uint8" or camera.get("synchronized", True) is not True:
            raise ValueError(f"{key} 必须是同步 uint8 RGB")
    return metadata


def load_episode(path: Path, *, fps: int, expected_joint_names: Sequence[str] | None = None,
                 terminal_phase: str | None = None, max_joint_step: float = 0.5,
                 tolerance_s: float = 1e-4) -> Episode:
    """验证完整内容，不连接仿真；异常回合由调用者隔离。"""
    if type(fps) is not int or fps <= 0:
        raise ValueError("fps 必须是正整数")
    if not np.isfinite(max_joint_step) or max_joint_step <= 0 or not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("max_joint_step 必须为有限正数，tolerance_s 必须为有限非负数")
    path = Path(path)
    metadata = _metadata(path)
    if "fps" in metadata and (type(metadata["fps"]) is not int or metadata["fps"] != fps):
        raise ValueError("metadata fps 必须与请求 fps 及 sim_time 的实际采样频率一致")
    report_path = path / "report.json"
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"report.json 损坏，不能确认成功：{error}") from error
        if not isinstance(report, dict) or any(report.get(key) is not True for key in ("success", "lift_verified", "released")):
            raise ValueError("report 必须确认 success、lift_verified 和 released，不能用 metadata 掩盖失败")
        placement = report.get("placement")
        if not isinstance(placement, dict) or placement.get("inside") is not True or report.get("error"):
            raise ValueError("report 必须确认真实筐底几何 placement.inside，且无执行错误")
        lift, speed = report.get("lift_height"), report.get("final_linear_speed")
        if (type(lift) not in (int, float) or not np.isfinite(lift) or lift <= 0.06
                or type(speed) not in (int, float) or not np.isfinite(speed) or not 0 <= speed < 0.02):
            raise ValueError("report 必须证实抬升超过 0.06 m 且末速度小于 0.02 m/s")
    if expected_joint_names is not None and tuple(metadata["joint_names"]) != tuple(expected_joint_names):
        raise ValueError("joint_names 顺序与数据集首个成功回合不一致")
    keys = {"state", "action", "base_rgb", "wrist_rgb", "sim_time", "phase"}
    try:
        archive = np.load(path / "episode.npz", allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("episode.npz 必须是 NPZ archive，不能是改名的 NPY")
        with archive as source:
            if not keys <= set(source.files):
                raise ValueError(f"episode.npz 缺少字段：{sorted(keys - set(source.files))}")
            arrays = {key: source[key] for key in keys}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"episode.npz 缺失、损坏或含不安全 object 数据：{error}") from error
    state = arrays["state"]
    if state.ndim != 2 or state.shape[0] < 2:
        raise ValueError("state 必须是至少两帧的 (T,8)")
    length = state.shape[0]
    if "frames" in metadata and (type(metadata["frames"]) is not int or metadata["frames"] != length):
        raise ValueError("metadata frames 必须与 NPZ 帧数一致")
    if report is not None and "frames" in report and (type(report["frames"]) is not int or report["frames"] != length):
        raise ValueError("report frames 必须与 NPZ 帧数一致")
    for key in ("state", "action"):
        values = arrays[key]
        if values.dtype != np.float32 or values.shape != (length, 8) or not np.isfinite(values).all():
            raise ValueError(f"{key} 必须是有限 float32 (T,8)，且各字段 T 一致")
        if np.any((values[:, 7] < 0) | (values[:, 7] > 1)):
            raise ValueError(f"{key} gripper closure 必须位于 [0,1]")
        if np.any(np.abs(np.diff(values[:, :7], axis=0)) > max_joint_step):
            raise ValueError(f"{key} 七关节存在超过 {max_joint_step} rad 的单步跳变")
    if "fps" in metadata:
        if np.any(np.abs(np.diff(arrays["action"][:, :7], axis=0)) > 1.2 / fps + 1e-6):
            raise ValueError("action 关节指令超过部署共用的 1.2 rad/s 限速")
        if np.any(np.abs(np.diff(arrays["action"][:, 7])) > 1.6 / fps + 1e-6):
            raise ValueError("action 夹爪指令超过部署共用的 1.6 closure/s 限速")
    for key in ("base_rgb", "wrist_rgb"):
        images = arrays[key]
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != length or images.shape[-1] != 3 or min(images.shape[1:3]) <= 0:
            raise ValueError(f"{key} 必须是同步 uint8 (T,H,W,3) RGB")
        # 逐通道检查空间内容；纯红等单色帧的跨通道 std 很高，仍是空图。
        if np.any(np.ptp(images, axis=(1, 2)).max(axis=1) == 0):
            raise ValueError(f"{key} 含空白/单色帧，须核查相机采集")
        # 旧的无 fps/report 格式仅维持结构兼容；真实采集元数据会提供 fps。
        if ("fps" in metadata or report is not None) and not np.any(images[1:] != images[:-1]):
            raise ValueError(f"{key} 整段冻帧；非空白的重复画面也不是有效相机采集")
    if arrays["base_rgb"].shape != arrays["wrist_rgb"].shape:
        raise ValueError("base_rgb/wrist_rgb 必须具有相同 shape")
    if np.array_equal(arrays["base_rgb"], arrays["wrist_rgb"]):
        raise ValueError("双相机数据完全相同，疑似把同一视角复制为 wrist_rgb")
    times = arrays["sim_time"]
    if times.shape != (length,) or not np.issubdtype(times.dtype, np.floating) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("sim_time 必须是有限非负浮点 (T,)")
    differences = np.diff(times.astype(np.float64))
    if np.any(differences <= 0) or np.any(np.abs(differences - 1.0 / fps) > tolerance_s):
        raise ValueError("sim_time 必须严格递增且以 1/fps 均匀采样；不能隐藏漏帧或异步观测")
    phase = arrays["phase"]
    if phase.shape != (length,) or phase.dtype.kind not in "US":
        raise ValueError("phase 必须是文本 (T,)，不能是 object/pickle")
    try:
        phase_text = phase.astype(str)
    except UnicodeError as error:
        raise ValueError("phase 文本编码无效") from error
    if any(not value.strip() for value in phase_text):
        raise ValueError("phase 不能包含空标签")
    arrays["phase"] = phase_text
    if terminal_phase is not None and phase_text[-1] != terminal_phase:
        raise ValueError(f"phase 末阶段必须为 {terminal_phase}")
    return Episode(path=path, arrays=arrays, metadata=metadata)


def make_features(episode: Episode) -> dict:
    """锁定 LeRobot 的 image feature shape 是 CHW；实际帧输入仍是 HWC。"""
    _, height, width, _ = episode.arrays["base_rgb"].shape
    names = [*episode.metadata["joint_names"], "gripper_closure"]
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": names.copy()},
        "action": {"dtype": "float32", "shape": (8,), "names": names.copy()},
        "observation.images.base": {"dtype": "image", "shape": (3, height, width), "names": ["channel", "height", "width"]},
        "observation.images.right_wrist": {"dtype": "image", "shape": (3, height, width), "names": ["channel", "height", "width"]},
        "sim_time": {"dtype": "float64", "shape": (1,), "names": ["seconds"]},
        "phase": {"dtype": "string", "shape": (1,), "names": ["phase"]},
    }


def iter_frames(episode: Episode) -> Iterator[dict]:
    for index in range(episode.length):
        yield {
            "observation.state": episode.arrays["state"][index].copy(),
            "action": episode.arrays["action"][index].copy(),
            "observation.images.base": episode.arrays["base_rgb"][index],
            "observation.images.right_wrist": episode.arrays["wrist_rgb"][index],
            "sim_time": np.array([episode.arrays["sim_time"][index]], dtype=np.float64),
            "phase": str(episode.arrays["phase"][index]),
            "task": episode.metadata["task"],
        }


def require_lerobot():
    """版本号 0.1.0 不足以核定接口；必须核对安装的 Git commit。"""
    try:
        package = importlib.metadata.distribution("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"未安装 LeRobot；需要官方 commit {LEROBOT_COMMIT}，本工具不会安装依赖") from error
    try:
        direct_url = json.loads(package.read_text("direct_url.json") or "{}")
    except ValueError as error:
        raise RuntimeError("LeRobot direct_url.json 损坏，无法核对 commit") from error
    if direct_url.get("vcs_info", {}).get("commit_id") != LEROBOT_COMMIT:
        raise RuntimeError(f"LeRobot 安装的 commit 必须是 {LEROBOT_COMMIT}，相同版本号不能代替 revision 核验")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def episode_directories(source: Path, *, include_reports: bool = False) -> list[Path]:
    """普通目录递归扫描，并显式检查划分目录的直接子回合符号链接。

    不递归遍历任意目录链接，避免扫描源数据以外的目录或链接环。
    失效的 episode_* 子链接仍列入候选，由质量检查明确拒绝。
    """
    source = Path(source)
    names = ("metadata.json", "episode.npz", "report.json") if include_reports else ("metadata.json", "episode.npz")
    directories = {file.parent for name in names for file in source.rglob(name)}
    for child in source.iterdir():
        if child.is_symlink() and child.name.startswith("episode_"):
            directories.add(child)
        elif child.is_dir() and any((child / name).exists() for name in names):
            directories.add(child)
    return sorted(directories)


def scan_episodes(source: Path, *, fps: int, terminal_phase: str | None = None,
                  max_joint_step: float = 0.5) -> tuple[list[dict], list[dict]]:
    """预检全部回合，跨回合核对列顺序和分辨率；不把图像长期留在内存。"""
    source = Path(source)
    if type(fps) is not int or fps <= 0:
        raise ValueError("fps 必须是正整数")
    if not source.is_dir():
        raise ValueError(f"输入目录不存在：{source}")
    directories = episode_directories(source)
    accepted, rejected = [], []
    joint_names, shape = None, None
    for directory in directories:
        try:
            episode = load_episode(directory, fps=fps, expected_joint_names=joint_names,
                                   terminal_phase=terminal_phase, max_joint_step=max_joint_step)
            current_shape = tuple(episode.arrays["base_rgb"].shape[1:])
            if shape is not None and current_shape != shape:
                raise ValueError("相机分辨率与首个成功回合不同")
            if joint_names is None:
                joint_names = tuple(episode.metadata["joint_names"])
                shape = current_shape
            accepted.append({"path": str(directory), "length": episode.length, "metadata": episode.metadata})
            del episode
        except ValueError as error:
            rejected.append({"path": str(directory), "reason": str(error)})
    return accepted, rejected


def convert_episodes(source: Path, output: Path, *, repo_id: str, fps: int,
                     terminal_phase: str | None = None, max_joint_step: float = 0.5,
                     dataset_class: Any = None) -> dict:
    """dataset_class 仅供无依赖的接口测试；正常调用严格核对官方 revision。"""
    validate_repo_id(repo_id)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output}")
    accepted, rejected = scan_episodes(source, fps=fps, terminal_phase=terminal_phase, max_joint_step=max_joint_step)
    if not accepted:
        raise ValueError(f"没有可转换的成功完整回合；拒绝明细：{json.dumps(rejected, ensure_ascii=False)}")
    writer_class = require_lerobot() if dataset_class is None else dataset_class
    first = load_episode(Path(accepted[0]["path"]), fps=fps, terminal_phase=terminal_phase, max_joint_step=max_joint_step)
    features = make_features(first)
    joint_names = tuple(first.metadata["joint_names"])
    del first
    dataset = writer_class.create(
        repo_id=repo_id, root=output, fps=fps, robot_type="unitree_g1_fixed_base_right_arm",
        features=features, use_videos=False, image_writer_processes=0, image_writer_threads=0,
    )
    frames = 0
    try:
        for item in accepted:
            episode = load_episode(Path(item["path"]), fps=fps, expected_joint_names=joint_names,
                                   terminal_phase=terminal_phase, max_joint_step=max_joint_step)
            if make_features(episode) != features:
                raise ValueError("预检之后相机 shape/关节字段发生变化，停止转换")
            for frame in iter_frames(episode):
                dataset.add_frame(frame)
            dataset.save_episode()
            frames += episode.length
            del episode
    finally:
        dataset.stop_image_writer()
    report = {
        "repo_id": repo_id, "output": str(output.resolve()), "fps": fps,
        "converted_episodes": len(accepted), "converted_frames": frames,
        "joint_names": list(joint_names), "gripper_semantics": {"open": 0.0, "closed": 1.0},
        "joint_action_semantics": "absolute_position_rad", "model_joint_transform": "delta_from_current_state_seven_joints",
        "openpi_commit": OPENPI_COMMIT, "lerobot_commit": LEROBOT_COMMIT,
        "accepted": accepted, "rejected": rejected,
        "quality_checks": {"max_joint_step_rad": max_joint_step, "timestamp_tolerance_s": 1e-4, "terminal_phase": terminal_phase},
    }
    (output / "g1_conversion_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G1 成功完整回合质量检查与官方 LeRobot 转换；不安装依赖、不使用 GPU。")
    parser.add_argument("--input", type=Path, required=True, help="原始回合根目录，或单个回合目录")
    parser.add_argument("--output", type=Path, help="新建的 LeRobot 数据集目录；不会覆盖已有文件")
    parser.add_argument("--repo-id", default="local/g1_medication")
    parser.add_argument("--fps", type=int, required=True, help="必须等于 sim_time 的实际均匀采样频率")
    parser.add_argument("--max-joint-step", type=float, default=0.5, help="关节位置单步最大变化，单位 rad")
    parser.add_argument("--terminal-phase", help="可选：要求成功回合的最后 phase 等于此标签")
    parser.add_argument("--validate-only", action="store_true", help="只输出质量报告；无需 LeRobot，不写文件")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            accepted, rejected = scan_episodes(args.input, fps=args.fps, terminal_phase=args.terminal_phase, max_joint_step=args.max_joint_step)
            print(json.dumps({"accepted": accepted, "rejected": rejected, "data_written": False}, ensure_ascii=False, indent=2))
            return int(not accepted or bool(rejected))
        if args.output is None:
            parser.error("转换必须指定 --output；仅检查请使用 --validate-only")
        report = convert_episodes(args.input, args.output, repo_id=args.repo_id, fps=args.fps,
                                  terminal_phase=args.terminal_phase, max_joint_step=args.max_joint_step)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
