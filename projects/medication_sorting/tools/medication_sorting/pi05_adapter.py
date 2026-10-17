"""固定基座 G1 右臂的外置 π0.5 适配器（不修改 OpenPI）。

物理契约：state/action 为 float32 的 8 维向量；前七维是同顺序的
右臂绝对关节位置（rad），末维是 gripper closure（0=open，1=closed）。
在线输入键是 state/base_rgb/wrist_rgb/prompt；相机为同步 uint8 HWC RGB。
LeRobot 解码的 float32 CHW [0,1] 只在图像入口恢复为 uint8 HWC。

训练顺序：repack → G1 inputs → 官方 DeltaActions(7 个关节) → G1 自有
quantile normalization → 官方 model transforms（含 32 维 padding）。
推理顺序：官方 Unnormalize → AbsoluteActions → 裁成 8 维并限制 closure。
所有动作 chunk 都相对当前观测的同一个 state，绝不相对前一个动作。
夹爪不能做 delta，也不能反转语义。关节限位由仿真控制端负责。

本模块普通导入只需要 NumPy。仅显式构建 OpenPI 配置或调用外置入口时
才导入 OpenPI；--check 仅读本地版本与源码，不加载模型或 GPU。
配置的 asset_id 等于自己的 repo_id，必须用本数据计算 norm stats。

CPU 检查：
    .venv/bin/python -m tools.medication_sorting.pi05_adapter check
依赖就绪、仿真释放 GPU 后，由主线程显式执行（本模块不会自动执行）：
    .venv/bin/python -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication norm-stats
    .venv/bin/python -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication train -- --exp-name=g1_trial
    .venv/bin/python -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication serve -- --policy.dir=/absolute/checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import threading
import tomllib
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = PROJECT_ROOT / "software/openpi"
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT_COMMIT = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
CONFIG_NAME = "pi05_g1_medication_sorting_lora"
PHYSICAL_DIM = 8
MODEL_DIM = 32
DEFAULT_TASK = "Put the medicine box into the purple basket."
CAPTURE_CONFIG = {
    "control_hz": 5, "camera_backend": "viewport", "resolution": [224,224], "gamma": .45,
    "base_position": [.58,-3.62,5.35], "base_rotation_degrees": [-90.,0.,180.],
    "hand_position_offset": [-.035,.16,.28], "hand_rotation_degrees": [-65.,0.,180.],
    "camera_names": {"base_rgb": "virtual_overhead", "wrist_rgb": "virtual_hand_follower"},
}
REPACK_STRUCTURE = {
    "state": "observation.state",
    "base_rgb": "observation.images.base",
    "wrist_rgb": "observation.images.right_wrist",
    "actions": "action",
    "prompt": "prompt",
}


async def serialize_arrays_sequentially(values, infos, args, *, transfer, commit):
    """写盘完成后释放当前主机数组，随后才复制下一片参数。"""
    for value, info, arg in zip(values, infos, args, strict=True):
        host = transfer(value, info, arg)
        await commit(host, info, arg)
        del host


def make_streaming_registry(copy_lock=None):
    """仅用于 pinned Orbax 0.11.13：逐数组写盘，保留原生格式和 sharding。"""
    import jax
    from orbax.checkpoint import type_handlers
    from orbax.checkpoint._src.metadata import array_metadata_store
    from orbax.checkpoint._src.serialization import replica_slices
    from orbax.checkpoint._src.futures import future

    copy_lock = copy_lock or threading.Lock()

    class BufferedMetadata:
        def __init__(self):
            self.store = array_metadata_store.Store()
            self.entries = []

        async def write(self, checkpoint_dir, array_metadatas, process_index):
            self.entries.extend(array_metadatas)

        def __getattr__(self, name):
            return getattr(self.store, name)

    class StreamingArrayHandler(type_handlers.ArrayHandler):
        async def serialize(self, values, infos, args=None):
            args = args or [type_handlers.SaveArgs()] * len(values)
            type_handlers.check_input_arguments(values, infos, args)
            self._ext_metadata = {}
            self._array_metadata_store.entries = []

            def transfer(value, info, arg):
                if jax.process_count() != 1:
                    raise ValueError("当前分批保存仅支持单主机训练")
                if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                    self._ext_metadata[info.name] = {
                        "random_key_impl": str(jax.random.key_impl(value)),
                    }
                    value = jax.random.key_data(value)
                return replica_slices.transfer_arrays_to_host(
                    [value], self._replica_id, self._use_replica_parallel,
                    enable_pinned_host_transfer=info.enable_pinned_host_transfer,
                )[0]

            async def commit(host, info, arg):
                await self._background_serialize([host], [info], [arg])

            # params 和 optimizer 两个 item 共用限流锁，避免并行 D2H 峰值叠加。
            await asyncio.to_thread(copy_lock.acquire)
            try:
                await serialize_arrays_sequentially(values, infos, args,
                                                     transfer=transfer, commit=commit)
                await self._array_metadata_store.store.write(
                    infos[0].parent_dir, self._array_metadata_store.entries,
                    jax.process_index(),
                )
                self._array_metadata_store.entries = []
            finally:
                copy_lock.release()
            return [future.NoopFuture()]

    handler = StreamingArrayHandler(array_metadata_store=BufferedMetadata())
    return type_handlers.create_type_handler_registry(*[
        (ty, handler if ty is jax.Array else type_handlers.get_type_handler(ty))
        for ty in type_handlers.supported_types()
    ])


def configure_checkpoint_memory(checkpoints_module) -> None:
    """限制 D2H 峰值与 TensorStore 缓冲；不改全局 Orbax 或 OpenPI 源文件。

    本模型最大参数为 2.416 GB，限额必须大于单参数大小。此参数不会
    分批等待写盘限制初始设备到主机拷贝，仍需实测完整模型保存。
    """
    initialize = checkpoints_module.initialize_checkpoint_dir
    original_ocp = checkpoints_module.ocp
    copy_lock = threading.Lock()

    class LimitedOrbax:
        @staticmethod
        def PyTreeCheckpointHandler(*args, **kwargs):
            return original_ocp.PyTreeCheckpointHandler(
                *args, **kwargs, save_concurrent_gb=3, restore_concurrent_gb=3,
                type_handler_registry=make_streaming_registry(copy_lock),
            )

        def __getattr__(self, name):
            return getattr(original_ocp, name)

    @functools.wraps(initialize)
    def bounded_initialize(*args, **kwargs):
        checkpoints_module.ocp = LimitedOrbax()
        try:
            return initialize(*args, **kwargs)
        finally:
            checkpoints_module.ocp = original_ocp

    checkpoints_module.initialize_checkpoint_dir = bounded_initialize


def configure_policy_provenance(policy_module=None) -> None:
    """在实际模型加载入口记录本地检查点来源；不改上游源码。"""
    if policy_module is None:
        from openpi.policies import policy_config as policy_module
    create_policy = policy_module.create_trained_policy

    @functools.wraps(create_policy)
    def traced_policy(train_config, checkpoint_dir, **kwargs):
        path = Path(checkpoint_dir).resolve()
        if not path.name.isdigit():
            raise ValueError("策略需要已提交的数字步号检查点")
        metadata = json.loads((path / "_CHECKPOINT_METADATA").read_text())
        committed = metadata.get("commit_timestamp_nsecs")
        if type(committed) is not int or committed <= 0:
            raise ValueError("策略检查点尚未完成提交")
        stats = sorted((path / "assets").rglob("norm_stats.json"))
        if len(stats) != 1:
            raise ValueError("策略检查点必须包含唯一的本任务统计量")
        provenance = {
            "path": str(path), "step": int(path.name),
            "commit_timestamp_nsecs": committed,
            "params_manifest_sha256": hashlib.sha256((path / "params/manifest.ocdbt").read_bytes()).hexdigest(),
            "params_metadata_sha256": hashlib.sha256((path / "params/_METADATA").read_bytes()).hexdigest(),
            "norm_stats_sha256": hashlib.sha256(stats[0].read_bytes()).hexdigest(),
        }
        config = dataclasses.replace(train_config, policy_metadata={
            **train_config.policy_metadata, "checkpoint_provenance": provenance,
        })
        return create_policy(config, path, **kwargs)

    policy_module.create_trained_policy = traced_policy


def validate_repo_id(repo_id: str) -> str:
    """限制数据集和 assets 相对路径，防止复用其他机器人的统计目录。"""
    if not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("repo_id 必须是 namespace/dataset，不能是绝对路径或含 .. 的路径")
    return repo_id


def validate_vector(value: Any, *, name: str, chunk: bool = False) -> np.ndarray:
    """物理数据严格检查 dtype、有限值、维度与夹爪语义，返回独立副本。"""
    array = np.asarray(value)
    valid_shape = array.ndim == 2 and array.shape[0] > 0 if chunk else array.ndim == 1
    if array.dtype != np.float32 or not valid_shape or array.shape[-1] != PHYSICAL_DIM:
        raise ValueError(f"{name} 必须是 float32 {'(H,8)' if chunk else '(8,)'}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含 NaN/Inf")
    if np.any((array[..., 7] < 0) | (array[..., 7] > 1)):
        raise ValueError(f"{name} gripper closure 必须在 [0,1]，0=open，1=closed")
    return array.copy()


def parse_rgb(value: Any, *, name: str) -> np.ndarray:
    """区分采集端 HWC uint8 与官方 LeRobot 解码端 CHW float32。"""
    array = np.asarray(value)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} 必须是非空三通道 RGB 图像")
    if array.dtype == np.uint8 and array.shape[-1] == 3:
        return np.ascontiguousarray(array)
    if array.dtype == np.float32 and array.shape[0] == 3:
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} LeRobot float32 图像必须有限且位于 [0,1]")
        return np.ascontiguousarray(np.rint(array.transpose(1, 2, 0) * 255).astype(np.uint8))
    raise ValueError(f"{name} 只接受 uint8 HWC 或 LeRobot float32 CHW RGB")


@dataclasses.dataclass(frozen=True)
class G1MedicationInputs:
    """相同入口同时用于训练和推理；右腕必须进入 right_wrist 槽位。"""

    def __call__(self, data: dict) -> dict:
        state = validate_vector(data["state"], name="state")
        base = parse_rgb(data["base_rgb"], name="base_rgb")
        wrist = parse_rgb(data["wrist_rgb"], name="wrist_rgb")
        if base.shape != wrist.shape:
            raise ValueError("base_rgb/wrist_rgb 必须具有相同 HWC shape")
        result = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": np.zeros_like(base), "right_wrist_0_rgb": wrist},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            result["actions"] = validate_vector(data["actions"], name="actions", chunk=True)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class G1MedicationOutputs:
    """仅在反归一化和绝对位置恢复以后截取八维，不改变关节弧度。"""

    def __call__(self, data: dict) -> dict:
        array = np.asarray(data["actions"])
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[-1] != MODEL_DIM:
            raise ValueError("actions 模型输出必须是 (H,32)")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError("actions 模型输出必须为有限浮点数")
        actions = array[:, :PHYSICAL_DIM].astype(np.float32, copy=True)
        if not np.isfinite(actions).all():
            raise ValueError("actions 超过 float32 可表示范围")
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        return {"actions": actions}


def make_data_transforms(_model_config=None):
    """使用官方成对变换，Group.push 会把 inverse 放到输出链最前面。"""
    from openpi import transforms

    mask = transforms.make_bool_mask(7, -1)
    return transforms.Group(inputs=[G1MedicationInputs()], outputs=[G1MedicationOutputs()]).push(
        inputs=[transforms.DeltaActions(mask)], outputs=[transforms.AbsoluteActions(mask)]
    )


def make_data_config(repo_id: str):
    """返回官方 SimpleDataConfig，prompt 和 chunk 参数与转换器一致。"""
    validate_repo_id(repo_id)
    from openpi import transforms
    from openpi.training import config

    return config.SimpleDataConfig(
        repo_id=repo_id,
        assets=config.AssetsConfig(asset_id=repo_id),
        base_config=config.DataConfig(
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(REPACK_STRUCTURE)]),
            action_sequence_keys=("action",),
            prompt_from_task=True,
        ),
        data_transforms=make_data_transforms,
        model_transforms=config.ModelTransformFactory(default_prompt=DEFAULT_TASK),
    )


def create_train_config(repo_id: str, *, action_horizon: int = 10, batch_size: int = 1):
    """JAX π0.5 LoRA 配置；基础权重用 pi05_base，统计量只用 G1 数据。"""
    validate_repo_id(repo_id)
    if type(action_horizon) is not int or action_horizon <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("action_horizon/batch_size 必须是正整数")
    from openpi.models import pi0_config
    from openpi.training import config, weight_loaders
    model = pi0_config.Pi0Config(
        pi05=True, action_dim=MODEL_DIM, action_horizon=action_horizon,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    )
    return config.TrainConfig(
        name=CONFIG_NAME, project_name="g1_medication_sorting", model=model,
        data=make_data_config(repo_id),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=model.get_freeze_filter(), ema_decay=None,
        batch_size=batch_size, num_workers=0, wandb_enabled=False,
        assets_base_dir=str(PROJECT_ROOT / "data/medication_sorting/openpi_assets"),
        checkpoint_base_dir=str(PROJECT_ROOT / "data/medication_sorting/checkpoints"),
        policy_metadata={
            "robot_type": "unitree_g1_fixed_base_right_arm", "state_dim": 8, "action_dim": 8,
            "joint_units": "rad", "joint_action_semantics": "absolute_position",
            "gripper_semantics": {"open": 0.0, "closed": 1.0},
            "observation_config": CAPTURE_CONFIG,
        },
    )


def register_config(train_config) -> None:
    """仅修改当前进程的官方配置字典；不会写入 software/openpi。"""
    from openpi.training import config

    if train_config.name in config._CONFIGS_DICT:
        raise ValueError(f"配置已注册：{train_config.name}")
    config._CONFIGS_DICT[train_config.name] = train_config


def check_openpi_checkout(root: Path = OPENPI_ROOT) -> None:
    root = Path(root)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if commit != OPENPI_COMMIT:
        raise RuntimeError(f"OpenPI commit 必须是 {OPENPI_COMMIT}，实际为 {commit}")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise RuntimeError("OpenPI 上游存在已跟踪文件改动，不能视为 pinned 接口")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    if project["tool"]["uv"]["sources"]["lerobot"]["rev"] != LEROBOT_COMMIT:
        raise RuntimeError(f"LeRobot commit 必须是 {LEROBOT_COMMIT}")


def dependency_report(root: Path = OPENPI_ROOT) -> dict:
    """版本冲突与缺失清单；只使用 distribution metadata，不导入模型。"""
    from packaging.requirements import Requirement

    check_openpi_checkout(root)
    project = tomllib.loads((Path(root) / "pyproject.toml").read_text())["project"]
    missing, conflicts = [], []
    for raw in project["dependencies"]:
        requirement = Requirement(raw)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(raw)
            continue
        if installed not in requirement.specifier:
            conflicts.append({"package": requirement.name, "installed": installed, "required": str(requirement.specifier)})
    # 额外核查 LeRobot 的明确固定要求，以及 cv2 升级引入的 NumPy 下界。
    for name, spec in (("gymnasium", "==0.29.1"), ("opencv-python", None)):
        try:
            package = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if spec is not None and package.version not in Requirement(f"{name}{spec}").specifier:
            conflicts.append({"package": name, "installed": package.version, "required_by_lerobot": spec})
        if name == "opencv-python":
            for raw in package.requires or []:
                requirement = Requirement(raw)
                if requirement.name == "numpy" and (requirement.marker is None or requirement.marker.evaluate()):
                    if "1.26.4" not in requirement.specifier:
                        conflicts.append({"package": name, "installed": package.version, "numpy_requirement": str(requirement.specifier), "openpi_numpy_requirement": "<2.0.0"})
    return {
        "openpi_commit": OPENPI_COMMIT, "lerobot_commit": LEROBOT_COMMIT,
        "python": sys.version.split()[0], "environment": str(PROJECT_ROOT / ".venv"),
        "conflicts": conflicts, "missing": missing,
        "verification_scope": "local_source_and_distribution_metadata_only",
        "gpu_touched": False, "dependencies_installed": False,
    }


def run_openpi_entrypoint(mode: str, *, repo_id: str, action_horizon: int = 10, batch_size: int = 1,
                          arguments: list[str] | None = None) -> None:
    """主线程未来可显式使用的入口；注册后运行官方脚本原始 CLI。"""
    if mode not in {"norm-stats", "train", "serve"}:
        raise ValueError(f"不支持的入口：{mode}")
    check_openpi_checkout()
    for directory in (OPENPI_ROOT / "src", OPENPI_ROOT / "packages/openpi-client/src"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    register_config(create_train_config(repo_id, action_horizon=action_horizon, batch_size=batch_size))
    if mode == "train":
        from openpi.training import checkpoints
        configure_checkpoint_memory(checkpoints)
    elif mode == "serve":
        configure_policy_provenance()
    scripts = {
        "norm-stats": ("compute_norm_stats.py", ["--config-name", CONFIG_NAME]),
        "train": ("train.py", [CONFIG_NAME]),
        "serve": ("serve_policy.py", ["policy:checkpoint", "--policy.config", CONFIG_NAME]),
    }
    script_name, prefix = scripts[mode]
    if mode == "serve":
        # Tyro 的根 Args 参数必须在 policy:checkpoint 子命令之前。
        # 保留 policy 参数及其他未知参数，由官方 CLI 给出真实诊断。
        root_arguments, policy_arguments = [], []
        values = list(arguments or [])
        index = 0
        while index < len(values):
            value = values[index]
            option = value.partition("=")[0]
            if option in {"--env", "--port", "--default-prompt", "--record", "--no-record", "--help", "-h"}:
                root_arguments.append(value)
                if "=" not in value and option in {"--env", "--port", "--default-prompt"} and index + 1 < len(values):
                    index += 1
                    root_arguments.append(values[index])
            else:
                policy_arguments.append(value)
            index += 1
        prefix = [*root_arguments, *prefix]
        arguments = policy_arguments
    script = OPENPI_ROOT / "scripts" / script_name
    previous = sys.argv
    try:
        sys.argv = [str(script), *prefix, *(arguments or [])]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固定基座 G1 右臂 π0.5 外置配置；check 不加载 GPU。", epilog="官方脚本参数放在 -- 后；serve 必须显式提供 --policy.dir。")
    parser.add_argument("--repo-id", default="local/g1_medication", help="与转换器一致的 LeRobot repo_id")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("mode", choices=("check", "norm-stats", "train", "serve"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.mode == "check":
        report = dependency_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["conflicts"] or report["missing"]))
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    run_openpi_entrypoint(args.mode, repo_id=args.repo_id, action_horizon=args.action_horizon,
                          batch_size=args.batch_size, arguments=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
