"""OpenPI π0.5 训练前置核查。

该工具只读取本地环境、OpenPI 源码、LeRobot 元数据和 checkpoint 目录。
它不会安装依赖、下载权重、连接仿真、初始化 GPU 或启动训练。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any


REPOSITORY_URL = "https://github.com/Physical-Intelligence/openpi"
PINNED_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
DEFAULT_PI05_ACTION_DIM = 32
REQUIRED_PACKAGES = (
    "openpi",
    "openpi-client",
    "lerobot",
    "jax",
    "flax",
    "orbax-checkpoint",
    "torch",
    "transformers",
    "numpy",
)


def _block(blockers: list[dict[str, str]], check: str, code: str, message: str) -> None:
    blockers.append({"check": check, "code": code, "message": message})


def _status(blockers: list[dict[str, str]], check: str) -> str:
    return "blocked" if any(item["check"] == check for item in blockers) else "pass"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(value, allow_nan=False)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "JSON 根节点不是对象"
    return value, None


def _parse_requirement(requirement: str) -> tuple[str, str]:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*(.*)", requirement)
    if match is None:
        return requirement, ""
    return match.group(1).lower().replace("_", "-"), match.group(2).strip()


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.match(r"\d+(?:\.\d+)*", value)
    return tuple(int(item) for item in numbers.group(0).split(".")) if numbers else ()


def _version_satisfies(actual: str, requirement: str) -> bool:
    if not requirement:
        return True
    actual_version = _version_tuple(actual)
    for part in requirement.split(","):
        match = re.match(r"(==|>=|<=|>|<|~=)\s*([0-9][^, ]*)", part.strip())
        if match is None:
            continue
        operator, expected = match.groups()
        expected_version = _version_tuple(expected)
        if operator == "==" and actual_version != expected_version:
            return False
        if operator == ">=" and actual_version < expected_version:
            return False
        if operator == "<=" and actual_version > expected_version:
            return False
        if operator == ">" and actual_version <= expected_version:
            return False
        if operator == "<" and actual_version >= expected_version:
            return False
        if operator == "~=" and (actual_version < expected_version or actual_version[:2] != expected_version[:2]):
            return False
    return True


def _load_expected_dependencies(repo_dir: Path) -> tuple[dict[str, str], str | None]:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.is_file():
        return {}, None
    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}, None
    dependencies = project.get("project", {}).get("dependencies", [])
    expected = dict(_parse_requirement(item) for item in dependencies)
    python_requirement = project.get("project", {}).get("requires-python")
    return expected, python_requirement


def _probe_environment(venv_dir: Path, repo_dir: Path, blockers: list[dict[str, str]]) -> dict[str, Any]:
    check_name = "environment"
    python_path = venv_dir / "bin/python"
    result: dict[str, Any] = {
        "venv": str(venv_dir),
        "python": str(python_path),
        "python_version": None,
        "packages": {},
        "expected_dependencies": {},
        "status": "pass",
    }
    if not python_path.is_file() or not python_path.exists():
        _block(blockers, check_name, "environment_python_missing", f"固定项目环境不存在：{python_path}")
        result["status"] = "blocked"
        return result

    probe = (
        "import importlib.metadata as m, json, sys; "
        "names = %r; versions = {}; "
        "\nfor name in names:\n"
        "    try: versions[name] = m.version(name)\n"
        "    except m.PackageNotFoundError: versions[name] = None\n"
        "print(json.dumps({'python': sys.version.split()[0], 'packages': versions}))"
    ) % (list(REQUIRED_PACKAGES),)
    try:
        completed = subprocess.run(
            [str(python_path), "-I", "-B", "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        payload = {}
    if not payload:
        _block(blockers, check_name, "environment_python_probe_failed", f"无法读取 {python_path} 的 Python/包元数据")
        result["status"] = "blocked"
        return result

    result["python_version"] = payload.get("python")
    result["packages"] = payload.get("packages", {})
    expected, python_requirement = _load_expected_dependencies(repo_dir)
    result["expected_dependencies"] = {name: expected.get(name) for name in REQUIRED_PACKAGES}
    if python_requirement and not _version_satisfies(result["python_version"], python_requirement):
        _block(
            blockers,
            check_name,
            "python_version_conflict",
            f"Python {result['python_version']} 不满足 OpenPI 要求 {python_requirement}",
        )

    for package in REQUIRED_PACKAGES:
        actual = result["packages"].get(package)
        requirement = expected.get(package, "")
        if actual is None:
            _block(blockers, check_name, "dependency_missing", f"项目环境缺少依赖 {package!r}")
        elif requirement and not _version_satisfies(actual, requirement):
            _block(
                blockers,
                check_name,
                "dependency_conflict",
                f"{package} 已安装 {actual}，OpenPI 声明要求 {requirement}",
            )
    result["status"] = _status(blockers, check_name)
    return result


def _repository_facts(repo_dir: Path, blockers: list[dict[str, str]]) -> dict[str, Any]:
    check_name = "repository"
    result: dict[str, Any] = {
        "path": str(repo_dir),
        "url": REPOSITORY_URL,
        "commit": None,
        "expected_commit": PINNED_COMMIT,
        "required_files": {},
        "submodules": [],
        "source_contract": {},
        "status": "pass",
    }
    if not repo_dir.is_dir():
        _block(blockers, check_name, "repository_missing", f"OpenPI 仓库不存在：{repo_dir}")
        result["status"] = "blocked"
        return result
    if not (repo_dir / ".git").exists():
        _block(blockers, check_name, "repository_not_git", f"目录不是 Git 仓库：{repo_dir}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    commit = completed.stdout.strip() if completed and completed.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        _block(blockers, check_name, "repository_commit_missing", "无法解析 OpenPI 的 40 位 Git commit，版本尚未锁定")
    else:
        result["commit"] = commit
        if commit != PINNED_COMMIT:
            _block(blockers, check_name, "repository_commit_mismatch", f"当前 commit {commit} 与核查锁定版本 {PINNED_COMMIT} 不同")
        try:
            changed = subprocess.run(
                ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            if changed.returncode or changed.stdout.strip():
                _block(blockers, check_name, "repository_dirty", "OpenPI 工作树已改动或无法核实，commit 不能代表当前源码")
        except (OSError, subprocess.TimeoutExpired):
            _block(blockers, check_name, "repository_status_unknown", "无法核查 OpenPI 工作树状态")

    required_files = (
        "README.md",
        "pyproject.toml",
        "src/openpi/models/pi0_config.py",
        "src/openpi/training/config.py",
        "src/openpi/training/data_loader.py",
        "src/openpi/transforms.py",
        "scripts/compute_norm_stats.py",
        "scripts/train.py",
        "scripts/train_pytorch.py",
        "scripts/serve_policy.py",
    )
    result["required_files"] = {path: (repo_dir / path).is_file() for path in required_files}
    for path, exists in result["required_files"].items():
        if not exists:
            _block(blockers, check_name, "repository_source_missing", f"OpenPI 源码缺少官方文件：{path}")

    try:
        readme = (repo_dir / "README.md").read_text(encoding="utf-8")
        config = (repo_dir / "src/openpi/training/config.py").read_text(encoding="utf-8")
        pi0_config = (repo_dir / "src/openpi/models/pi0_config.py").read_text(encoding="utf-8")
    except OSError:
        readme = config = pi0_config = ""
    result["source_contract"] = {
        "gpu_memory_from_readme": {
            "lora_min_gb": _extract_memory(readme, r"Fine-Tuning \(LoRA\)"),
            "full_min_gb": _extract_memory(readme, r"Fine-Tuning \(Full\)"),
        },
        "pi05_default_action_dim": _extract_int_after(pi0_config, "action_dim: int ="),
        "pi05_configs": sorted(set(re.findall(r'name="(pi05[^" ]*)"', config))),
        "official_entrypoints": {
            "norm_stats": "scripts/compute_norm_stats.py",
            "jax_train": "scripts/train.py",
            "pytorch_train": "scripts/train_pytorch.py",
            "inference": "scripts/serve_policy.py",
        },
        "pytorch_lora_supported": False if "LoRA (low-rank adaptation) training" in readme else None,
        "model_rgb_slots": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    }
    try:
        submodule_status = subprocess.run(
            ["git", "-C", str(repo_dir), "submodule", "status"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        result["submodules"] = [line for line in submodule_status.stdout.splitlines() if line]
    except (OSError, subprocess.TimeoutExpired):
        result["submodules"] = []
    result["status"] = _status(blockers, check_name)
    return result


def _extract_memory(text: str, label: str) -> str | None:
    match = re.search(rf"{label}.*?\|\s*([^|]+GB)\s*\|", text)
    return match.group(1).strip() if match else None


def _extract_int_after(text: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _shape(feature: Any) -> list[int] | None:
    value = feature.get("shape") if isinstance(feature, dict) else None
    if not isinstance(value, list) or not value or not all(type(item) is int and item > 0 for item in value):
        return None
    return value


def _feature_names(feature: Any) -> list[str] | None:
    value = feature.get("names") if isinstance(feature, dict) else None
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else None


def _check_dataset(data_dir: Path, blockers: list[dict[str, str]]) -> dict[str, Any]:
    check_name = "dataset"
    result: dict[str, Any] = {
        "path": str(data_dir),
        "metadata": str(data_dir / "meta/info.json"),
        "features": {},
        "image_keys": [],
        "state_dim": None,
        "action_dim": None,
        "state_names": None,
        "action_names": None,
        "openpi_model_action_dim": DEFAULT_PI05_ACTION_DIM,
        "status": "pass",
    }
    if not data_dir.is_dir():
        _block(blockers, check_name, "dataset_missing", f"LeRobot 数据集目录不存在：{data_dir}")
        result["status"] = "blocked"
        return result
    metadata_path = data_dir / "meta/info.json"
    if not metadata_path.is_file():
        _block(blockers, check_name, "dataset_metadata_missing", f"缺少 LeRobot 元数据：{metadata_path}")
        result["status"] = "blocked"
        return result
    info, error = _read_json(metadata_path)
    if info is None:
        _block(blockers, check_name, "dataset_metadata_invalid", f"无法读取 LeRobot 元数据 {metadata_path}：{error}")
        result["status"] = "blocked"
        return result
    features = info.get("features")
    if not isinstance(features, dict):
        _block(blockers, check_name, "dataset_features_missing", "LeRobot info.json 缺少 features 对象")
        result["status"] = "blocked"
        return result
    result["features"] = features
    if type(info.get("fps")) not in (int, float) or not 0 < info["fps"] < float("inf"):
        _block(blockers, check_name, "dataset_fps_missing", "LeRobot info.json 缺少有效 fps")

    state_feature = features.get("observation.state", features.get("state"))
    action_feature = features.get("action", features.get("actions"))
    state_shape = _shape(state_feature)
    action_shape = _shape(action_feature)
    result["state_dim"] = state_shape[-1] if state_shape and len(state_shape) == 1 else None
    result["action_dim"] = action_shape[-1] if action_shape and len(action_shape) == 1 else None
    result["state_names"] = _feature_names(state_feature)
    result["action_names"] = _feature_names(action_feature)
    if result["state_dim"] is None:
        _block(blockers, check_name, "dataset_state_feature_missing", "LeRobot 数据没有一维 observation.state 特征")
    if result["action_dim"] is None:
        _block(blockers, check_name, "dataset_action_feature_missing", "LeRobot 数据没有一维 action 特征")
    if result["state_dim"] is not None and result["action_dim"] is not None:
        if result["state_dim"] != result["action_dim"]:
            _block(
                blockers,
                check_name,
                "dataset_state_action_dim_mismatch",
                f"state 维度 {result['state_dim']} 与 action 维度 {result['action_dim']} 不一致",
            )
        if result["action_dim"] > DEFAULT_PI05_ACTION_DIM:
            _block(
                blockers,
                check_name,
                "dataset_action_dim_exceeds_pi05",
                f"action 维度 {result['action_dim']} 大于 π0.5 默认 action_dim={DEFAULT_PI05_ACTION_DIM}，不能靠零填充适配",
            )
    if not result["state_names"] or len(result["state_names"]) != result["state_dim"]:
        _block(blockers, check_name, "dataset_state_names_missing", "未提供与 state 维度一致的 G1 状态名称，无法核实单臂关节映射")
    if not result["action_names"] or len(result["action_names"]) != result["action_dim"]:
        _block(blockers, check_name, "dataset_action_names_missing", "未提供与 action 维度一致的 G1 动作名称，无法核实单臂关节映射")
    action_names = result["action_names"] or []
    if action_names and not any("gripper" in name.lower() or "finger" in name.lower() for name in action_names):
        _block(blockers, check_name, "dataset_gripper_dimension_unresolved", "action 名称中没有 gripper/finger，夹爪维度未核实")

    image_keys = [
        key
        for key, feature in features.items()
        if isinstance(key, str)
        and ("image" in key.lower() or "camera" in key.lower())
        and isinstance(feature, dict)
        and feature.get("dtype") in {"image", "video"}
        and (shape := _shape(feature)) is not None
        and len(shape) == 3
        and (shape[0] == 3 or shape[-1] == 3)
    ]
    result["image_keys"] = sorted(image_keys)
    if len(image_keys) < 2:
        _block(blockers, check_name, "dataset_rgb_views_insufficient", "LeRobot 元数据少于两个 RGB 视角，无法满足双视角输入")
    roles = {
        "base": any(any(token in key.lower() for token in ("head", "base", "front", "top", "exterior")) for key in image_keys),
        "wrist": any(any(token in key.lower() for token in ("wrist", "hand")) for key in image_keys),
    }
    result["rgb_roles"] = roles
    if len(image_keys) >= 2 and not all(roles.values()):
        _block(blockers, check_name, "dataset_rgb_role_mapping_missing", "双视角尚未明确映射为 OpenPI 的 base 与 wrist 输入")
    if not any(item.stat().st_size > 0 for item in (data_dir / "data").glob("**/*.parquet")):
        _block(blockers, check_name, "dataset_records_missing", "LeRobot data/ 中没有非空 Parquet 记录")
    task_files = tuple(data_dir / f"meta/tasks.{ext}" for ext in ("jsonl", "parquet", "json"))
    if not any(path.is_file() for path in task_files) and "prompt" not in features:
        _block(blockers, check_name, "dataset_tasks_missing", "没有 tasks 元数据或 prompt 特征，π0.5 无法得到语言指令")
    result["custom_g1_mapping"] = {
        "status": "metadata_declared" if result["state_names"] and result["action_names"] else "blocked",
        "runtime_verified": False,
        "requirement": "单臂关节名称 + 夹爪名称必须由采集元数据提供；本工具不从场景文件猜测维度",
    }
    result["status"] = _status(blockers, check_name)
    return result


def _check_checkpoint(checkpoint_value: str | Path | None, project_root: Path, blockers: list[dict[str, str]]) -> dict[str, Any]:
    check_name = "checkpoint"
    value = str(checkpoint_value) if checkpoint_value is not None else str(project_root / "checkpoints")
    result: dict[str, Any] = {
        "path": value,
        "format": None,
        "weights_present": False,
        "norm_stats": [],
        "status": "pass",
    }
    if value.startswith(("gs://", "s3://", "hf://", "http://", "https://")):
        _block(blockers, check_name, "checkpoint_remote_unchecked", "checkpoint 是远程地址；本工具不下载权重，只核查本地路径")
        result["status"] = "blocked"
        return result
    path = Path(value)
    if not path.is_dir():
        _block(blockers, check_name, "checkpoint_missing", f"本地 checkpoint 目录不存在：{path}")
        result["status"] = "blocked"
        return result
    if (path / "model.safetensors").is_file() and (path / "model.safetensors").stat().st_size > 0:
        result["format"] = "pytorch"
        result["weights_present"] = True
    elif (path / "params").is_dir() and any(item.is_file() and item.stat().st_size > 0 for item in (path / "params").glob("**/*")):
        result["format"] = "jax"
        result["weights_present"] = True
    else:
        _block(blockers, check_name, "checkpoint_weights_missing", "checkpoint 没有 model.safetensors 或 params")
    result["norm_stats"] = [str(item.relative_to(path)) for item in path.glob("assets/**/norm_stats.json")]
    if not result["norm_stats"]:
        _block(blockers, check_name, "checkpoint_norm_stats_missing", "checkpoint 缺少 assets/<asset_id>/norm_stats.json")
    for relative_path in result["norm_stats"]:
        stats_json, error = _read_json(path / relative_path)
        stats = stats_json.get("norm_stats", {}) if stats_json else {}
        valid = isinstance(stats, dict)
        for key in ("state", "actions"):
            entry = stats.get(key, {}) if isinstance(stats, dict) else {}
            arrays = [entry.get(name) for name in ("mean", "std", "q01", "q99")] if isinstance(entry, dict) else []
            valid = valid and len(arrays) == 4 and all(
                isinstance(array, list) and array and all(type(value) in (int, float) and math.isfinite(value) for value in array)
                for array in arrays
            )
            if valid:
                valid = len({len(array) for array in arrays}) == 1 and all(value >= 0 for value in arrays[1])
        if not valid:
            _block(blockers, check_name, "checkpoint_norm_stats_invalid", f"π0.5 norm stats 无效：{relative_path}；需要 state/actions 的 mean、std、q01、q99 有限数值数组。{error or ''}")
    result["status"] = _status(blockers, check_name)
    return result


def run_checks(
    *,
    project_root: Path,
    repo_dir: Path | None = None,
    data_dir: Path | None = None,
    checkpoint_dir: Path | str | None = None,
    venv_dir: Path | None = None,
) -> dict[str, Any]:
    """执行只读核查并返回可序列化的 JSON 对象。"""

    project_root = Path(project_root).resolve()
    repo_dir = Path(repo_dir or project_root / "software/openpi")
    data_dir = Path(data_dir or project_root / "data/medication_sorting")
    venv_dir = Path(venv_dir or project_root / ".venv")
    blockers: list[dict[str, str]] = []
    result: dict[str, Any] = {
        "schema_version": 1,
        "verification_scope": "local_metadata_only",
        "ready": False,
        "repository": _repository_facts(repo_dir, blockers),
        "environment": _probe_environment(venv_dir, repo_dir, blockers),
        "dataset": _check_dataset(data_dir, blockers),
        "checkpoint": _check_checkpoint(checkpoint_dir, project_root, blockers),
        "blockers": blockers,
        "execution": {
            "training_started": False,
            "weights_downloaded": False,
            "gpu_touched": False,
            "data_written": False,
        },
    }
    result["ready"] = not blockers
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="只读核查 OpenPI π0.5 训练前置条件，输出 JSON")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repo", type=Path, default=None, help="OpenPI 本地仓库路径")
    parser.add_argument("--data", type=Path, default=None, help="LeRobot 数据集根目录")
    parser.add_argument("--checkpoint", default=None, help="本地 checkpoint 目录；远程地址只报告阻塞，不下载")
    parser.add_argument("--venv", type=Path, default=None, help="固定项目环境路径，默认 .venv")
    parser.add_argument("--pretty", action="store_true", help="格式化 JSON")
    args = parser.parse_args()
    result = run_checks(
        project_root=args.project_root,
        repo_dir=args.repo,
        data_dir=args.data,
        checkpoint_dir=args.checkpoint,
        venv_dir=args.venv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True, allow_nan=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
