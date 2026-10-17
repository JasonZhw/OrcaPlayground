"""唯一 .venv 的 OpenPI JAX LoRA 依赖层安装、元数据检查与运行入口。

默认 plan/--dry-run 完全离线，不运行 pip、不创建目录。install --execute
显式运行安装入口，仅向 .venv/pi05-deps 写包，不修改 .venv 的 site-packages。
registry 包一次 --no-deps 安装；LeRobot 单独按 Git commit --no-deps 安装。
完整传递依赖由 pinned uv.lock 的 Python3.12/Linux 分支展开，避免 resolver
升级 Torch 或把 OrcaGym 的 NumPy2 约束拉进训练层。

JAX/jaxlib/CUDA12 plugin/PJRT 均为 0.5.3；CUDA runtime/cuPTI 12.9.79，
nvcc/nvrtc/nvJitLink 12.9.86，cuDNN 9.10.2.21。NVIDIA 支持矩阵明确列出
CUDA12.9、Blackwell、SM12.0；这不是当前机器已经通过 JAX GPU 验证的结论。
Torch 保留 2.14.1+cu130，torchvision 用 0.29.1（metadata 要求 Torch>=2.14）。
这是混合版本环境，必须验证 NumPy/Torch bridge、实际 LeRobot 写读、JAX
GPU bf16 GEMM/conv，并由主线程完成真实数据/权重的一步训练及推理。

transformers==4.53.2 必须安装：OpenPI 的 JAX 公共模块仍直接导入
AutoProcessor、GemmaForCausalLM、PaliGemmaForConditionalGeneration。
JAX 路线无需复制 transformers_replace；不要修改基环境的 transformers。
LeRobot 必须是 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5，API 路径为
lerobot.common.datasets.lerobot_dataset，使用 create/add_frame/save_episode。
省去硬件设备驱动、RLDS/TensorFlow、torchcodec；只使用 image 数据和 pyav。

用法：
  .venv/bin/python tools/medication_sorting/setup_pi05.py --dry-run
  .venv/bin/python tools/medication_sorting/setup_pi05.py metadata
  .venv/bin/python tools/medication_sorting/setup_pi05.py install --execute
  tools/medication_sorting/pi05_python.sh --cpu -m tools.medication_sorting.setup_pi05 verify
  tools/medication_sorting/pi05_python.sh --dry-run -m tools.medication_sorting.pi05_adapter train -- --exp-name=g1_trial

GPU 验证要先与主线程协调，再显式运行 wrapper ... setup_pi05 verify --gpu。
本工具不后台训练、benchmark、连接仿真或自动下载模型权重。
说明与精确步骤见 docs/medication_sorting/pi05_dependency_layer.md。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tomllib
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT_COMMIT = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
MANIFEST = "pi05-layer-manifest.json"
CUDA_PINS = {
    "nvidia-cublas-cu12": "12.9.1.4",
    "nvidia-cuda-cupti-cu12": "12.9.79",
    "nvidia-cuda-nvcc-cu12": "12.9.86",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cudnn-cu12": "9.10.2.21",
    "nvidia-cufft-cu12": "11.4.1.4",
    "nvidia-cusolver-cu12": "11.7.5.82",
    "nvidia-cusparse-cu12": "12.5.10.65",
    "nvidia-nccl-cu12": "2.26.5",
    "nvidia-nvjitlink-cu12": "12.9.86",
}
OVERRIDES = {**CUDA_PINS, "numpy": "1.26.4", "torchvision": "0.29.1", "wrapt": "1.17.3"}
ROOT_PACKAGES = {
    "augmax", "dm-tree", "einops", "equinox", "flax", "jax", "jaxlib",
    "jax-cuda12-plugin", "jax-cuda12-pjrt", "optax", "chex", "jaxtyping",
    "beartype", "orbax-checkpoint", "ml-collections", "ml-dtypes", "tensorstore",
    "numpydantic", "opencv-python-headless", "imageio", "sentencepiece",
    "transformers", "tyro", "tqdm-loggable", "wandb", "treescope", "polars",
    "rich", "filelock", "flatbuffers", "typing-extensions", "numpy", "scipy",
    "datasets", "pyarrow", "av", "draccus", "jsonlines", "torchvision",
    "gcsfs", "msgpack", "websockets", *CUDA_PINS,
}
OMITTED = [
    "torch（复用基环境）", "gym-aloha", "gymnasium", "cmake", "diffusers",
    "flask", "gdown", "h5py", "numba", "omegaconf", "pymunk", "pynput",
    "pyzmq", "rerun-sdk", "torchcodec", "zarr", "硬件驱动 extras", "RLDS/TensorFlow/dlimp",
]
SOURCES = [
    "https://pypi.org/pypi/jax-cuda12-plugin/0.5.3/json",
    "https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html",
    "https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html",
    "https://pypi.org/pypi/torchvision/0.29.1/json",
    f"https://github.com/huggingface/lerobot/tree/{LEROBOT_COMMIT}/lerobot/common/datasets",
]


def check_openpi_checkout(root: Path = PROJECT_ROOT) -> None:
    repository = Path(root) / "software/openpi"
    result = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if result.stdout.strip() != OPENPI_COMMIT:
        raise RuntimeError(f"OpenPI 必须固定在 {OPENPI_COMMIT}")
    status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise RuntimeError("上游 OpenPI 存在已跟踪文件修改，不能用作 pinned 安装来源")
    project = tomllib.loads((repository / "pyproject.toml").read_text())
    if project["tool"]["uv"]["sources"]["lerobot"]["rev"] != LEROBOT_COMMIT:
        raise RuntimeError("LeRobot pinned commit 与安装入口不一致")


def registry_pins(lock_path: Path) -> dict[str, str]:
    """展开官方锁文件的依赖及明确使用的 extras；不调用 pip resolver。"""
    from packaging.markers import Marker, default_environment

    environment = {**default_environment(), "extra": ""}
    lock = tomllib.loads(Path(lock_path).read_text())
    packages = {}
    for package in lock["package"]:
        if package.get("version") is None or "registry" not in package.get("source", {}):
            continue
        markers = package.get("resolution-markers", [])
        if not markers or any(Marker(marker).evaluate(environment) for marker in markers):
            if package["name"] in packages:
                raise RuntimeError(f"锁文件分支不唯一：{package['name']}")
            packages[package["name"]] = package
    pins, visited = {}, set()
    pending = [(name, ()) for name in sorted(ROOT_PACKAGES)]
    while pending:
        name, extras = pending.pop()
        if name == "torch" or (name, extras) in visited:
            continue
        visited.add((name, extras))
        package = packages.get(name)
        if package is None:
            raise RuntimeError(f"官方 uv.lock 缺少当前平台包：{name}")
        pins[name] = OVERRIDES.get(name, package["version"])
        dependencies = list(package.get("dependencies", []))
        for extra in extras:
            dependencies.extend(package.get("optional-dependencies", {}).get(extra, []))
        for dependency in dependencies:
            marker = dependency.get("marker")
            if marker is None or Marker(marker).evaluate(environment):
                pending.append((dependency["name"], tuple(sorted(dependency.get("extra", [])))))
    return dict(sorted(pins.items()))


def build_install_plan(*, project_root: Path = PROJECT_ROOT, check_checkout: bool = True) -> dict:
    root = Path(project_root).absolute()
    if check_checkout:
        check_openpi_checkout(root)
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("此入口固定适配当前 Python3.12 / Linux x86_64")
    # 测试可以只替换输出根目录，源码和 lock 仍来自固定、只读的项目仓库。
    lock_path = (root if check_checkout else PROJECT_ROOT) / "software/openpi/uv.lock"
    pins = registry_pins(lock_path)
    target = root / ".venv/pi05-deps"
    python = root / ".venv/bin/python"
    common = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
              "--no-deps", "--target", str(target)]
    steps = [
        {"name": "锁定的 Python/JAX/CUDA12 依赖", "argv": [*common, "--only-binary=:all:",
         *[f"{name}=={version}" for name, version in pins.items()]]},
        {"name": "仅安装官方 pinned LeRobot 源码", "argv": [*common,
         f"git+https://github.com/huggingface/lerobot.git@{LEROBOT_COMMIT}"]},
    ]
    return {
        "python": str(python), "target": str(target), "project_root": str(root),
        "openpi_commit": OPENPI_COMMIT, "lerobot_commit": LEROBOT_COMMIT,
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "registry_pins": pins, "steps": steps,
        "commands": [shlex.join(step["argv"]) for step in steps],
        "omitted": OMITTED, "source_imports": [str(root / "software/openpi/src"), str(root / "software/openpi/packages/openpi-client/src")],
        "official_environment": False, "end_to_end_required": True,
        "warnings": [
            "保留 Torch2.14.1+cu130，偏离官方 Torch2.7.1；需实际端到端训练和推理验证",
            "CUDA12/13 的 cuDNN 和 NCCL 有相同 SONAME，PYTHONPATH 无法证明原生库完全隔离",
            "torchvision0.29.1 metadata 允许现有 Torch；pinned LeRobot 的旧 API 仍必须实测",
            "wrapt 从官方锁1.14.1 调整到1.17.3：前者没有 Python3.12 wheel，后者满足 dm-tree/deprecated 的约束",
        ],
        "download_estimate": "约 3–4 GB wheels，加 LeRobot Git 源码与临时构建依赖；metadata 命令给出逐包估算",
        "disk_estimate": "建议至少 10–12 GB 空闲磁盘，含解压 target 与 pip 缓存；不含 pi05_base 权重",
        "sources": SOURCES, "executed": False, "network_accessed": False, "gpu_touched": False,
    }


def _protected_snapshot() -> dict:
    names = ("torch", "numpy", "orca-gym", "orca-lab", "orcalab-pyside", "opencv-python")
    # wrapper 中 NumPy 来自 target；受保护快照始终读取基环境，不能按 sys.path 查询。
    base = PROJECT_ROOT / ".venv/lib/python3.12/site-packages"
    from packaging.utils import canonicalize_name
    distributions = {canonicalize_name(package.metadata["Name"]): package
                     for package in importlib.metadata.distributions(path=[str(base)])
                     if package.metadata["Name"]}
    result = {}
    for name in names:
        package = distributions.get(canonicalize_name(name))
        if package is None:
            result[name] = None
        else:
            result[name] = {"version": package.version,
                            "metadata_sha256": hashlib.sha256((package.read_text("METADATA") or "").encode()).hexdigest(),
                            "record_sha256": hashlib.sha256((package.read_text("RECORD") or "").encode()).hexdigest()}
    return result


def execute_install(plan: dict, *, execute: bool = False, retry: bool = False) -> dict:
    if not execute:
        return plan
    target = Path(plan["target"])
    expected = Path(plan["project_root"]) / ".venv/pi05-deps"
    if target != expected or target.is_symlink() or target.resolve() != expected.absolute():
        raise ValueError("target 必须是本项目 .venv/pi05-deps，不能是符号链接或 site-packages")
    partial = None
    if target.exists() and any(target.iterdir()):
        if not retry:
            raise FileExistsError("target 已非空；已记录的 partial 安装可用 install --execute --retry 恢复")
        try:
            partial = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError("非空 target 缺少安装快照，不能自动重试覆盖") from error
        if partial.get("installed") is not False:
            raise RuntimeError("--retry 仅恢复已记录的 partial 安装；不覆盖完整安装")
        for key in ("python", "project_root", "target", "openpi_commit", "lerobot_commit", "lock_sha256", "registry_pins"):
            if partial.get(key) != plan.get(key):
                raise RuntimeError(f"partial 安装计划/版本不匹配：{key}")
    if Path(sys.executable).resolve() != Path(plan["python"]).resolve():
        raise RuntimeError("实际安装必须使用唯一 .venv/bin/python")
    before = _protected_snapshot()
    if partial is not None and partial.get("protected_before") != before:
        raise RuntimeError("受保护包快照与安装失败前不一致，拒绝重试")
    target.mkdir(parents=True, exist_ok=True)
    report = {**plan, "executed": True, "network_accessed": True, "protected_before": before,
              "installed": False, "verified_end_to_end": False}
    manifest_path = target / MANIFEST
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("PIP_") or key in {"PYTHONPATH", "PYTHONHOME"}:
            environment.pop(key)
    environment.update(PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        for step in plan["steps"]:
            # --target 不带 --upgrade 会跳过 partial 包目录；重试同一精确计划即可修复。
            command = [*step["argv"], *(["--upgrade"] if partial is not None else [])]
            subprocess.run(command, env=environment, cwd=plan["project_root"], check=True)
    finally:
        if _protected_snapshot() != before:
            raise RuntimeError("基环境受保护包 metadata 发生变化，停止；不能声称环境保持不变")
    report["installed"] = True
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def check_layer(target: Path) -> dict:
    target = Path(target)
    for name in ("torch", "torch.py", "orca_gym", "orcalab", "sitecustomize.py", "usercustomize.py"):
        if (target / name).exists():
            raise RuntimeError(f"训练层禁止覆盖受保护模块：{name}")
    if target.is_symlink() or list(target.glob("torch-*.dist-info")):
        raise RuntimeError("训练层不能是符号链接或包含替代 torch distribution")
    try:
        manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("缺少有效安装 manifest；请先显式运行安装入口 install --execute") from error
    if manifest.get("installed") is not True or manifest.get("openpi_commit") != OPENPI_COMMIT or manifest.get("lerobot_commit") != LEROBOT_COMMIT:
        raise RuntimeError("安装 manifest 未完成或 commit 不匹配")
    if manifest.get("protected_before") != _protected_snapshot():
        raise RuntimeError("受保护包快照已变化，需要重新核对混合环境")
    return manifest


def runtime_environment(parent: dict | None = None, *, cpu: bool = False, project_root: Path = PROJECT_ROOT) -> dict:
    root = Path(project_root).absolute()
    target = root / ".venv/pi05-deps"
    environment = dict(os.environ if parent is None else parent)
    for key in ("PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        environment.pop(key, None)
    environment.update(
        PYTHONPATH=os.pathsep.join(map(str, (target, root, root / "software/openpi/src", root / "software/openpi/packages/openpi-client/src"))),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PI05_DEPS_LAYER=str(target),
        # Torch 2.14 的 _preload_cuda_deps 按 sys.path 找到 CU12 NCCL2.26，
        # 缺少 CU13 Torch 所需 ncclCommResume。让它通过原生 RPATH 加载，
        # 只改变当前子进程；仍需实测 JAX 的混合原生库路径。
        TORCH_USE_RTLD_GLOBAL="1",
    )
    # 按 JAX pip CUDA 的建议不设置 LD_LIBRARY_PATH；不要把 CU12 动态库强塞给 CU13 Torch。
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    nvcc = target / "nvidia/cuda_nvcc/bin"
    environment["PATH"] = str(nvcc) + os.pathsep + environment.get("PATH", os.defpath)
    if cpu:
        environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    return environment


def runtime_command(arguments: list[str], *, project_root: Path = PROJECT_ROOT) -> list[str]:
    if not arguments:
        raise ValueError("wrapper 需要 Python 参数，例如 -m tools.medication_sorting.pi05_adapter")
    # 只解析解释器参数；模块/脚本后的同名参数必须原样交给应用。
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if not argument.startswith("-") or argument in {"-", "--"}:
            break
        if argument.startswith("--"):
            index += 2 if argument == "--check-hash-based-pycs" else 1
            continue
        flags = argument[1:]
        stop = False
        for offset, flag in enumerate(flags):
            if flag in "IES":
                raise ValueError("-I/-E/-S 会绕过训练依赖层或基环境，不能用于此 wrapper")
            if flag in "mcWX":
                value = flags[offset + 1:]
                if not value:
                    index += 1
                    value = arguments[index] if index < len(arguments) else ""
                if flag == "m" and value.split(".")[0] in {"pip", "venv", "ensurepip"}:
                    raise ValueError("不要通过训练 wrapper 安装包或创建环境；使用 setup_pi05 install --execute")
                stop = flag in "mc"
                break
        if stop:
            break
        index += 1
    return [str(Path(project_root).absolute() / ".venv/bin/python"), "-s", *arguments]


def run_python(arguments: list[str], *, dry_run: bool = False, cpu: bool = False) -> dict:
    command = runtime_command(arguments)
    environment = runtime_environment(cpu=cpu)
    report = {"argv": command, "environment": {key: environment[key] for key in
              ("PYTHONPATH", "PATH", "PYTHONNOUSERSITE", "PI05_DEPS_LAYER", "TORCH_USE_RTLD_GLOBAL", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
              "removed_environment": ["LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME"], "executed": False, "gpu_touched": False}
    if cpu:
        report["environment"].update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    if dry_run:
        return report
    check_openpi_checkout()
    check_layer(PROJECT_ROOT / ".venv/pi05-deps")
    os.execve(command[0], command, environment)
    raise AssertionError("execve 不应返回")


def dependency_conflicts(packages: list[dict], pins: dict[str, str], *, torch_version: str) -> list[dict]:
    """核对发行版基础约束及 JAX CUDA extras；不执行或模拟 pip resolver。"""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    extras = {"jax": ("cuda12",), "jax-cuda12-plugin": ("with-cuda",), "fsspec": ("gcs",)}
    versions = {**pins, "torch": torch_version}
    conflicts = []
    for package in packages:
        for raw in package["requires_dist"]:
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra})
                    for extra in ("", *extras.get(package["package"], ()))):
                continue
            name = canonicalize_name(requirement.name)
            selected = versions.get(name)
            if selected is None or selected not in requirement.specifier:
                conflicts.append({"parent": package["package"], "package": name,
                                  "requirement": raw, "selected": selected})
    return conflicts


def metadata_report(plan: dict) -> dict:
    """只获取 PyPI JSON 元数据，不下载 wheels；按本机 tag 估计压缩下载量。"""
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    tags = set(sys_tags())
    def fetch(item):
        name, version = item
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        with urlopen(url, timeout=30) as stream:
            data = json.load(stream)
        wheels = []
        for wheel in data["urls"]:
            if wheel["filename"].endswith(".whl") and parse_wheel_filename(wheel["filename"])[3] & tags:
                wheels.append(wheel)
        if not wheels:
            raise RuntimeError(f"{name}=={version} 无兼容 wheel；不应默默进入源码编译")
        wheel = min(wheels, key=lambda value: value["size"])
        return {"package": name, "version": version, "filename": wheel["filename"], "bytes": wheel["size"],
                "sha256": wheel["digests"]["sha256"], "metadata_url": url,
                "requires_dist": data["info"].get("requires_dist") or []}
    with ThreadPoolExecutor(max_workers=8) as pool:
        packages = list(pool.map(fetch, plan["registry_pins"].items()))
    conflicts = dependency_conflicts(packages, plan["registry_pins"], torch_version=importlib.metadata.version("torch"))
    return {"packages": packages, "download_bytes": sum(package["bytes"] for package in packages),
            "download_gib": round(sum(package["bytes"] for package in packages) / 1024**3, 3),
            "dependency_conflicts": conflicts, "selected_registry_constraints_consistent": not conflicts,
            "constraint_scope": "registry_base_plus_jax_cuda12_plugin_with_cuda_fsspec_gcs_not_full_lerobot_hardware",
            "lerobot_git_and_build_dependencies_not_counted": True, "wheels_downloaded": False, "gpu_touched": False}


def verify_runtime(*, gpu: bool = False) -> dict:
    """安装后显式运行的真实数据链路检查；不声称验证了训练或推理。"""
    layer = PROJECT_ROOT / ".venv/pi05-deps"
    manifest = check_layer(layer)
    if os.environ.get("PI05_DEPS_LAYER") != str(layer):
        raise RuntimeError("verify 必须通过 pi05_python.sh 的依赖层运行")
    if not gpu and (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("JAX_PLATFORMS") != "cpu"):
        raise RuntimeError("CPU verify 请使用 pi05_python.sh --cpu；GPU 检查需主线程协调后显式 --gpu")
    import tempfile
    import numpy as np
    import torch
    import torchvision
    from transformers import AutoProcessor, GemmaForCausalLM, PaliGemmaForConditionalGeneration
    import jax
    from openpi import transforms
    from tools.medication_sorting import convert_episodes, pi05_adapter

    if not Path(np.__file__).resolve().is_relative_to(layer.resolve()):
        raise RuntimeError("NumPy 未从 pi05-deps 加载，Python 依赖隔离失败")
    if Path(torch.__file__).resolve().is_relative_to(layer.resolve()):
        raise RuntimeError("Torch 被依赖层替换")
    np.testing.assert_array_equal(torch.from_numpy(np.arange(8, dtype=np.float32)).numpy(), np.arange(8, dtype=np.float32))
    dataset_class = convert_episodes.require_lerobot()
    with tempfile.TemporaryDirectory(prefix="g1-pi05-data-check-") as temporary:
        directory = Path(temporary)
        raw = directory / "raw/episode_000"
        raw.mkdir(parents=True)
        np.savez(raw / "episode.npz", state=np.zeros((4, 8), np.float32), action=np.full((4, 8), 0.2, np.float32),
                 base_rgb=np.tile(np.arange(768, dtype=np.uint8).reshape(1, 16, 16, 3), (4, 1, 1, 1)),
                 wrist_rgb=np.tile((np.arange(768, dtype=np.uint8) + 30).reshape(1, 16, 16, 3), (4, 1, 1, 1)),
                 sim_time=np.arange(4, dtype=np.float64) / 20, phase=np.array(["approach", "grasp", "place", "release"]))
        metadata = {"success": True, "task": pi05_adapter.DEFAULT_TASK, "joint_names": [f"right_joint_{i}" for i in range(7)],
                    "gripper_semantics": {"open": 0, "closed": 1},
                    "camera_metadata": {key: {"color_space": "RGB", "layout": "HWC"} for key in ("base_rgb", "wrist_rgb")}}
        (raw / "metadata.json").write_text(json.dumps(metadata))
        output = directory / "converted"
        conversion = convert_episodes.convert_episodes(raw.parent, output, repo_id="local/g1_smoke", fps=20)
        dataset = dataset_class("local/g1_smoke", root=output, delta_timestamps={"action": [0, 0.05]})
        sample = dataset[0]
        expected = sample["action"].numpy().copy()
        group = pi05_adapter.make_data_transforms()
        data = {"state": sample["observation.state"].numpy(), "base_rgb": sample["observation.images.base"].numpy(),
                "wrist_rgb": sample["observation.images.right_wrist"].numpy(), "actions": expected, "prompt": sample["task"]}
        for transform in group.inputs:
            data = transform(data)
        data = transforms.PadStatesAndActions(32)(data)
        for transform in group.outputs:
            data = transform(data)
        np.testing.assert_allclose(data["actions"], expected, atol=2e-7)
    # 创建配置会导入真实 OpenPI 公共模块，但不实例化模型或下载 tokenizer/权重。
    configuration = pi05_adapter.create_train_config("local/g1_smoke")
    report = {"versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "torchvision", "jax", "jaxlib", "transformers", "lerobot")},
              "numpy_origin": np.__file__, "torch_origin": torch.__file__, "same_python": sys.executable,
              "numpy_torch_bridge": True, "lerobot_parquet_write_read": True,
              "g1_transform_roundtrip": True, "converted_frames": conversion["converted_frames"],
              "lora_config": configuration.name, "official_environment": False,
              "training_step_tested": False, "inference_tested": False, "gpu_touched": gpu,
              "warnings": manifest["warnings"]}
    if gpu:
        devices = jax.devices("gpu")
        import jax.numpy as jnp
        matrix = jnp.ones((32, 32), dtype=jnp.bfloat16)
        product = jax.jit(lambda value: value @ value)(matrix)
        np.testing.assert_allclose(np.asarray(product, dtype=np.float32), np.full((32, 32), 32))
        image = jnp.ones((1, 8, 8, 3), jnp.bfloat16)
        kernel = jnp.ones((3, 3, 3, 4), jnp.bfloat16)
        convolution = jax.jit(lambda a, b: jax.lax.conv_general_dilated(a, b, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")))(image, kernel)
        np.testing.assert_allclose(np.asarray(convolution, dtype=np.float32), np.full((1, 6, 6, 4), 27))
        report.update(gpu_devices=[str(device) for device in devices], gpu_bf16_gemm=True, gpu_bf16_conv=True,
                      gpu_scope="small_ops_only_no_training_or_benchmark")
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--dry-run"]:
        arguments = ["plan"]
    parser = argparse.ArgumentParser(description="同一 .venv 的 π0.5 依赖层：默认只打印计划；显式运行安装入口。")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("plan", help="离线构造完整安装命令，无副作用")
    sub.add_parser("metadata", help="只读取 PyPI JSON，核对 wheels 和压缩下载估算")
    installer = sub.add_parser("install", help="默认 dry-run；--execute 才写入 pi05-deps")
    installer.add_argument("--execute", action="store_true", help="显式执行 target 安装，不修改基环境")
    installer.add_argument("--dry-run", action="store_true")
    installer.add_argument("--retry", action="store_true", help="核对计划与受保护快照后修复 partial target")
    runner = sub.add_parser("run", help="wrapper 使用的 Python 进程入口")
    runner.add_argument("--dry-run", action="store_true")
    runner.add_argument("--cpu", action="store_true")
    runner.add_argument("arguments", nargs=argparse.REMAINDER)
    verifier = sub.add_parser("verify", help="安装后真实 NumPy/Torch/LeRobot/G1 CPU 数据链路检查")
    verifier.add_argument("--gpu", action="store_true", help="主线程协调 GPU 后显式验证小规模 bf16 GEMM/conv")
    args = parser.parse_args(arguments)
    try:
        if args.mode == "run":
            values = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            report = run_python(values, dry_run=args.dry_run, cpu=args.cpu)
        elif args.mode == "verify":
            report = verify_runtime(gpu=args.gpu)
        else:
            plan = build_install_plan()
            if args.mode == "metadata":
                report = metadata_report(plan)
            elif args.mode == "install":
                if args.execute and args.dry_run:
                    raise ValueError("--execute 和 --dry-run 不能同时使用")
                report = execute_install(plan, execute=args.execute, retry=args.retry)
            else:
                report = plan
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"error": str(error), "installed": False}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
