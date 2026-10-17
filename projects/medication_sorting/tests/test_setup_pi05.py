"""依赖层入口的离线测试：只构造命令与环境，绝不安装包或初始化 GPU。"""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "tools/medication_sorting/pi05_python.sh"


def test_setup_entrypoint_exists_for_safe_install_contract():
    assert importlib.util.find_spec("tools.medication_sorting.setup_pi05") is not None, "缺少只写 target 的安装入口"


def test_wrapper_exists_for_same_interpreter_contract():
    assert WRAPPER.is_file(), "缺少同一 .venv/bin/python 的进程隔离 wrapper"


@pytest.fixture(scope="module")
def setup():
    return importlib.import_module("tools.medication_sorting.setup_pi05")


def test_plan_pins_jax_cuda129_cudnn_and_keeps_torch(setup):
    plan = setup.build_install_plan()
    pins = plan["registry_pins"]
    assert pins["jax"] == pins["jaxlib"] == "0.5.3"
    assert pins["jax-cuda12-plugin"] == pins["jax-cuda12-pjrt"] == "0.5.3"
    assert pins["nvidia-cuda-runtime-cu12"] == "12.9.79"
    assert pins["nvidia-cuda-nvcc-cu12"] == "12.9.86"
    assert pins["nvidia-cudnn-cu12"] == "9.10.2.21"
    assert pins["numpy"] == "1.26.4"
    assert pins["transformers"] == "4.53.2"
    assert pins["torchvision"] == "0.29.1"
    assert "torch" not in pins
    assert plan["official_environment"] is False
    assert plan["end_to_end_required"] is True


def test_every_install_step_targets_layer_without_dependency_resolution(setup):
    plan = setup.build_install_plan()
    for step in plan["steps"]:
        command = step["argv"]
        assert command[:4] == [str(ROOT / ".venv/bin/python"), "-m", "pip", "install"]
        assert command[command.index("--target") + 1] == str(ROOT / ".venv/pi05-deps")
        assert "--no-deps" in command
        assert "--user" not in command
        assert not any(value.startswith("torch==") or "jax[cuda12]" in value for value in command)
    assert "--only-binary=:all:" in plan["steps"][0]["argv"]
    lerobot_step = plan["steps"][1]["argv"]
    assert any("lerobot" in value and setup.LEROBOT_COMMIT in value for value in lerobot_step)


@pytest.mark.parametrize("name", ["cmake", "pynput", "rerun-sdk", "pymunk", "zarr", "diffusers", "torchcodec", "gym-aloha", "gymnasium", "tensorflow", "dlimp", "h5py", "pyrealsense2", "dynamixel-sdk"])
def test_hardware_and_rlds_dependencies_not_installed(setup, name):
    assert name not in setup.build_install_plan()["registry_pins"]


def test_lock_closure_contains_required_transitive_dependencies(setup):
    pins = setup.build_install_plan()["registry_pins"]
    for name in ("pydantic-core", "optax", "tensorstore", "ml-dtypes", "draccus", "jsonlines", "av",
                 "datasets", "pyarrow", "tokenizers", "safetensors", "gcsfs", "etils", "protobuf", "msgpack", "websockets"):
        assert name in pins
    assert pins["ml-dtypes"] == "0.4.1"
    assert pins["tensorstore"] == "0.1.74"
    assert pins["dm-tree"] == "0.1.9"  # 官方 lock 的 Python 3.12 分支。


def test_default_dry_run_has_no_runner_or_filesystem_side_effect(setup, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run 不能执行安装/下载命令")
    monkeypatch.setattr(setup.subprocess, "run", forbidden)
    plan = setup.build_install_plan(project_root=tmp_path, check_checkout=False)
    assert setup.execute_install(plan, execute=False) == plan
    assert not (tmp_path / ".venv").exists()


def test_runtime_environment_only_changes_child_process(setup):
    parent = {"PYTHONPATH": "/tmp/ros-injected", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
              "LD_PRELOAD": "/tmp/injected.so", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin", "KEEP": "yes"}
    original = parent.copy()
    child = setup.runtime_environment(parent)
    assert parent == original
    paths = child["PYTHONPATH"].split(os.pathsep)
    assert paths[0] == str(ROOT / ".venv/pi05-deps")
    assert str(ROOT) in paths
    assert str(ROOT / "software/openpi/src") in paths
    assert str(ROOT / "software/openpi/packages/openpi-client/src") in paths
    assert "/tmp/ros-injected" not in paths
    assert "LD_LIBRARY_PATH" not in child
    assert "LD_PRELOAD" not in child
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert child["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.75"
    assert child["CUDA_VISIBLE_DEVICES"] == "0"
    assert child["KEEP"] == "yes"


def test_cpu_runtime_hides_gpu(setup):
    child = setup.runtime_environment({"CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORMS": "cuda"}, cpu=True)
    assert child["CUDA_VISIBLE_DEVICES"] == ""
    assert child["JAX_PLATFORMS"] == "cpu"


def test_runtime_selects_torch_rpath_instead_of_overlay_cuda_preload(setup):
    parent = {"TORCH_USE_RTLD_GLOBAL": "0"}
    child = setup.runtime_environment(parent, cpu=True)
    assert child["TORCH_USE_RTLD_GLOBAL"] == "1"
    assert parent == {"TORCH_USE_RTLD_GLOBAL": "0"}


def test_installed_wrapper_can_import_retained_torch_and_overlay_numpy():
    manifest = ROOT / ".venv/pi05-deps/pi05-layer-manifest.json"
    if not manifest.is_file() or not json.loads(manifest.read_text()).get("installed"):
        pytest.skip("实际训练依赖层尚未安装；离线测试不启动安装")
    # 防止外部环境中的临时修复使回归测试误通过。
    environment = dict(os.environ)
    environment.pop("TORCH_USE_RTLD_GLOBAL", None)
    script = "import torch,numpy as np; assert torch.__version__ == '2.14.1+cu130'; assert np.__version__ == '1.26.4'; np.testing.assert_array_equal(torch.from_numpy(np.arange(8,dtype=np.float32)).numpy(),np.arange(8,dtype=np.float32))"
    result = subprocess.run(["bash", str(WRAPPER), "--cpu", "-c", script],
                            cwd=ROOT, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["pip", "venv", "ensurepip"])
def test_runtime_cannot_accidentally_install_into_base_environment(setup, module):
    with pytest.raises(ValueError, match="安装|环境"):
        setup.runtime_command(["-m", module])


def test_runtime_command_preserves_arguments_as_data(setup):
    args = ["-c", "print('中文; $(touch /tmp/unwanted); `bad`')", "a b"]
    command = setup.runtime_command(args)
    assert command == [str(ROOT / ".venv/bin/python"), "-s", *args]


def test_nonempty_target_never_overwritten_by_install(setup, tmp_path):
    plan = setup.build_install_plan(project_root=tmp_path, check_checkout=False)
    target = Path(plan["target"])
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("保留")
    with pytest.raises(FileExistsError):
        setup.execute_install(plan, execute=True)
    assert (target / "keep.txt").read_text() == "保留"


def test_target_symlink_to_base_site_packages_rejected(setup, tmp_path):
    plan = setup.build_install_plan(project_root=tmp_path, check_checkout=False)
    target = Path(plan["target"])
    base = tmp_path / ".venv/lib/python3.12/site-packages"
    base.mkdir(parents=True)
    target.symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError, match="target|符号链接"):
        setup.execute_install(plan, execute=True)


def test_partial_layer_cannot_be_used_as_ready_environment(setup, tmp_path):
    target = tmp_path / ".venv/pi05-deps"
    target.mkdir(parents=True)
    (target / "numpy").mkdir()
    with pytest.raises(RuntimeError, match="安装|manifest"):
        setup.check_layer(target)


def test_layer_cannot_shadow_protected_torch(setup, tmp_path):
    target = tmp_path / ".venv/pi05-deps"
    target.mkdir(parents=True)
    (target / "torch").mkdir()
    with pytest.raises(RuntimeError, match="torch"):
        setup.check_layer(target)


def test_wrapper_dry_run_from_other_cwd_no_layer_no_gpu(tmp_path):
    result = subprocess.run(["bash", str(WRAPPER), "--dry-run", "-c", "print('a b')"],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["argv"] == [str(ROOT / ".venv/bin/python"), "-s", "-c", "print('a b')"]
    assert report["environment"]["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / ".venv/pi05-deps")
    assert report["executed"] is False
    assert report["gpu_touched"] is False


def test_setup_cli_dry_run_defaults_offline(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/medication_sorting/setup_pi05.py"), "--dry-run"],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["executed"] is False
    assert report["target"] == str(ROOT / ".venv/pi05-deps")
    assert report["network_accessed"] is False


def test_setup_import_does_not_import_training_backends():
    script = "import sys; import tools.medication_sorting.setup_pi05; assert not {'torch','numpy','jax','lerobot','transformers','openpi'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("argument", ["-mpip", "-mvenv", "-mensurepip", "-I", "-E", "-S", "-sE"])
def test_runtime_rejects_flags_that_bypass_layer_or_install(setup, argument):
    with pytest.raises(ValueError):
        setup.runtime_command([argument])


def test_script_arguments_do_not_get_mistaken_for_python_flags(setup):
    command = setup.runtime_command(["train.py", "-I", "-m", "pip"])
    assert command[-4:] == ["train.py", "-I", "-m", "pip"]


@pytest.fixture
def isolated_install(setup, tmp_path, monkeypatch):
    plan = setup.build_install_plan(project_root=tmp_path, check_checkout=False)
    python = Path(plan["python"])
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    snapshot = {"torch": {"version": "2.14.1", "metadata_sha256": "protected"}}
    monkeypatch.setattr(setup, "_protected_snapshot", lambda: snapshot.copy())
    return plan, snapshot


def test_failed_install_records_snapshot_and_retry_heals_partial(setup, isolated_install, monkeypatch):
    plan, snapshot = isolated_install
    target = Path(plan["target"])
    commands = []
    def interrupted(argv, **kwargs):
        commands.append(argv)
        target.mkdir(exist_ok=True)
        (target / "numpy").mkdir(exist_ok=True)
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(setup.subprocess, "run", interrupted)
    with pytest.raises(subprocess.CalledProcessError):
        setup.execute_install(plan, execute=True)
    manifest = json.loads((target / setup.MANIFEST).read_text())
    assert manifest["installed"] is False
    assert manifest["protected_before"] == snapshot
    with pytest.raises(RuntimeError, match="未完成"):
        setup.check_layer(target)
    def completed(argv, **kwargs):
        commands.append(argv)
        assert "--upgrade" in argv  # --target 默认会跳过已存在的包目录。
        assert "--no-deps" in argv
        assert "PYTHONPATH" not in kwargs["env"]
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(setup.subprocess, "run", completed)
    report = setup.execute_install(plan, execute=True, retry=True)
    assert report["installed"] is True
    assert report["protected_before"] == snapshot
    assert len(commands) == 3
    assert setup.check_layer(target)["installed"] is True


def test_partial_retry_refuses_changed_protected_packages(setup, isolated_install, monkeypatch):
    plan, snapshot = isolated_install
    target = Path(plan["target"])
    def failed(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(setup.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        setup.execute_install(plan, execute=True)
    saved = (target / setup.MANIFEST).read_bytes()
    snapshot["torch"] = {"version": "different", "metadata_sha256": "changed"}
    def forbidden(*args, **kwargs):
        raise AssertionError("快照不匹配时不能运行 pip")
    monkeypatch.setattr(setup.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="受保护|快照"):
        setup.execute_install(plan, execute=True, retry=True)
    assert (target / setup.MANIFEST).read_bytes() == saved


def test_install_detects_protected_package_change_even_on_pip_failure(setup, isolated_install, monkeypatch):
    plan, snapshot = isolated_install
    def mutated(argv, **kwargs):
        snapshot["torch"] = {"version": "changed"}
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(setup.subprocess, "run", mutated)
    with pytest.raises(RuntimeError, match="受保护"):
        setup.execute_install(plan, execute=True)
    assert json.loads((Path(plan["target"]) / setup.MANIFEST).read_text())["installed"] is False


def test_partial_retry_refuses_different_pins(setup, isolated_install, monkeypatch):
    plan, _ = isolated_install
    def failed(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(setup.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        setup.execute_install(plan, execute=True)
    changed = {**plan, "registry_pins": {**plan["registry_pins"], "jax": "0.6.0"}}
    with pytest.raises(RuntimeError, match="计划|版本"):
        setup.execute_install(changed, execute=True, retry=True)


def test_snapshot_reads_base_metadata_when_overlay_is_on_sys_path(setup, tmp_path, monkeypatch):
    base = tmp_path / ".venv/lib/python3.12/site-packages"
    overlay = tmp_path / ".venv/pi05-deps"
    for root, version in ((base, "2.2.6"), (overlay, "1.26.4")):
        info = root / f"numpy-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Name: numpy\nVersion: {version}\n")
    monkeypatch.setattr(setup, "PROJECT_ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(overlay))
    assert setup._protected_snapshot()["numpy"]["version"] == "2.2.6"


def test_wrapt_pin_has_python312_wheel_instead_of_legacy_source_build(setup):
    assert setup.build_install_plan()["registry_pins"]["wrapt"] == "1.17.3"


def test_metadata_audit_checks_cuda_extra_bounds_and_retained_torch(setup):
    packages = [
        {"package": "jax-cuda12-plugin", "requires_dist": [
            'nvidia-cudnn-cu12>=9.1,<10; extra == "with_cuda"',
            'nvidia-cuda-nvcc-cu12>=12.6.85; extra == "with_cuda"']},
        {"package": "torchvision", "requires_dist": ["torch>=2.14.0", "numpy", 'pytest; extra == "test"']},
    ]
    pins = {"nvidia-cudnn-cu12": "9.10.2.21", "nvidia-cuda-nvcc-cu12": "12.9.86", "numpy": "1.26.4"}
    assert setup.dependency_conflicts(packages, pins, torch_version="2.14.1") == []
    conflicts = setup.dependency_conflicts(packages, {**pins, "nvidia-cudnn-cu12": "10.0.0"}, torch_version="2.7.1")
    assert {item["package"] for item in conflicts} == {"nvidia-cudnn-cu12", "torch"}
