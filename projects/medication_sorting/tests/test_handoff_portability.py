"""交接副本的跨用户路径检查，不连接仿真、不使用GPU。"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tools.medication_sorting import rendered_rgb
from tools.medication_sorting.run_randomized_experiment import Experiment


ROOT = Path(__file__).resolve().parents[1]


def test_official_base_download_manifest_is_included_without_weights():
    from tools.medication_sorting.download_pi05 import DOWNLOAD_MANIFEST, PREFIX

    assert DOWNLOAD_MANIFEST.is_file(), "从头训练需要随代码交接官方基座下载清单"
    items = json.loads(DOWNLOAD_MANIFEST.read_text())["items"]
    assert items
    for item in items:
        assert item["bucket"] == "openpi-assets"
        assert item["name"].startswith(PREFIX)
        assert ".." not in Path(item["name"]).parts
        assert int(item["size"]) >= 0
        assert item.get("md5Hash") or item.get("crc32c")


@pytest.mark.parametrize("custom_directory", [False, True])
def test_rendered_capture_uses_current_user_or_explicit_directory(tmp_path, monkeypatch, custom_directory):
    home = tmp_path / "another-user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ORCALAB_CAPTURE_DIR", raising=False)
    expected = home / ".orcalab"
    if custom_directory:
        expected = tmp_path / "shared-captures"
        monkeypatch.setenv("ORCALAB_CAPTURE_DIR", str(expected))
    paths = []
    camera = object.__new__(rendered_rgb.RenderedRGB)
    camera.robot = SimpleNamespace(site_pose=lambda: {"xpos": np.zeros(3)}, sim_time=0.0)

    async def capture_slot(folder, slot, position, rotation, *, native_path, native_index):
        paths.append(native_path)
        return np.zeros((224, 224, 3), dtype=np.uint8), 1

    camera._capture_slot = capture_slot
    asyncio.run(camera._capture(tmp_path, 0))
    assert paths == [expected / "view_transform_300.png", expected / "view_transform_301.png"]


@pytest.mark.parametrize("explicit_root", [False, True])
def test_environment_script_resolves_its_project_and_respects_override(tmp_path, explicit_root):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"SY_ROOT", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"}}
    expected = ROOT
    if explicit_root:
        expected = tmp_path / "explicit-project"
        environment["SY_ROOT"] = str(expected)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$SY_ROOT"', "bash", str(ROOT / "tools/orca/environment.sh")],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(expected)


def test_long_experiment_keeps_callers_desktop_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":42")
    monkeypatch.setenv("XAUTHORITY", str(tmp_path / "session.Xauthority"))
    experiment = Experiment(tmp_path / "run")
    assert experiment.environment["DISPLAY"] == ":42"
    assert experiment.environment["XAUTHORITY"] == str(tmp_path / "session.Xauthority")


def test_long_experiment_does_not_invent_server_desktop_session(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("XAUTHORITY", raising=False)
    environment = Experiment(tmp_path / "run").environment
    assert "DISPLAY" not in environment
    assert "XAUTHORITY" not in environment


def test_launcher_works_without_original_servers_private_qt_library(tmp_path):
    project = tmp_path / "friends-project"
    for relative in ("tools/medication_sorting", "tools/orca", ".venv/bin", "software/OrcaPlayground"):
        (project / relative).mkdir(parents=True)
    for relative in ("tools/medication_sorting/start_orcalab.sh", "tools/orca/environment.sh"):
        shutil.copyfile(ROOT / relative, project / relative)
    python = project / ".venv/bin/python3"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    app = project / ".venv/bin/orcalab"
    app.write_text('#!/bin/sh\nprintf "APP_OK:%s:%s:%s\\n" "$PWD" "$DISPLAY" "${LD_PRELOAD:-}"\n')
    app.chmod(0o755)
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"SY_ROOT", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"}}
    environment.update(DISPLAY=":42", XAUTHORITY=str(tmp_path / "friends.Xauthority"))
    result = subprocess.run(["bash", str(project / "tools/medication_sorting/start_orcalab.sh")],
                            env=environment, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"APP_OK:{project}/software/OrcaPlayground::42:" in result.stdout.splitlines()
    assert "cannot be preloaded" not in result.stderr
