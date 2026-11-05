"""Manual collection must preserve existing episodes and use the saved layout."""

import fcntl
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.euler.medication_sorting import main as manual
from examples.euler.medication_sorting.recording import gripper_mapping


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(manual, "HOST_WIDE_COLLECTION_LOCK", tmp_path / "host.lock")
    monkeypatch.setattr(manual.shutil, "disk_usage", lambda _: SimpleNamespace(free=10_000_000_000))
    return tmp_path / "dataset"


class FakeProcess:
    def __init__(self, returncode=None, timeouts=0):
        self.returncode = returncode
        self.timeouts = timeouts
        self.events = []

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.events.append(("signal", value))

    def terminate(self):
        self.events.append(("terminate",))

    def kill(self):
        self.events.append(("kill",))

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = -signal.SIGINT
        return self.returncode


def fake_run(monkeypatch, *, success=True, raw_valid=True, returncode=0):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        (output / "episode").mkdir(parents=True)
        (output / "result.json").write_text(json.dumps({"success": success}))
        (output / "episode/metadata.json").write_text(
            json.dumps(
                {
                    "raw_valid": raw_valid,
                    "frame_count": 10,
                    "format": "medication_sorting_raw_v2",
                    "action_dim": 16,
                    "gripper_mapping": gripper_mapping(),
                }
            )
        )
        return FakeProcess(returncode=returncode)

    monkeypatch.setattr(manual.subprocess, "Popen", run)
    return calls


def test_current_layout_and_next_id_preserve_failed_episode(setup, monkeypatch, capsys):
    failed = setup / "raw/train/episode_0000"
    failed.mkdir(parents=True)
    (failed / "failure.txt").write_text("keep")
    calls = fake_run(monkeypatch)
    output = manual.collect_one(setup, endpoint="server:50051")
    assert output == setup / "raw/train/episode_0001"
    assert (failed / "failure.txt").read_text() == "keep"
    command, options = calls[0]
    for flag, value in (
        ("--layout-mode", "current"),
        ("--xy-jitter", "0"),
        ("--yaw-jitter-deg", "0"),
        ("--object", "all"),
        ("--endpoint", "server:50051"),
        ("--task-index", "1"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--execute" in command and "--record" in command
    assert "stdout" not in options and "stderr" not in options
    assert "保存" in capsys.readouterr().out


def test_validation_split_and_explicit_task_index(setup, monkeypatch):
    (setup / "raw/train/episode_0002").mkdir(parents=True)
    calls = fake_run(monkeypatch)
    output = manual.collect_one(setup, split="val", task_index=2)
    assert output == setup / "raw/val/episode_0003"
    command = calls[0][0]
    assert command[command.index("--task-index") + 1] == "2"


@pytest.mark.parametrize("success,raw_valid,code", [(False, True, 0), (True, False, 0), (True, True, 1)])
def test_motion_and_recording_both_must_pass(setup, monkeypatch, success, raw_valid, code):
    fake_run(monkeypatch, success=success, raw_valid=raw_valid, returncode=code)
    with pytest.raises(RuntimeError, match="失败"):
        manual.collect_one(setup)
    assert (setup / "raw/train/episode_0000/episode/metadata.json").is_file()


def test_low_disk_does_not_launch(setup, monkeypatch):
    monkeypatch.setattr(manual.shutil, "disk_usage", lambda _: SimpleNamespace(free=6_000_000_000))
    calls = fake_run(monkeypatch)
    with pytest.raises(RuntimeError, match="空间"):
        manual.collect_one(setup)
    assert not calls
    assert not (setup / "raw/train/episode_0000").exists()


def test_shared_host_lock_prevents_concurrent_collection(setup, monkeypatch):
    calls = fake_run(monkeypatch)
    with manual.HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            manual.collect_one(setup)
    assert not calls


def test_execute_is_required(setup, monkeypatch):
    monkeypatch.setattr(manual.sys, "argv", ["collect_manual", "--dataset-root", str(setup)])
    with pytest.raises(SystemExit) as exc:
        manual.main()
    assert exc.value.code == 2


def test_total_budget_preflight_does_not_launch(setup, monkeypatch):
    calls = fake_run(monkeypatch)
    monkeypatch.setattr(manual, "tree_bytes", lambda _: 24_000_000_000)
    with pytest.raises(RuntimeError, match="容量"):
        manual.collect_one(setup)
    assert not calls


@pytest.mark.parametrize("limit", ["disk", "total", "timeout"])
def test_runtime_limits_stop_and_reap_before_unlock(setup, monkeypatch, limit):
    process = FakeProcess()
    monkeypatch.setattr(manual.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(manual.time, "sleep", lambda _: None)
    frees = iter([10_000_000_000, 3_000_000_000 if limit == "disk" else 10_000_000_000])
    monkeypatch.setattr(manual.shutil, "disk_usage", lambda _: SimpleNamespace(free=next(frees)))
    sizes = iter([0, 25_000_000_000 if limit == "total" else 0])
    monkeypatch.setattr(manual, "tree_bytes", lambda _: next(sizes))
    times = iter([0, 3601 if limit == "timeout" else 5])
    monkeypatch.setattr(manual.time, "monotonic", lambda: next(times))
    original_wait = process.wait

    def wait(timeout=None):
        with manual.HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return original_wait(timeout)

    monkeypatch.setattr(process, "wait", wait)
    with pytest.raises(RuntimeError, match="采集中止"):
        manual.collect_one(setup)
    assert process.events == [("signal", signal.SIGINT), ("wait", 60)]
    assert process.poll() is not None


def test_interrupted_child_escalates_and_reaps(setup, monkeypatch):
    process = FakeProcess(timeouts=2)
    monkeypatch.setattr(manual.subprocess, "Popen", lambda *args, **kwargs: process)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(manual.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        manual.collect_one(setup)
    assert process.events == [
        ("signal", signal.SIGINT),
        ("wait", 60),
        ("terminate",),
        ("wait", 10),
        ("kill",),
        ("wait", None),
    ]
    assert process.poll() is not None


def test_native_collection_rejects_legacy_directory_before_launch(setup, monkeypatch):
    meta = setup / "raw/train/episode_0000/episode/metadata.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"format": "medication_sorting_raw_v1", "action_dim": 18}))
    calls = fake_run(monkeypatch)
    with pytest.raises(ValueError, match="native16"):
        manual.collect_one(setup)
    assert calls == []
