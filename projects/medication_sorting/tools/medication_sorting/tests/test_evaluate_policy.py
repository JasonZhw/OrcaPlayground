"""VLA 评估驱动的 mock 测试：只写临时假报告，不启动推理、模型或仿真。"""

import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    module = importlib.import_module("tools.medication_sorting.evaluate_policy")
    monkeypatch.setattr(module, "EXECUTION_LOCK", tmp_path / "evaluation.lock")
    monkeypatch.setattr(module, "active_controllers", lambda: [])
    return module


def argument(command, key):
    return command[command.index(key) + 1]


def fake_report(command, *, success=True, grasp=True, error=None):
    folder = Path(argument(command, "--output"))
    folder.mkdir(parents=True)
    report = {"success": success, "grasp_verified": grasp, "expert_actions_used": False,
              "controller": "pi05_checkpoint_policy", "test_only_synthetic_fixture": True}
    if error:
        report["error"] = error
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_evaluation_is_sequential_forwards_controls_and_preserves_failures(evaluator, tmp_path, monkeypatch):
    calls, events = [], []
    def run(command, **kwargs):
        seed = int(argument(command, "--seed"))
        events.append(("start", seed))
        assert events.count(("start", seed)) == 1
        calls.append((command, kwargs))
        failed = seed == 414
        fake_report(command, success=not failed, error="真实子进程错误的测试样例" if failed else None)
        events.append(("end", seed))
        return subprocess.CompletedProcess(command, int(failed), stdout=f"seed {seed}\n", stderr="failed\n" if failed else "")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    output = tmp_path / "evaluation"
    result = evaluator.evaluate_policy(output, checkpoint_reference="caller/checkpoint", episodes=3,
                                       seed_start=413, max_steps=19, execute_chunk=3, control_hz=10,
                                       host="localhost", port=9123, camera_backend="viewport")
    assert events == [(event, seed) for seed in (413, 414, 415) for event in ("start", "end")]
    assert result["success_count"] == 2
    assert result["failed_count"] == 1
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["grasp_verified_count"] == 3
    assert result["error_count"] == 1
    assert result["exit_code"] == 1
    assert result["checkpoint_reference"] == {"value": "caller/checkpoint", "source": "caller_supplied", "verified": False}
    assert result["heldout_status"] == "not_verified_against_training_manifest"
    for command, kwargs in calls:
        assert command[0] == str(evaluator.PYTHON)
        assert argument(command, "--steps") == "19"
        assert argument(command, "--execute-chunk") == "3"
        assert argument(command, "--control-hz") == "10"
        assert argument(command, "--host") == "localhost"
        assert argument(command, "--port") == "9123"
        assert argument(command, "--camera-backend") == "viewport"
        assert kwargs["cwd"] == evaluator.ROOT
        assert kwargs["shell"] is False
    assert json.loads((output / "seed_414/report.json").read_text())["error"] == "真实子进程错误的测试样例"
    assert (output / "logs/seed_414.stdout.txt").read_text() == "seed 414\n"
    assert (output / "logs/seed_414.stderr.txt").read_text() == "failed\n"
    assert json.loads((output / "summary.json").read_text()) == result


def test_missing_corrupt_and_contradictory_reports_are_task_failures(evaluator, tmp_path, monkeypatch):
    def run(command, **kwargs):
        seed = int(argument(command, "--seed"))
        folder = Path(argument(command, "--output"))
        folder.mkdir(parents=True)
        if seed == 414:
            (folder / "report.json").write_text("{broken")
        if seed == 415:
            (folder / "report.json").write_text(json.dumps({"success": True, "grasp_verified": False}))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="unverified", episodes=3)
    assert result["success_count"] == 0
    assert result["failed_count"] == result["error_count"] == 3
    assert result["success_rate"] == 0.0
    assert result["exit_code"] == 1
    assert (tmp_path / "eval/seed_414/report.json").read_text() == "{broken"


def test_launch_exception_is_preserved_and_next_seed_can_run(evaluator, tmp_path, monkeypatch):
    def run(command, **kwargs):
        if argument(command, "--seed") == "413":
            raise OSError("测试启动失败")
        fake_report(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="provided", episodes=2)
    assert result["failed_count"] == 1 and result["success_count"] == 1
    assert "测试启动失败" in result["episodes"][0]["error"]
    record = json.loads((tmp_path / "eval/seed_413/evaluation.json").read_text())
    assert "测试启动失败" in record["error"]
    assert record["returncode"] is None


def test_active_collector_blocks_without_launch_or_output(evaluator, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "active_controllers", lambda: [{"pid": 99, "command": "collect_episodes.py"}])
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kwargs: pytest.fail("禁止启动第二个控制进程"))
    output = tmp_path / "eval"
    with pytest.raises(RuntimeError, match="控制进程"):
        evaluator.evaluate_policy(output, checkpoint_reference="provided")
    assert not output.exists()


def test_shared_lock_blocks_another_evaluation(evaluator, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kwargs: pytest.fail("评估不能并发"))
    with evaluator.EXECUTION_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="评估"):
            evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="provided")


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"execute_chunk": 0}, {"episodes": 0}, {"control_hz": 6}, {"control_hz": 5.0}])
def test_invalid_controls_fail_before_launch(evaluator, tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kw: pytest.fail("无效配置不能启动"))
    with pytest.raises(ValueError):
        evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="provided", **kwargs)


def test_cli_returns_nonzero_for_any_task_failure(evaluator, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "evaluate_policy", lambda *args, **kwargs: {"exit_code": 1, "failed_count": 1})
    assert evaluator.main(["--output", str(tmp_path / "eval"), "--checkpoint-reference", "provided"]) == 1


def test_all_verified_tasks_return_zero_without_claiming_checkpoint_provenance(evaluator, tmp_path, monkeypatch):
    def run(command, **kwargs):
        fake_report(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="supplied", episodes=1)
    assert result["exit_code"] == result["failed_count"] == 0
    assert result["success_count"] == result["grasp_verified_count"] == 1
    assert result["success_rate"] == 1.0
    assert result["checkpoint_reference"]["verified"] is False


@pytest.mark.parametrize("nonfinite", ["NaN", "1e999"])
def test_nonfinite_report_is_preserved_rejected_and_does_not_stop_next_seed(evaluator, tmp_path, monkeypatch, nonfinite):
    raw = '{"success":true,"grasp_verified":true,"expert_actions_used":false,"controller":"pi05_checkpoint_policy","error":' + nonfinite + '}'
    def run(command, **kwargs):
        if argument(command, "--seed") == "413":
            folder = Path(argument(command, "--output"))
            folder.mkdir(parents=True)
            (folder / "report.json").write_text(raw)
        else:
            fake_report(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="supplied", episodes=2)
    assert result["failed_count"] == result["success_count"] == 1
    assert result["exit_code"] == 1
    assert (tmp_path / "eval/seed_413/report.json").read_text() == raw
    assert json.loads((tmp_path / "eval/seed_413/evaluation.json").read_text())["success"] is False


def test_new_controller_between_seeds_prevents_any_additional_launch(evaluator, tmp_path, monkeypatch):
    calls = []
    busy = []
    monkeypatch.setattr(evaluator, "active_controllers", lambda: busy)
    def run(command, **kwargs):
        calls.append(command)
        fake_report(command)
        busy.append({"pid": 123, "command": "collect_episodes.py"})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="supplied", episodes=3)
    assert len(calls) == result["success_count"] == 1
    assert result["task_count"] == 3 and result["failed_count"] == 2
    assert result["exit_code"] == 1
    for seed in (414, 415):
        record = json.loads((tmp_path / f"eval/seed_{seed}/evaluation.json").read_text())
        assert "控制进程" in record["error"] and record["returncode"] is None


def test_proc_detection_recognizes_same_project_controllers_without_signaling(evaluator, tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid, command, cwd in (
        (910001, [str(evaluator.PYTHON), "-m", "tools.medication_sorting.collect_episodes"], evaluator.ROOT),
        (910002, [str(evaluator.PYTHON), str(evaluator.ROOT / "tools/medication_sorting/infer_policy.py")], tmp_path),
        (910003, ["python", "run_expert.py"], tmp_path),
        (910004, [str(evaluator.PYTHON), "serve_policy.py"], evaluator.ROOT),
    ):
        folder = proc / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
        (folder / "cwd").symlink_to(cwd, target_is_directory=True)
    monkeypatch.undo()  # 本测试需要真实 active_controllers；/proc 仍为临时目录。
    monkeypatch.setattr(evaluator, "Path", lambda path: proc if str(path) == "/proc" else Path(path))
    assert {row["pid"] for row in evaluator.active_controllers()} == {910001, 910002}


def test_cli_defaults_to_five_short_5hz_tasks_starting_at_seed_413(evaluator, tmp_path, monkeypatch):
    calls = []
    def evaluate(output, **kwargs):
        calls.append(kwargs)
        return {"exit_code": 0}
    monkeypatch.setattr(evaluator, "evaluate_policy", evaluate)
    assert evaluator.main(["--output", str(tmp_path / "eval"), "--checkpoint-reference", "provided"]) == 0
    assert calls[0]["max_steps"] == 150
    assert calls[0]["episodes"] == 5
    assert calls[0]["control_hz"] == 5
    assert calls[0]["seed_start"] == 413


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.2", "ws://localhost", "wss://127.0.0.1", "ws://[::1]"])
def test_loopback_launch_uses_proxy_free_child_env_without_changing_parent(evaluator, tmp_path, monkeypatch, host):
    proxy_keys = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "WS_PROXY", "ws_proxy", "WSS_PROXY", "wss_proxy")
    for key in proxy_keys:
        monkeypatch.setenv(key, "socks://127.0.0.1:1080")
    monkeypatch.setenv("NO_PROXY", "upper.example")
    monkeypatch.setenv("no_proxy", "lower.example")
    monkeypatch.setenv("EVALUATION_TEST_VALUE", "preserved")
    parent_env = dict(os.environ)
    def run(command, **kwargs):
        child_env = kwargs.get("env", {})
        assert not any(key in child_env for key in proxy_keys)
        assert child_env["EVALUATION_TEST_VALUE"] == "preserved"
        assert child_env["NO_PROXY"] == child_env["no_proxy"]
        assert {"upper.example", "lower.example", "127.0.0.1", "localhost", "::1"}.issubset(set(child_env["NO_PROXY"].split(",")))
        assert dict(os.environ) == parent_env
        fake_report(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="supplied", episodes=1, host=host)
    assert result["success_count"] == 1
    assert result["controls"]["proxy_policy"] == "loopback_child_proxy_excluded"
    assert dict(os.environ) == parent_env


def test_remote_policy_host_keeps_existing_proxy_configuration(evaluator, tmp_path, monkeypatch):
    monkeypatch.setenv("all_proxy", "socks5h://proxy.invalid:1080")
    parent_env = dict(os.environ)
    def run(command, **kwargs):
        assert kwargs.get("env") == parent_env
        fake_report(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "eval", checkpoint_reference="supplied", episodes=1, host="policy.example")
    assert result["success_count"] == 1
    assert result["controls"]["proxy_policy"] == "inherited_for_non_loopback"


def test_child_environment_handles_upper_and_lower_proxy_keys_together(evaluator, monkeypatch):
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "WS_PROXY", "ws_proxy", "WSS_PROXY", "wss_proxy"):
        monkeypatch.setenv(key, "socks://proxy.invalid:1080")
    monkeypatch.setenv("NO_PROXY", "existing.example")
    monkeypatch.setenv("no_proxy", "other.example")
    child, policy = evaluator._child_environment("127.0.0.1:8000")
    assert policy == "loopback_child_proxy_excluded"
    assert not any(key in child for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "WS_PROXY", "ws_proxy", "WSS_PROXY", "wss_proxy"))
    assert child["NO_PROXY"] == child["no_proxy"]
    assert "existing.example" in child["NO_PROXY"] and "other.example" in child["NO_PROXY"]


def heldout_manifest(tmp_path):
    manifest = {"status": "materialized", "train": [], "validation": []}
    for partition, seed in (("train", 202610131), ("validation", 202610130)):
        source = tmp_path / f"raw_{seed}"
        source.mkdir()
        rng = np.random.default_rng(seed)
        offset = rng.uniform([-.006, -.008], [.006, .008]).tolist()
        yaw = float(rng.uniform(-.025, .025))
        (source / "metadata.json").write_text(json.dumps({"seed": seed, "success": True, "complete": True, "randomized_reset": True}))
        (source / "report.json").write_text(json.dumps({"success": True, "reset_offset": offset, "reset_yaw": yaw}))
        (source / "trace.jsonl").write_text(json.dumps({"time": .2, "phase": "settle_initial", "box_pos": [1, 2, 3], "box_quat": [1, 0, 0, 0]}) + "\n")
        link = tmp_path / partition / "episode_000000"
        link.parent.mkdir()
        link.symlink_to(source, target_is_directory=True)
        manifest[partition].append({"source": str(source), "link": str(link), "content_sha256": hashlib.sha256(str(seed).encode()).hexdigest()})
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_offline_seed_audit_retains_observed_source_pose_and_labels_eval_reset_expected(evaluator, tmp_path):
    manifest = heldout_manifest(tmp_path)
    before = manifest.read_bytes()
    result = evaluator.audit_heldout_seeds(manifest, range(413, 418))
    assert result["seed_disjoint_from_train"] and result["seed_disjoint_from_validation"]
    assert result["manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["train_seeds"] == [202610131]
    assert result["validation_seeds"] == [202610130]
    assert result["checkpoint_training_provenance_verified"] is False
    assert result["evaluation_initial_poses_observed"] is False
    assert result["source_episodes"][0]["first_trace_observation"]["box_pos"] == [1, 2, 3]
    assert result["source_episodes"][0]["first_trace_observation"]["time"] == .2
    assert result["source_episodes"][0]["reset_matches_seed"] is True
    assert result["evaluation_expected_resets"][0]["seed"] == 413
    assert result["evaluation_expected_resets"][0]["observed"] is False
    assert result["evaluation_expected_resets"][0]["offset"] == pytest.approx([-.0054453542643813505, -.0002198706696706406])
    assert manifest.read_bytes() == before


def test_offline_seed_audit_reports_collision_without_claiming_heldout(evaluator, tmp_path):
    result = evaluator.audit_heldout_seeds(heldout_manifest(tmp_path), [202610131])
    assert result["seed_disjoint_from_train"] is False
    assert result["seed_disjoint_from_validation"] is True
    assert result["overlap_train_seeds"] == [202610131]


def test_seed_audit_rejects_source_report_reset_that_does_not_match_saved_seed(evaluator, tmp_path):
    manifest = heldout_manifest(tmp_path)
    source = Path(json.loads(manifest.read_text())["train"][0]["source"])
    report = json.loads((source / "report.json").read_text())
    report["reset_offset"] = [0., 0.]
    (source / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="reset"):
        evaluator.audit_heldout_seeds(manifest, [413])


def test_audit_only_cli_never_launches_or_creates_evaluation_output(evaluator, tmp_path, monkeypatch, capsys):
    manifest = heldout_manifest(tmp_path)
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kwargs: pytest.fail("审计模式不得启动推理"))
    monkeypatch.setattr(evaluator, "_execution_lock", lambda: pytest.fail("审计模式不得占用控制锁"))
    assert evaluator.main(["--audit-only", "--training-manifest", str(manifest)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["seed_disjoint_from_train"] and report["read_only"]
    assert not evaluator.EXECUTION_LOCK.exists()


def test_manifest_seed_collision_blocks_actual_launch_before_output(evaluator, tmp_path, monkeypatch):
    manifest = heldout_manifest(tmp_path)
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kwargs: pytest.fail("重叠 seed 禁止启动"))
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="seed"):
        evaluator.evaluate_policy(output, checkpoint_reference="supplied", episodes=1,
                                  seed_start=202610131, training_manifest=manifest)
    assert not output.exists()


def test_manifest_hash_leakage_is_rejected_by_offline_audit(evaluator, tmp_path):
    path = heldout_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["validation"][0]["content_sha256"] = manifest["train"][0]["content_sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        evaluator.audit_heldout_seeds(path, [413])


def test_supplied_manifest_and_observed_eval_reset_do_not_claim_checkpoint_provenance(evaluator, tmp_path, monkeypatch):
    manifest = heldout_manifest(tmp_path)
    def run(command, **kwargs):
        fake_report(command)
        path = Path(argument(command, "--output")) / "report.json"
        report = json.loads(path.read_text())
        rng = np.random.default_rng(413)
        report.update(seed=413, reset_offset=rng.uniform([-.006,-.008],[.006,.008]).tolist(), reset_yaw=float(rng.uniform(-.025,.025)))
        path.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    result = evaluator.evaluate_policy(tmp_path / "evaluation", checkpoint_reference="supplied", episodes=1, training_manifest=manifest)
    assert result["success_count"] == 1
    assert result["heldout_status"] == "seed_disjoint_from_supplied_manifest_only"
    assert result["heldout_audit"]["checkpoint_training_provenance_verified"] is False
    assert result["episodes"][0]["reported_reset_matches_seed"] is True
