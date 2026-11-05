"""Automatic batches preserve existing data and count attempts, not successes."""

import json
from unittest.mock import patch

import pytest

from examples.euler.medication_sorting import collect_dataset as c
from tests.test_medication_collection import make_plan, successful_run


def test_random_plan_appends_and_freezes_successful_reference(tmp_path):
    reference = tmp_path / "raw/train/episode_0000"
    successful_run({}, {}, reference)
    poses = {
        key: {"position": [0, 0, 1], "quaternion": [1, 0, 0, 0]}
        for key in ("calcium", "melatonin", "herbbox", "pharyngitis")
    }
    (reference / "initial_layout.json").write_text(json.dumps({"poses": poses}))
    (tmp_path / "raw/train/episode_0002").mkdir()
    path = c.create_random_plan(tmp_path, reference, 15, 100, 0.003, 1)
    plan = json.loads(path.read_text())
    assert [r["episode_id"] for r in plan["episodes"]] == [f"episode_{i:04d}" for i in range(3, 18)]
    assert len({r["seed"] for r in plan["episodes"]}) == 15
    assert json.loads((tmp_path / plan["reference_layout"]).read_text())["poses"] == poses
    second = c.create_random_plan(tmp_path, reference, 1, 200, 0.003, 1)
    assert json.loads(second.read_text())["episodes"][0]["episode_id"] == "episode_0018"


def test_failed_reference_rejected(tmp_path):
    with pytest.raises(ValueError, match="successful"):
        c.create_random_plan(tmp_path, tmp_path / "missing", 15, 100, 0.003, 1)


def test_motion_failure_counts_once_and_resume_does_not_retry(tmp_path):
    path = make_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan["continue_motion_failures"] = True
    plan["episodes"] = plan["episodes"][:3]
    path.write_text(json.dumps(plan))

    def run(plan, row, out, *args):
        if row["episode_id"] == "episode_0000":
            out.mkdir(parents=True)
            (out / "result.json").write_text(
                json.dumps(
                    {"success": False, "reason": "IKError: collision", "recording": {"errors": [], "raw_valid": False}}
                )
            )
            return 1
        return successful_run(plan, row, out)

    with patch.object(c, "run_episode", side_effect=run):
        status = c.collect(path, 10**9, 1, 1)
    assert status["status"] == "complete"
    assert len(status["completed"]) == 2
    assert len(status["failed_attempts"]) == 1
    with patch.object(c, "run_episode") as run_again:
        resumed = c.collect(path, 10**9, 1, 1)
    assert len(resumed["failed_attempts"]) == 1
    run_again.assert_not_called()


def test_camera_failure_is_not_retryable_even_with_ik_error():
    assert not c.motion_failure({"reason": "IKError: collision", "recording": {"errors": ["camera disconnected"]}})
    assert c.motion_failure({"reason": "IKError: collision", "recording": {"errors": []}})
    assert not c.motion_failure({"reason": "TimeoutError: missing RGB"})


def test_lost_grasp_is_motion_failure_but_camera_error_still_stops():
    report = {
        "reason": "RuntimeError: calcium: lost grasp during calcium_lower (relative drift 0.040 m)",
        "recording": {"errors": []},
    }
    assert c.motion_failure(report)
    report["recording"]["errors"] = ["append: missing frame"]
    assert not c.motion_failure(report)


def test_cleanup_keeps_failure_metadata_and_refuses_success(tmp_path):
    out = tmp_path / "episode_0004"
    successful_run({}, {}, out)
    video = out / "episode/videos/head.mkv"
    video.parent.mkdir()
    video.write_bytes(b"video")
    trace = out / "trace.json"
    trace.write_text("[1,2,3]")
    with pytest.raises(ValueError):
        c.discard_failed_recording(out)
    assert video.exists()
    (out / "result.json").write_text('{"success": false, "reason": "IKError: collision"}')
    meta = out / "episode/metadata.json"
    data = json.loads(meta.read_text())
    data["raw_valid"] = False
    meta.write_text(json.dumps(data))
    c.discard_failed_recording(out)
    assert not video.exists()
    assert not trace.exists()
    assert meta.exists() and (out / "result.json").exists()
    assert (out / "discarded_recording.json").exists()
    c.discard_failed_recording(out)  # idempotent when the saved plan is resumed


def test_missed_end_pose_is_failed_attempt_but_recording_errors_still_stop():
    report = {
        'success': False,
        'reason': 'RuntimeError: melatonin_transfer: actual end pose missed target (0.0147 m, 0.067 rad)',
        'recording': {'errors': [], 'raw_valid': False},
    }
    assert c.motion_failure(report)
    report['recording']['errors'] = ['missing camera frame']
    assert not c.motion_failure(report)
    report['recording']['errors'] = []
    report['recording_error'] = 'decoder interrupted'
    assert not c.motion_failure(report)
