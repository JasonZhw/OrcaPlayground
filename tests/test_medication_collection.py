"""Capacity limits, split isolation and resume must not corrupt collection."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples.euler.medication_sorting import collect_dataset as collection
from examples.euler.medication_sorting.recording import gripper_mapping


def make_plan(tmp_path):
    rows = [
        {
            "episode_id": f"episode_{i:04d}",
            "split": "train" if i < 8 else "validation",
            "seed": i + 100,
            "task_index": i % 3,
            "status": "planned",
        }
        for i in range(10)
    ]
    p = tmp_path / "collection_plan.json"
    p.write_text(
        json.dumps(
            {"episodes": rows, "layout_mode": "baseline", "initial_xy_jitter_m": 0.005, "initial_yaw_jitter_deg": 3}
        )
    )
    return p


def successful_run(plan, row, output, *args):
    output.mkdir(parents=True)
    (output / "episode").mkdir()
    (output / "result.json").write_text(json.dumps({"success": True}))
    (output / "episode/metadata.json").write_text(
        json.dumps(
            {
                "raw_valid": True,
                "frame_count": 100,
                "format": "medication_sorting_raw_v2",
                "action_dim": 16,
                "gripper_mapping": gripper_mapping(),
            }
        )
    )
    return 0


def test_stops_before_starting_episode_that_would_use_reserved_space(tmp_path):
    p = make_plan(tmp_path)
    with (
        patch.object(collection.shutil, "disk_usage", return_value=SimpleNamespace(free=5000)),
        patch.object(collection, "run_episode") as run,
    ):
        status = collection.collect(p, 100000, 4000, 2000)
    assert status["status"] == "capacity_reached"
    assert status["completed"] == []
    run.assert_not_called()


def test_interleaves_validation_and_resume_does_not_execute_completed_episodes(tmp_path):
    p = make_plan(tmp_path)
    with patch.object(collection, "run_episode", side_effect=successful_run) as run:
        status = collection.collect(p, 10**9, 1, 1)
    assert status["status"] == "complete"
    assert [x.args[1]["split"] for x in run.call_args_list] == ["train"] * 4 + ["validation"] + ["train"] * 4 + [
        "validation"
    ]
    with patch.object(collection, "run_episode") as run:
        resumed = collection.collect(p, 10**9, 1, 1)
    assert len(resumed["completed"]) == 10
    run.assert_not_called()


def test_corrupt_recording_stops_batch_and_is_not_marked_complete(tmp_path):
    p = make_plan(tmp_path)

    def invalid_run(plan, row, output, *args):
        successful_run(plan, row, output)
        (output / "episode/metadata.json").write_text('{"raw_valid": false}')
        return 0

    with patch.object(collection, "run_episode", side_effect=invalid_run) as run:
        status = collection.collect(p, 10**9, 1, 1)
    assert status["status"] == "failed"
    assert status["completed"] == []
    assert run.call_count == 1


def test_shared_seed_between_splits_rejected_before_execution(tmp_path):
    p = make_plan(tmp_path)
    plan = json.loads(p.read_text())
    plan["episodes"][-1]["seed"] = plan["episodes"][0]["seed"]
    p.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="seed"), patch.object(collection, "run_episode") as run:
        collection.collect(p, 10**9, 1, 1)
    run.assert_not_called()
