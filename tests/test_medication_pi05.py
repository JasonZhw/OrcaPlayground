"""Protect the dual-arm learning boundary against silent semantic changes."""

import copy
import importlib

import numpy as np
import pytest


def adapter():
    return importlib.import_module("examples.euler.medication_sorting.pi05.adapter")


def sample():
    actions = np.tile(np.arange(16, dtype=np.float32) / 20, (50, 1))
    return {
        "observation.state": np.arange(30, dtype=np.float32) / 10,
        "action": actions,
        "prompt": "Sort the medicine bottles and boxes into their bins.",
        "observation.images.head": np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8),
        "observation.images.left_wrist": np.full((480, 640, 3), [0, 255, 0], dtype=np.uint8),
        "observation.images.right_wrist": np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8),
    }


def test_three_camera_identity_and_absolute_actions_survive_adapter():
    raw = sample()
    got = adapter().MedicationInputs()(raw)
    np.testing.assert_array_equal(got["state"], raw["observation.state"])
    np.testing.assert_array_equal(got["actions"], raw["action"])
    for slot, color in [
        ("base_0_rgb", [255, 0, 0]),
        ("left_wrist_0_rgb", [0, 255, 0]),
        ("right_wrist_0_rgb", [0, 0, 255]),
    ]:
        assert got["image"][slot].shape == (480, 640, 3)
        np.testing.assert_array_equal(got["image"][slot][123, 234], color)
        assert got["image_mask"][slot]
    assert got["prompt"] == raw["prompt"]


def test_reader_float_chw_and_online_uint8_hwc_are_equivalent():
    raw = sample()
    raw["observation.images.head"][10, 20] = [17, 128, 231]
    online = adapter().MedicationInputs()(raw)
    reader = copy.deepcopy(raw)
    for key in [k for k in raw if k.startswith("observation.images.")]:
        reader[key] = np.moveaxis(raw[key].astype(np.float32) / 255, -1, 0)
    train = adapter().MedicationInputs()(reader)
    for key in online["image"]:
        np.testing.assert_array_equal(train["image"][key], online["image"][key])


@pytest.mark.parametrize(
    "key,value",
    [
        ("observation.state", np.zeros(15)),
        ("observation.state", np.full(30, np.nan)),
        ("action", np.zeros((50, 18))),
        ("action", np.zeros((49, 16))),
        ("action", np.full((50, 16), np.inf)),
        ("prompt", ""),
        ("prompt", None),
        ("observation.images.head", np.zeros((224, 224, 3), np.uint8)),
        ("observation.images.head", np.full((3, 480, 640), 1.1)),
        ("observation.images.head", np.full((3, 480, 640), np.nan)),
    ],
)
def test_invalid_inputs_are_rejected(key, value):
    raw = sample()
    raw[key] = value
    with pytest.raises(ValueError):
        adapter().MedicationInputs()(raw)


def test_online_observation_does_not_require_ground_truth_actions():
    raw = sample()
    del raw["action"]
    assert "actions" not in adapter().MedicationInputs()(raw)


def test_output_removes_padding_without_altering_absolute_joint_targets():
    raw = np.tile(np.arange(32, dtype=np.float32) / 20, (50, 1))
    got = adapter().MedicationOutputs()({"actions": raw})["actions"]
    assert got.shape == (50, 16)
    np.testing.assert_array_equal(got, raw[:, :16])


def dataset_info():
    state = [
        f"{side}_{kind}_{i}"
        for side in ["left", "right"]
        for kind, size in [("arm", 7), ("gripper", 8)]
        for i in range(size)
    ]
    action = [
        n for side in ["left", "right"] for n in [*[f"{side}_arm_{i}" for i in range(7)], f"{side}_gripper_closure"]
    ]
    features = {
        "observation.state": {"shape": [30], "dtype": "float32", "names": state},
        "action": {"shape": [16], "dtype": "float32", "names": action},
        **{
            f"observation.images.{cam}": {"shape": [480, 640, 3], "dtype": "video"}
            for cam in ["head", "left_wrist", "right_wrist"]
        },
    }
    return dict(codebase_version="v2.1", fps=50, total_episodes=20, total_frames=219594, features=features)


def validation():
    return dict(
        passed=True, total_episodes=20, total_frames=219594, state_dim=30, action_dim=16, fps=50, crop=None, resize=None
    )


def test_dataset_contract_accepts_current_verified_format():
    adapter().validate_info(dataset_info(), validation())


@pytest.mark.parametrize(
    "change", ["joint_swap", "wrong_fps", "missing_camera", "wrong_shape", "failed_validation", "wrong_frame_count"]
)
def test_dataset_contract_rejects_silent_misconfiguration(change):
    info, report = dataset_info(), validation()
    if change == "joint_swap":
        names = info["features"]["action"]["names"]
        names[0], names[8] = names[8], names[0]
    elif change == "wrong_fps":
        info["fps"] = 5
    elif change == "missing_camera":
        del info["features"]["observation.images.left_wrist"]
    elif change == "wrong_shape":
        info["features"]["observation.images.head"]["shape"] = [640, 480, 3]
    elif change == "failed_validation":
        report["passed"] = False
    else:
        report["total_frames"] -= 1
    with pytest.raises(ValueError):
        adapter().validate_info(info, report)


def test_numeric_windows_repeat_last_action_without_crossing_episode():
    module = importlib.import_module("examples.euler.medication_sorting.pi05.statistics")
    actions = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32)
    got = module.action_windows(actions, 0, 3, horizon=3)
    np.testing.assert_array_equal(
        got,
        [
            [[10, 11], [20, 21], [30, 31]],
            [[20, 21], [30, 31], [30, 31]],
            [[30, 31], [30, 31], [30, 31]],
        ],
    )
    np.testing.assert_array_equal(module.action_windows(actions, 2, 3, horizon=3), got[2:])


def test_resume_audit_preserves_first_launch(tmp_path):
    g = importlib.import_module("examples.euler.medication_sorting.pi05.guards")
    first = g.write_launch_audit(tmp_path, "smoke", False, {"step": 0})
    second = g.write_launch_audit(tmp_path, "smoke", True, {"step": 9})
    assert first != second
    import json

    assert json.loads(first.read_text()) == {"step": 0}
    assert json.loads(second.read_text()) == {"step": 9}


def test_source_guard_detects_code_edit_with_unchanged_lock(tmp_path):
    import hashlib
    import json

    g = importlib.import_module("examples.euler.medication_sorting.pi05.guards")
    root = tmp_path / "openpi"
    root.mkdir()
    code = root / "model.py"
    code.write_text("original")
    manifest = {
        "revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "files": {"model.py": hashlib.sha256(b"original").hexdigest()},
    }
    (tmp_path / "source_files.json").write_text(json.dumps(manifest))
    g.verify_source(root)
    code.write_text("changed")
    with pytest.raises(ValueError, match="source"):
        g.verify_source(root)


@pytest.mark.parametrize("mutation", ["stats", "data", "contract"])
def test_norm_guard_rejects_foreign_or_stale_stats(tmp_path, mutation):
    import hashlib
    import json

    g = importlib.import_module("examples.euler.medication_sorting.pi05.guards")
    data = tmp_path / "data"
    data.mkdir()
    source = data / "ep.parquet"
    source.write_bytes(b"verified_data")
    assets = tmp_path / "assets"
    assets.mkdir()
    stats = assets / "norm_stats.json"
    stats.write_bytes(b"own_stats")
    info = {"total_frames": 3, "total_episodes": 1}
    report = dict(
        frames=3,
        episodes=1,
        state_dim=30,
        action_dim=16,
        action_horizon=50,
        stats_sha256=hashlib.sha256(b"own_stats").hexdigest(),
        sources=[{"path": str(source), "sha256": hashlib.sha256(b"verified_data").hexdigest()}],
    )
    p = assets / "norm_provenance.json"
    p.write_text(json.dumps(report))
    g.verify_norm(assets, data, info)
    if mutation == "stats":
        stats.write_bytes(b"foreign_stats")
    elif mutation == "data":
        source.write_bytes(b"changed_data")
    else:
        report["action_dim"] = 9
        p.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        g.verify_norm(assets, data, info)
