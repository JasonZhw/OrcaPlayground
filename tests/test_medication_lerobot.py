"""Conversion preserves native views and pre-step rows, and rejects corrupt sources."""

import hashlib
import json
from copy import deepcopy
from fractions import Fraction

import av
import numpy as np
import pytest

from examples.euler.medication_sorting.recording import ACTION_NAMES, CAMERAS, STATE_NAMES, TASK_TEXTS, gripper_mapping


def test_known_expert_release_difference_preserves_control_contract(raw_root):
    from examples.euler.medication_sorting.lerobot_export import check_configuration

    old = json.loads((raw_root / "raw/train/episode_0000/episode/metadata.json").read_text())
    new = deepcopy(old)
    new["control"]["release_fallback"] = "checked_hover_if_lowering_unplannable"
    before = deepcopy(new)
    check_configuration(old, new)
    check_configuration(new, old)
    assert new == before


@pytest.mark.parametrize("damage", ["period", "physics", "unknown_release", "camera", "joint_order"])
def test_other_configuration_differences_still_rejected(raw_root, damage):
    from examples.euler.medication_sorting.lerobot_export import check_configuration

    old = json.loads((raw_root / "raw/train/episode_0000/episode/metadata.json").read_text())
    new = deepcopy(old)
    new["control"]["release_fallback"] = "checked_hover_if_lowering_unplannable"
    if damage == "period":
        new["control"]["period_seconds"] = 0.04
    elif damage == "physics":
        new["physics"]["impratio"] = 3
    elif damage == "unknown_release":
        new["control"]["release_fallback"] = "unchecked_release"
    elif damage == "camera":
        new["cameras"]["head_cam"]["properties"]["vertical_fov"] = 90
    else:
        new["actuator_joint_names"] = ["different_joint"]
    with pytest.raises(ValueError):
        check_configuration(old, new)


def test_expert_provenance_preserves_actual_events_and_missing_records(raw_root):
    from examples.euler.medication_sorting.lerobot_export import expert_provenance

    directory = raw_root / "raw/train/episode_0000/episode"
    metadata = json.loads((directory / "metadata.json").read_text())
    assert expert_provenance(directory, metadata)["release_fallbacks"]["recorded"] is False
    events = [{"object": "calcium", "reason": "IKError", "checked": True}]
    (directory.parent / "release_fallbacks.json").write_text(json.dumps(events))
    (directory.parent / "selection.json").write_text(json.dumps({"selected": "calcium"}))
    metadata["expert_trajectory"] = {"planner": "static_servo_headroom_v1"}
    report = expert_provenance(directory, metadata)
    assert report["release_fallbacks"]["value"] == events
    assert report["selection"]["value"] == {"selected": "calcium"}
    assert report["expert_trajectory"]["value"] == metadata["expert_trajectory"]
    assert report["control"] == metadata["control"]


@pytest.fixture
def raw_root(tmp_path):
    root = tmp_path / "raw_data"
    directory = root / "raw/train/episode_0000/episode"
    (directory / "videos").mkdir(parents=True)
    n = 4
    state = np.arange(n * 30, dtype=np.float64).reshape(n, 30) / 100
    action = np.arange(n * 16, dtype=np.float64).reshape(n, 16) / 200
    np.savez_compressed(
        directory / "episode.npz",
        state=state,
        action=action,
        sim_time=1.5 + np.arange(n) / 50,
        frame_index=np.arange(87, 87 + n),
        stage=np.array(["approach", "close", "lift", "release"]),
    )
    metadata = dict(
        format="medication_sorting_raw_v2",
        success=True,
        complete=True,
        raw_valid=True,
        errors=[],
        fps=50.0,
        frame_count=n,
        task=TASK_TEXTS[0],
        state_dim=30,
        action_dim=16,
        state_names=list(STATE_NAMES),
        action_names=list(ACTION_NAMES),
        gripper_mapping=gripper_mapping(),
        camera_order=list(CAMERAS),
        physics={"impratio": 2},
        control={"period_seconds": 0.02},
        cameras={},
        synchronization={"alignment": "pre-step observation_t paired with action_t", "latest_frame_fallback": False},
    )
    for c, camera in enumerate(CAMERAS):
        rows = []
        with av.open(str(directory / f"videos/{camera}.mkv"), "w") as container:
            stream = container.add_stream("ffv1", rate=50)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
            for i in range(n):
                rgb = np.zeros((24, 32, 3), np.uint8)
                rgb[:] = [40 + 40 * c, 30 + 30 * i, 120]
                rgb[:, 16:] += 10
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, 50)
                for packet in stream.encode(frame):
                    container.mux(packet)
                rows.append(
                    dict(
                        row_index=i,
                        frame_index=87 + i,
                        timestamp_ns=1000 + 100 * i,
                        sim_time=1.5 + i / 50,
                        is_keyframe=True,
                        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    )
                )
            for packet in stream.encode():
                container.mux(packet)
        sidecar = f"videos/{camera}.frames.json"
        (directory / sidecar).write_text(json.dumps(rows))
        metadata["cameras"][camera] = dict(
            path=f"videos/{camera}.mkv",
            frame_metadata=sidecar,
            decoded_frame_count=n,
            native_shape=[24, 32, 3],
            properties={"width": 32, "height": 24, "vertical_fov": 60},
        )
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory.parent / "result.json").write_text(json.dumps({"success": True}))
    return root


def test_native_frames_and_nonzero_render_indices_are_preserved(raw_root):
    from examples.euler.medication_sorting.lerobot_export import iter_frames, load_episode

    directory = raw_root / "raw/train/episode_0000/episode"
    metadata, arrays = load_episode(directory)
    frames = list(iter_frames(directory, metadata, arrays))
    assert len(frames) == 4
    assert frames[0]["head_cam"].shape == (24, 32, 3)
    np.testing.assert_array_equal(frames[2]["camera_right"][0, 0], [120, 90, 120])
    assert arrays["frame_index"].tolist() == [87, 88, 89, 90]


@pytest.mark.parametrize("damage", ["state_order", "time", "camera_index", "hash", "failed"])
def test_invalid_recording_cannot_enter_training(raw_root, damage):
    from examples.euler.medication_sorting.lerobot_export import iter_frames, load_episode

    directory = raw_root / "raw/train/episode_0000/episode"
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    if damage == "state_order":
        metadata["state_names"].reverse()
    elif damage == "failed":
        metadata["raw_valid"] = False
    elif damage == "time":
        with np.load(directory / "episode.npz") as source:
            arrays = dict(source)
        arrays["sim_time"][2] += 0.01
        np.savez_compressed(directory / "episode.npz", **arrays)
    else:
        sidecar = directory / metadata["cameras"]["head_cam"]["frame_metadata"]
        rows = json.loads(sidecar.read_text())
        rows[1]["frame_index" if damage == "camera_index" else "rgb_sha256"] = (
            999 if damage == "camera_index" else "bad"
        )
        sidecar.write_text(json.dumps(rows))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        metadata, arrays = load_episode(directory)
        list(iter_frames(directory, metadata, arrays))


def test_conversion_roundtrip_with_official_reader(raw_root, tmp_path):
    import shutil

    pytest.importorskip("lerobot.common.datasets.lerobot_dataset")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from examples.euler.medication_sorting.lerobot_export import export, validate

    second = raw_root / "raw/train/episode_0001"
    shutil.copytree(raw_root / "raw/train/episode_0000", second)
    path = second / "episode/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["control"]["release_fallback"] = "checked_hover_if_lowering_unplannable"
    path.write_text(json.dumps(metadata))
    (second / "release_fallbacks.json").write_text(json.dumps([{"object": "calcium", "checked": True}]))
    output = tmp_path / "g1_med_sort_v1"
    export(raw_root, output, "local/g1_med_sort_v1", "train")
    report = validate(output, raw_root)
    assert report["passed"] is True
    dataset = LeRobotDataset(
        "local/g1_med_sort_v1", root=output, video_backend="pyav", delta_timestamps={"action": [0.0, 0.02, 0.04]}
    )
    assert len(dataset) == 8
    expert = json.loads((output / "provenance/episode_000001_expert.json").read_text())
    assert expert["release_fallbacks"]["value"] == [{"object": "calcium", "checked": True}]
    assert tuple(dataset.meta.features["action"]["shape"]) == (16,)
    assert dataset.meta.features["action"]["names"][7] == "left_gripper_closure"
    row = dataset[2]
    np.testing.assert_array_equal(row["observation.state"].numpy(), (np.arange(60, 90) / 100).astype("float32"))
    np.testing.assert_array_equal(
        row["action"].numpy(),
        (np.array([np.arange(32, 48), np.arange(48, 64), np.arange(48, 64)]) / 200).astype("float32"),
    )
    assert row["task"] == TASK_TEXTS[0]
    assert tuple(row["observation.images.head"].shape) == (3, 24, 32)
    with pytest.raises(FileExistsError):
        export(raw_root, output, "local/g1_med_sort_v1", "train")


def test_train_export_skips_failed_and_does_not_include_validation(raw_root, tmp_path):
    import shutil

    from examples.euler.medication_sorting.lerobot_export import discover_episodes

    source = raw_root / "raw/train/episode_0000"
    failed = raw_root / "raw/train/episode_0001"
    shutil.copytree(source, failed)
    metadata_file = failed / "episode/metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["raw_valid"] = False
    metadata_file.write_text(json.dumps(metadata))
    shutil.copytree(source, raw_root / "raw/val/episode_0000")
    entries, skipped = discover_episodes(raw_root, "train")
    assert len(entries) == 1
    assert entries[0] == source / "episode"
    assert len(skipped) == 1


def test_validation_export_includes_existing_batch_folder(raw_root):
    import shutil

    from examples.euler.medication_sorting.lerobot_export import discover_episodes

    source = raw_root / "raw/train/episode_0000"
    shutil.copytree(source, raw_root / "raw/validation/episode_0004")
    shutil.copytree(source, raw_root / "raw/val/episode_0005")
    entries, skipped = discover_episodes(raw_root, "val")
    assert {entry.parent.name for entry in entries} == {"episode_0004", "episode_0005"}
    assert not skipped


def test_long_episode_float32_timestamps_accept_rounding_but_not_frame_shift():
    from examples.euler.medication_sorting.lerobot_export import check_table_timestamps

    timestamps = (np.arange(13000, dtype=np.float64) / 50).astype(np.float32)
    check_table_timestamps(timestamps, 13000)
    timestamps[12801] += 0.02
    with pytest.raises(AssertionError):
        check_table_timestamps(timestamps, 13000)


def test_legacy_actions_rejected_instead_of_converted(raw_root):
    from examples.euler.medication_sorting.lerobot_export import load_episode

    directory = raw_root / "raw/train/episode_0000/episode"
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(format="medication_sorting_raw_v1", action_dim=18)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_episode(directory)


def test_changed_gripper_contract_rejected(raw_root):
    from examples.euler.medication_sorting.lerobot_export import load_episode

    directory = raw_root / "raw/train/episode_0000/episode"
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["gripper_mapping"]["ad_knots"][-1][1] = 1.7
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mapping"):
        load_episode(directory)
