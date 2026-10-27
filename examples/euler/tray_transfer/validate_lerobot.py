"""Read the exported dataset through LeRobot and compare against the source recordings."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .episode_recording import CAMERAS
from .lerobot_export import load_episode, write_json


def validate(root, source_root):
    import av
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root, source_root = Path(root), Path(source_root)
    provenance = json.loads((root / "provenance/export.json").read_text())
    expected_paths = [entry["path"] for entry in provenance["manifest"]["episodes"]]
    actual_paths = [entry["path"] for entry in provenance["episodes"]]
    if not expected_paths or actual_paths != expected_paths:
        raise ValueError("Incomplete export: exported episodes do not match the full manifest")
    dataset = LeRobotDataset(provenance["repo_id"], root=root, video_backend="pyav",
                            delta_timestamps={"action": [i / 50 for i in range(50)]})
    report = {"passed": False, "reader": "LeRobotDataset v2.1 + pyav", "fps": dataset.fps,
              "episodes": [], "model_training_run": False, "independent_evaluation_split": False}
    assert dataset.fps == 50
    offset = 0
    for entry in provenance["episodes"]:
        ep = entry["episode_index"]
        metadata, source = load_episode(source_root / entry["path"])
        count = metadata["frame_count"]
        table = pq.read_table(root / dataset.meta.get_data_file_path(ep)).to_pydict()
        for key, native in (("observation.state", "state"), ("action", "action")):
            np.testing.assert_array_equal(np.array(table[key], dtype=np.float32), source[native].astype(np.float32))
        np.testing.assert_array_equal(table["frame_index"], np.arange(count))
        np.testing.assert_array_equal(table["episode_index"], np.full(count, ep))
        np.testing.assert_array_equal(table["index"], np.arange(offset, offset + count))
        np.testing.assert_allclose(table["timestamp"], np.arange(count) / 50, atol=1e-5, rtol=0)
        tasks = {int(x) for x in table["task_index"]}
        assert len(tasks) == 1
        assert dataset.meta.tasks[tasks.pop()] == metadata["task"]
        episode = {"episode_index": ep, "source": entry["path"], "frame_count": count, "cameras": {}}
        phase_changes = np.flatnonzero(source["phase"][1:] != source["phase"][:-1]) + 1
        sample_indices = {0, count // 2, count - 1, *phase_changes.tolist()}
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            video_path = root / dataset.meta.get_video_file_path(ep, key)
            errors, timestamps = [], []
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                assert float(stream.average_rate) == 50
                assert (stream.width, stream.height) == (224, 224)
                for index, frame in enumerate(container.decode(stream)):
                    timestamps.append(float(frame.pts * frame.time_base))
                    if index in sample_indices:
                        with Image.open(source_root / entry["path"] / "frames" / camera / f"{index:06d}.png") as image:
                            expected = np.asarray(image, dtype=np.float32)
                        actual = frame.to_ndarray(format="rgb24").astype(np.float32)
                        errors.append(float(np.abs(actual - expected).mean()))
            assert len(timestamps) == count, (camera, len(timestamps), count)
            np.testing.assert_allclose(timestamps, np.arange(count) / 50, atol=1e-6, rtol=0)
            assert max(errors) < 5, (camera, max(errors))
            episode["cameras"][camera] = {"decoded_frames": len(timestamps),
                "source_comparison_samples": len(errors), "max_rgb_mean_abs_error_255": max(errors)}
        # Exercise actual training-style action chunks at both ends and across phase transitions.
        for index in sorted(sample_indices):
            row = dataset[offset + index]
            expected = source["action"][np.minimum(index + np.arange(50), count - 1)].astype(np.float32)
            np.testing.assert_array_equal(row["action"].numpy(), expected)
            np.testing.assert_array_equal(row["observation.state"].numpy(), source["state"][index].astype(np.float32))
            assert row["task"] == metadata["task"]
            for camera in CAMERAS:
                pixels = row[f"observation.images.{camera}"]
                assert tuple(pixels.shape) == (3, 224, 224)
                assert bool(pixels.isfinite().all()) and 0 <= float(pixels.min()) <= float(pixels.max()) <= 1
        episode["action_chunk_samples"] = len(sample_indices)
        report["episodes"].append(episode)
        offset += count
        print(f"Validated episode {ep}: {count} frames, both videos and action chunks", flush=True)
    assert offset == len(dataset) == dataset.meta.total_frames
    assert len(report["episodes"]) == dataset.meta.total_episodes
    report.update(passed=True, total_episodes=len(report["episodes"]), total_frames=offset,
                  state_dim=15, action_dim=9, action_horizon=50,
                  source_float32_values_identical=True, task_text_mapping_verified=True,
                  all_video_frame_counts_and_timestamps_verified=True)
    write_json(root / "validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    validate(arguments.root, arguments.source_root)
