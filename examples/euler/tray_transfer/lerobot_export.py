"""Export explicitly selected pre-step recordings with the pinned LeRobot v2.1 API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .episode_recording import CAMERAS, TASK_TEXTS, decode_action

LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episode(directory):
    """Validate the recording boundary without modifying or re-timing its data."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("success") is not True or metadata.get("complete") is not True:
        raise ValueError(f"Not a complete successful episode: {directory}")
    if metadata["task"] not in TASK_TEXTS:
        raise ValueError("Unexpected task text; expected one of the three recorded English prompts")
    if metadata["fps"] != 50 or metadata["state_dim"] != 15 or metadata["action_dim"] != 9:
        raise ValueError("Expected 50 Hz, measured state15 and absolute action9")
    with np.load(directory / "episode.npz", allow_pickle=False) as source:
        arrays = dict(source)
    count = metadata["frame_count"]
    if count <= 0:
        raise ValueError("Empty episode")
    for key, shape in (("state", (count, 15)), ("action", (count, 9)),
                       ("raw_action", (count, 15))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key} shape or values")
    np.testing.assert_allclose(np.stack([decode_action(a) for a in arrays["action"]]),
                               arrays["raw_action"], atol=1e-7, rtol=0)
    if not np.array_equal(arrays["frame_index"], np.arange(count)):
        raise ValueError("Non-contiguous frame indices")
    if (arrays["camera_indices"].shape != (count, 2)
            or not np.array_equal(arrays["camera_indices"],
                                  np.repeat(arrays["simulate_index"][:, None], 2, axis=1))):
        raise ValueError("Unsynchronized camera indices")
    if not np.allclose(np.diff(arrays["sim_time"]), .02, atol=1e-7, rtol=0):
        raise ValueError("Non-uniform simulation timestamps")
    for camera in CAMERAS:
        expected = {f"{i:06d}.png" for i in range(count)}
        actual = {path.name for path in (directory / "frames" / camera).glob("*.png")}
        if actual != expected:
            raise ValueError(f"Missing or unexpected {camera} frames")
    # Camera timestamp_ns is wall-clock capture time, not simulated time.
    arrays["timestamp"] = np.arange(count, dtype=np.float64) / 50
    return metadata, arrays


def export(manifest_path, source_root, output, repo_id):
    # Import here so source-contract tests run in the existing simulation environment.
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.utils import DEFAULT_IMAGE_PATH
    from lerobot.common.datasets.video_utils import encode_video_frames

    if CODEBASE_VERSION != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, found {CODEBASE_VERSION}")
    output, source_root = Path(output), Path(source_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest["episodes"]
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Manifest contains duplicate episode paths")
    sources = [load_episode(source_root / entry["path"]) for entry in entries]
    first = sources[0][0]
    for metadata, _ in sources:
        if any(metadata[key] != first[key] for key in ("state_names", "action_names", "camera_order")):
            raise ValueError("Source joint/camera ordering differs")

    class H264Dataset(LeRobotDataset):
        """Keep the upstream writer, choosing broadly decodable, high-quality H.264."""
        def encode_episode_videos(self, episode_index):
            paths = {}
            for key in self.meta.video_keys:
                destination = self.root / self.meta.get_video_file_path(episode_index, key)
                images = self.root / DEFAULT_IMAGE_PATH.format(
                    image_key=key, episode_index=episode_index, frame_index=0)
                if destination.exists():
                    raise FileExistsError(destination)
                encode_video_frames(images.parent, destination, self.fps, vcodec="h264",
                                    pix_fmt="yuv420p", crf=18, g=2, overwrite=True)
                paths[key] = str(destination)
            return paths

    features = {
        "observation.state": {"dtype": "float32", "shape": (15,), "names": first["state_names"]},
        "action": {"dtype": "float32", "shape": (9,), "names": first["action_names"]},
        **{f"observation.images.{camera}": {"dtype": "video", "shape": (224, 224, 3),
                                            "names": ["height", "width", "channels"]}
           for camera in CAMERAS},
    }
    dataset = H264Dataset.create(repo_id, fps=50, root=output,
                                robot_type="unitree_g1_right_arm_omnipicker_ad",
                                features=features, use_videos=True, video_backend="pyav",
                                image_writer_threads=4)
    provenance = {
        "repo_id": repo_id, "lerobot_revision": LEROBOT_REVISION,
        "manifest": manifest, "source_root_at_conversion": str(source_root.resolve()),
        "alignment": "observation_t -> absolute action_t; no shift or resampling",
        "video_encoding": "H264 CRF18 YUV420P; RGB originals retained separately",
        "state_dim": 15, "action_dim": 9, "fps": 50,
        "gripper_expansion": "[0,0,a,d,0,0,-a,d]", "augmentation": None,
        "split_policy": "all accepted demonstrations are train; no independent evaluation split",
        "numeric_dtype_conversion": "float64 to float32", "episodes": [],
    }
    origin_dir = output / "provenance"
    origin_dir.mkdir()
    try:
        for index, (entry, (metadata, arrays)) in enumerate(zip(entries, sources)):
            directory = source_root / entry["path"]
            count = metadata["frame_count"]
            print(f"Export {index + 1}/{len(entries)}: {entry['path']} ({count} frames)", flush=True)
            for frame in range(count):
                images = {}
                for camera in CAMERAS:
                    with Image.open(directory / "frames" / camera / f"{frame:06d}.png") as image:
                        if image.mode != "RGB" or image.size != (224, 224):
                            raise ValueError(f"Invalid RGB frame: {directory}, {camera}, {frame}")
                        images[f"observation.images.{camera}"] = np.array(image)
                dataset.add_frame({"observation.state": arrays["state"][frame].astype(np.float32),
                                   "action": arrays["action"][frame].astype(np.float32),
                                   "task": metadata["task"],
                                   **images})
            dataset.save_episode()
            trajectory_hash = hashlib.sha256(arrays["state"].tobytes() + arrays["action"].tobytes()).hexdigest()
            provenance["episodes"].append({
                **entry, "episode_index": index, "frame_count": count, "task": metadata["task"],
                "source_npz_sha256": sha256(directory / "episode.npz"),
                "source_metadata_sha256": sha256(directory / "metadata.json"),
                "trajectory_sha256": trajectory_hash,
                "includes_return_home": "return_home" in arrays["phase"],
            })
            write_json(origin_dir / f"episode_{index:06d}_metadata.json", metadata)
            # Keep source phase/clock indices out of model observations, but preserve provenance.
            np.savez_compressed(origin_dir / f"episode_{index:06d}_timeline.npz",
                                phase=arrays["phase"], sim_time=arrays["sim_time"],
                                simulate_index=arrays["simulate_index"], camera_indices=arrays["camera_indices"])
            write_json(origin_dir / "export.json", provenance)
    finally:
        dataset.stop_image_writer()
    print(f"Exported {dataset.meta.total_episodes} episodes / "
          f"{dataset.meta.total_frames} frames to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/g1_lightblue_box_tray_v1")
    args = parser.parse_args()
    export(args.manifest, args.source_root, args.output, args.repo_id)


if __name__ == "__main__":
    main()
