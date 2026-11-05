"""Convert one raw split to LeRobot v2.1, retaining every native frame and action.

No crop, resize, resampling or action shift. Use the pinned conversion environment;
this module imports no simulator. Failed recordings are listed and excluded.
"""

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .recording import ACTION_NAMES, CAMERAS, RAW_FORMAT, STATE_NAMES, gripper_mapping

LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
IMAGE_KEYS = dict(
    zip(CAMERAS, ("observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"))
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Source path escapes episode: {relative}")
    return path


def discover_episodes(source_root, split):
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    entries, skipped = [], []
    folders = ("train",) if split == "train" else ("val", "validation")
    trials = sorted(trial for folder in folders for trial in (Path(source_root) / "raw" / folder).glob("episode_*"))
    for trial in trials:
        path = trial / "episode" / "metadata.json"
        if not path.is_file():
            skipped.append({"path": str(trial), "reason": "Missing finalized recording"})
            continue
        metadata = json.loads(path.read_text())
        result_path = trial / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        if result.get("success") is not True or any(
            metadata.get(key) is not True for key in ("success", "complete", "raw_valid")
        ):
            skipped.append({"path": str(trial), "reason": "Motion or recording failed"})
            continue
        entries.append(path.parent)
    if not entries:
        raise ValueError(f"No successful finalized {split} recordings under {source_root}")
    if len({entry.resolve() for entry in entries}) != len(entries):
        raise ValueError("Duplicate source episode paths")
    return entries, skipped


def load_episode(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (
        metadata.get("format") != RAW_FORMAT
        or any(metadata.get(key) is not True for key in ("success", "complete", "raw_valid"))
        or metadata.get("errors") != []
    ):
        raise ValueError(f"Invalid or failed raw episode: {directory}")
    if (metadata.get("fps"), metadata.get("state_dim"), metadata.get("action_dim")) != (50, 30, 16):
        raise ValueError("Expected 50 Hz, state30 and native action16")
    if (
        metadata.get("state_names") != list(STATE_NAMES)
        or metadata.get("action_names") != list(ACTION_NAMES)
        or metadata.get("camera_order") != list(CAMERAS)
    ):
        raise ValueError("Unexpected joint/action/camera order")
    if metadata.get("gripper_mapping") != gripper_mapping():
        raise ValueError("Gripper mapping differs from the native action16 contract")
    task = metadata.get("task")
    if not isinstance(task, str) or not task.strip() or not task.isascii():
        raise ValueError("Expected a nonempty English task")
    synchronization = metadata.get("synchronization", {})
    if (
        synchronization.get("alignment") != "pre-step observation_t paired with action_t"
        or synchronization.get("latest_frame_fallback") is not False
    ):
        raise ValueError("Expected exact indexed pre-step observations")
    count = metadata["frame_count"]
    if type(count) is not int or count < 1:
        raise ValueError("Empty or invalid frame count")
    with np.load(directory / "episode.npz", allow_pickle=False) as source:
        arrays = dict(source)
    for key, shape in (
        ("state", (count, 30)),
        ("action", (count, 16)),
        ("sim_time", (count,)),
        ("frame_index", (count,)),
    ):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key} shape or values")
    if np.any((arrays["action"][:, [7, 15]] < 0) | (arrays["action"][:, [7, 15]] > 1)):
        raise ValueError("Gripper closure must be in [0, 1]")
    if (
        arrays["frame_index"].dtype.kind not in "iu"
        or np.any(arrays["frame_index"] < 0)
        or np.any(np.diff(arrays["frame_index"]) <= 0)
    ):
        raise ValueError("Render indices must be increasing integers")
    if np.any(arrays["sim_time"] < 0) or not np.allclose(np.diff(arrays["sim_time"]), 0.02, atol=1e-7, rtol=0):
        raise ValueError("Nonuniform simulation timestamps")
    if arrays["stage"].shape != (count,) or not all(isinstance(s, str) and s for s in arrays["stage"]):
        raise ValueError("Invalid stage labels")
    for camera in CAMERAS:
        info = metadata["cameras"][camera]
        shape = info["native_shape"]
        if (
            len(shape) != 3
            or shape[2] != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or info["decoded_frame_count"] != count
        ):
            raise ValueError(f"{camera}: invalid video dimensions or count")
        rows = json.loads(source_file(directory, info["frame_metadata"]).read_text())
        if len(rows) != count:
            raise ValueError(f"{camera}: sidecar length mismatch")
        stamps = [row["timestamp_ns"] for row in rows]
        if any(type(t) is not int or t < 0 for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"{camera}: invalid native timestamps")
        for i, row in enumerate(rows):
            if (
                row["row_index"] != i
                or row["frame_index"] != arrays["frame_index"][i]
                or row["is_keyframe"] is not True
                or not np.isclose(row["sim_time"], arrays["sim_time"][i], atol=1e-7, rtol=0)
            ):
                raise ValueError(f"{camera}: camera/state alignment mismatch at row {i}")
        if not source_file(directory, info["path"]).is_file():
            raise ValueError(f"{camera}: missing video")
    return metadata, arrays


def iter_frames(directory, metadata, arrays):
    """Stream bounded memory RGB and verify every source pixel against its sidecar."""
    import av

    directory = Path(directory)
    with ExitStack() as stack:
        decoders, sidecars = {}, {}
        for camera in CAMERAS:
            info = metadata["cameras"][camera]
            container = stack.enter_context(av.open(str(source_file(directory, info["path"]))))
            decoders[camera] = iter(container.decode(video=0))
            sidecars[camera] = json.loads(source_file(directory, info["frame_metadata"]).read_text())
        for i in range(len(arrays["state"])):
            images = {}
            for camera in CAMERAS:
                frame = next(decoders[camera], None)
                if frame is None:
                    raise ValueError(f"{camera}: missing frame {i}")
                rgb = frame.to_ndarray(format="rgb24")
                if (
                    list(rgb.shape) != metadata["cameras"][camera]["native_shape"]
                    or hashlib.sha256(rgb.tobytes()).hexdigest() != sidecars[camera][i]["rgb_sha256"]
                ):
                    raise ValueError(f"{camera}: source pixels/shape differ at row {i}")
                # Matroska timestamps have millisecond precision; original cadence is 20 ms.
                if frame.pts is None or abs(float(frame.pts * frame.time_base) - i / 50) > 0.001:
                    raise ValueError(f"{camera}: video timestamp mismatch at row {i}")
                images[camera] = rgb
            yield images
        if any(next(decoder, None) is not None for decoder in decoders.values()):
            raise ValueError("Extra video frames without robot rows")


def camera_contract(metadata):
    return {
        camera: (metadata["cameras"][camera]["native_shape"], metadata["cameras"][camera]["properties"])
        for camera in CAMERAS
    }


def check_configuration(reference, metadata):
    """Allow the recorded expert release policy only; keep robot contracts strict."""
    controls = []
    for item in (reference, metadata):
        control = dict(item["control"])
        if "release_fallback" in control:
            if control.pop("release_fallback") != "checked_hover_if_lowering_unplannable":
                raise ValueError("Unknown expert release_fallback policy")
        controls.append(control)
    if controls[0] != controls[1] or any(
        metadata.get(key) != reference.get(key) for key in ("physics", "arms", "actuator_joint_names")
    ):
        raise ValueError("Mixed physics or robot control configuration")
    if camera_contract(metadata) != camera_contract(reference):
        raise ValueError("Mixed camera configuration")


def expert_provenance(directory, metadata):
    """Retain expert decisions without treating missing historical records as empty."""
    report = {"control": metadata["control"]}
    for key in ("target_selection", "expert_trajectory"):
        report[key] = {"recorded": key in metadata, "value": metadata.get(key)}
    for key in ("selection", "release_fallbacks"):
        path = Path(directory).parent / f"{key}.json"
        report[key] = {"recorded": path.is_file()}
        if path.is_file():
            report[key].update(value=json.loads(path.read_text()), sha256=sha256(path))
    return report


def export(source_root, output, repo_id="local/g1_med_sort_v2", split="train"):
    """Publish output only after conversion and reader validation both succeed."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output or partial output already exists: {output}")
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.utils import DEFAULT_IMAGE_PATH
    from lerobot.common.datasets.video_utils import encode_video_frames

    if CODEBASE_VERSION != "v2.1":
        raise ValueError(f"Expected pinned LeRobot v2.1, found {CODEBASE_VERSION}")
    entries, skipped = discover_episodes(source_root, split)
    first, _ = load_episode(entries[0])
    for directory in entries:
        metadata, _ = load_episode(directory)
        try:
            check_configuration(first, metadata)
        except ValueError as error:
            raise ValueError(f"{directory}: {error}") from error

    class H264Dataset(LeRobotDataset):
        def encode_episode_videos(self, episode_index):
            paths = {}
            for key in self.meta.video_keys:
                destination = self.root / self.meta.get_video_file_path(episode_index, key)
                image = self.root / DEFAULT_IMAGE_PATH.format(image_key=key, episode_index=episode_index, frame_index=0)
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite an episode video: {destination}")
                # v2.1 uses overwrite for parent-directory reuse too; episodes share it.
                encode_video_frames(
                    image.parent, destination, self.fps, vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, overwrite=True
                )
                paths[key] = str(destination)
            return paths

    action_names = first["action_names"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (30,), "names": first["state_names"]},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        **{
            key: {
                "dtype": "video",
                "shape": tuple(first["cameras"][camera]["native_shape"]),
                "names": ["height", "width", "channels"],
            }
            for camera, key in IMAGE_KEYS.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 2_000_000_000:
        raise OSError("Conversion needs free space for temporary PNG frames; less than 2 GB available")
    dataset = H264Dataset.create(
        repo_id,
        fps=50,
        root=partial,
        features=features,
        use_videos=True,
        robot_type="unitree_g1_dual_arm_omnipicker_closure",
        video_backend="pyav",
        image_writer_threads=4,
    )
    origin = partial / "provenance"
    origin.mkdir()
    provenance = dict(
        repo_id=repo_id,
        lerobot_revision=LEROBOT_REVISION,
        split=split,
        source_root=str(source_root),
        source_episodes=[str(p.relative_to(source_root)) for p in entries],
        skipped=skipped,
        episodes=[],
        complete=False,
        fps=50,
        state_dim=30,
        action_dim=len(action_names),
        raw_format=RAW_FORMAT,
        gripper_mapping=gripper_mapping(),
        alignment="observation_t -> absolute action_t; no shift or resampling",
        image_processing={
            "crop": None,
            "resize": None,
            "camera_keys": IMAGE_KEYS,
            "encoding": "H264 CRF18 YUV420P; lossy encoding, raw FFV1 retained",
        },
        numeric_conversion="float64 to float32",
        model_training_verified=False,
    )
    write_json(origin / "export.json", provenance)
    for item in skipped:
        print("SKIP", item["path"], item["reason"], flush=True)
    try:
        for ep, directory in enumerate(entries):
            metadata, arrays = load_episode(directory)
            actions = arrays["action"].astype(np.float32)
            print(f"EXPORT {ep + 1}/{len(entries)}: {directory} ({metadata['frame_count']} frames)", flush=True)
            for i, images in enumerate(iter_frames(directory, metadata, arrays)):
                if i % 100 == 0 and shutil.disk_usage(partial).free < 1_000_000_000:
                    raise OSError("Less than 1 GB free during conversion; partial output retained")
                dataset.add_frame(
                    {
                        "observation.state": arrays["state"][i].astype(np.float32),
                        "action": actions[i],
                        "task": metadata["task"],
                        **{IMAGE_KEYS[c]: rgb for c, rgb in images.items()},
                    }
                )
                # Bound queued RGB memory while allowing the PNG writers to work in parallel.
                if (i + 1) % 32 == 0:
                    dataset.image_writer.wait_until_done()
                if (i + 1) % 500 == 0:
                    print(f"  frames {i + 1}/{metadata['frame_count']}", flush=True)
            dataset.save_episode()
            write_json(origin / f"episode_{ep:06d}_metadata.json", metadata)
            write_json(origin / f"episode_{ep:06d}_expert.json", expert_provenance(directory, metadata))
            np.savez_compressed(
                origin / f"episode_{ep:06d}_timeline.npz",
                **{key: arrays[key] for key in ("sim_time", "frame_index", "stage")},
            )
            provenance["episodes"].append(
                dict(
                    episode_index=ep,
                    path=str(directory.relative_to(source_root)),
                    frame_count=metadata["frame_count"],
                    task=metadata["task"],
                    npz_sha256=sha256(directory / "episode.npz"),
                    metadata_sha256=sha256(directory / "metadata.json"),
                )
            )
            write_json(origin / "export.json", provenance)
    finally:
        dataset.stop_image_writer()
    validate(partial, source_root)
    provenance["complete"] = True
    write_json(origin / "export.json", provenance)
    partial.rename(output)
    print(f"DONE: {output} — {len(entries)} episodes, split={split}; reader validation passed", flush=True)


def check_table_timestamps(timestamps, count):
    # LeRobot stores timestamp as float32; compare with the same representable values.
    expected = (np.arange(count, dtype=np.float64) / 50).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(timestamps, dtype=np.float32), expected)


def validate(root, source_root):
    """Read parquet, all video timestamps, sampled RGB, and padded action chunks."""
    import av
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root, source_root = Path(root), Path(source_root)
    origin = json.loads((root / "provenance/export.json").read_text())
    if origin.get("raw_format") != RAW_FORMAT or origin.get("action_dim") != 16:
        raise ValueError("Expected native16 export; legacy datasets are not converted")
    if origin.get("gripper_mapping") != gripper_mapping():
        raise ValueError("Scalar gripper mapping differs from the dataset contract")
    if not origin["episodes"] or origin["source_episodes"] != [e["path"] for e in origin["episodes"]]:
        raise ValueError("Incomplete export manifest")
    dataset = LeRobotDataset(
        origin["repo_id"], root=root, video_backend="pyav", delta_timestamps={"action": [i / 50 for i in range(50)]}
    )
    report = dict(
        passed=False,
        split=origin["split"],
        reader="LeRobot v2.1 + pyav",
        episodes=[],
        state_dim=30,
        action_dim=origin["action_dim"],
        raw_format=RAW_FORMAT,
        fps=50,
        crop=None,
        resize=None,
        model_training_verified=False,
    )
    offset = 0
    for entry in origin["episodes"]:
        directory = source_file(source_root, entry["path"])
        if (
            sha256(directory / "episode.npz") != entry["npz_sha256"]
            or sha256(directory / "metadata.json") != entry["metadata_sha256"]
        ):
            raise ValueError("Source changed after conversion")
        metadata, arrays = load_episode(directory)
        actions = arrays["action"].astype(np.float32)
        ep, count = entry["episode_index"], metadata["frame_count"]
        table = pq.read_table(root / dataset.meta.get_data_file_path(ep)).to_pydict()
        for key, source in (("observation.state", "state"), ("action", "action")):
            expected = actions if source == "action" else arrays[source].astype(np.float32)
            np.testing.assert_array_equal(np.array(table[key], dtype=np.float32), expected)
        np.testing.assert_array_equal(table["frame_index"], np.arange(count))
        np.testing.assert_array_equal(table["episode_index"], np.full(count, ep))
        np.testing.assert_array_equal(table["index"], np.arange(offset, offset + count))
        check_table_timestamps(table["timestamp"], count)
        if {dataset.meta.tasks[t] for t in table["task_index"]} != {metadata["task"]}:
            raise ValueError("Task text mapping differs")
        changes = np.flatnonzero(arrays["stage"][1:] != arrays["stage"][:-1]) + 1
        samples = {0, count // 2, count - 1, *changes.tolist()}
        expected_images = {
            i: images for i, images in enumerate(iter_frames(directory, metadata, arrays)) if i in samples
        }
        errors = {}
        for camera, key in IMAGE_KEYS.items():
            timestamps, rgb_errors = [], []
            with av.open(str(root / dataset.meta.get_video_file_path(ep, key))) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if [frame.height, frame.width, 3] != metadata["cameras"][camera]["native_shape"]:
                        raise ValueError("Output video dimensions differ")
                    timestamps.append(float(frame.pts * frame.time_base))
                    if i in samples:
                        rgb_errors.append(
                            float(
                                np.abs(
                                    frame.to_ndarray(format="rgb24").astype(float) - expected_images[i][camera]
                                ).mean()
                            )
                        )
            if len(timestamps) != count or not rgb_errors or max(rgb_errors) > 8:
                raise ValueError(f"{camera}: output count or RGB alignment differs")
            np.testing.assert_allclose(timestamps, np.arange(count) / 50, atol=1e-6, rtol=0)
            errors[camera] = max(rgb_errors)
        for i in sorted(samples):
            row = dataset[offset + i]
            expected = actions[np.minimum(i + np.arange(50), count - 1)]
            np.testing.assert_array_equal(row["action"].numpy(), expected)
            np.testing.assert_array_equal(row["observation.state"].numpy(), arrays["state"][i].astype(np.float32))
            if row["task"] != metadata["task"]:
                raise ValueError("Reader task mismatch")
            for camera, key in IMAGE_KEYS.items():
                h, w, _ = metadata["cameras"][camera]["native_shape"]
                pixels = row[key].numpy()
                if pixels.shape != (3, h, w) or not np.isfinite(pixels).all():
                    raise ValueError("Reader image shape or values differ")
                if np.abs(pixels.transpose(1, 2, 0) * 255 - expected_images[i][camera]).mean() > 8:
                    raise ValueError("Reader image does not match the expected source frame")
        report["episodes"].append(dict(episode_index=ep, frames=count, max_rgb_mae_255=errors))
        offset += count
        print(f"VALIDATED episode {ep}: {count} rows, three videos, task and action chunks", flush=True)
    if offset != len(dataset) or dataset.meta.total_episodes != len(origin["episodes"]):
        raise ValueError("Dataset totals differ")
    report.update(passed=True, total_frames=offset, total_episodes=len(origin["episodes"]))
    write_json(root / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/g1_med_sort_v2")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(args.output, args.source_root)
    else:
        export(args.source_root, args.output, args.repo_id, args.split)


if __name__ == "__main__":
    main()
