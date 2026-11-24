"""Convert raw successful episodes directly to the reference LeRobot v3 recipe."""

import argparse
import json
from pathlib import Path
import subprocess

from image_preprocessing import prepare_camera


TASK = "Pick up the blue Intel processor box and place it in the wooden basket."
FRAME_BYTES = 1280 * 480 * 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--new-collection", action="store_true",
                        help="Allow fresh 200-episode collection with a different total frame count")
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    folders = sorted(raw.glob("episode_*"))
    if not folders:
        parser.error("No episode_* folders found")
    if output.exists() or Path(f"{output}.summary.json").exists():
        parser.error("Output already exists")

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    accepted, skipped = [], []
    for folder in folders:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if not result.get("success"):
            skipped.append(folder.name)
            continue
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
            states = trajectory["state"].astype(np.float32)
            actions = trajectory["action"].astype(np.float32)
            times = trajectory["time"]
        require(states.shape == actions.shape == (result["num_frames"], 9),
                f"State/action shape mismatch: {folder.name}")
        require(np.isfinite(states).all() and np.isfinite(actions).all(),
                f"Non-finite state/action: {folder.name}")
        require(times.shape == (len(states),) and np.isfinite(times).all()
                and np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-6),
                f"Invalid 50 Hz timestamps: {folder.name}")
        require(result["video_hz"] == result["control_hz"] == 50,
                f"Control/video rate mismatch: {folder.name}")
        accepted.append((folder, len(states)))
    require(len(accepted) == 200, "Expected 200 successful episodes")
    if not args.new_collection:
        require(sum(count for _, count in accepted) == 277084,
                "Expected 277084 original frames; use --new-collection for newly collected data")

    image = {"dtype": "image", "shape": (224, 224, 3),
             "names": ["height", "width", "channel"]}
    features = {
        "observation.images.image": image,
        "observation.images.wrist_image": image,
        "observation.state": {"dtype": "float32", "shape": (9,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (9,), "names": ["action"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=output.name, root=output, fps=50,
        robot_type="unitree_g1_fixed_torso_dual_arm", use_videos=False,
        features=features,
    )
    kept = []
    for folder, count in accepted:
        with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
            states = trajectory["state"].astype(np.float32)
            actions = trajectory["action"].astype(np.float32)
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(folder / "cameras.mp4"),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE,
        )
        try:
            for index in range(count):
                payload = process.stdout.read(FRAME_BYTES)
                require(len(payload) == FRAME_BYTES,
                        f"Video shorter than trajectory: {folder.name}, frame {index}")
                pair = np.frombuffer(payload, dtype=np.uint8).reshape(480, 1280, 3)
                dataset.add_frame({
                    "observation.images.image": prepare_camera(pair[:, :640]),
                    "observation.images.wrist_image": prepare_camera(pair[:, 640:]),
                    "observation.state": states[index],
                    "action": actions[index], "task": TASK,
                })
            require(not process.stdout.read(1),
                    f"Video longer than trajectory: {folder.name}")
            require(process.wait(timeout=60) == 0, f"FFmpeg failed: {folder.name}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        dataset.save_episode()
        kept.append({"episode": folder.name, "frames": count})
        print("CONVERTED", len(kept), folder.name, count, flush=True)
    dataset.finalize()
    summary = {"episodes": len(kept), "frames": sum(row["frames"] for row in kept),
               "skipped_not_success": skipped, "kept": kept}
    Path(f"{output}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print("CONVERSION_COMPLETE", summary["episodes"], summary["frames"])


if __name__ == "__main__":
    main()
