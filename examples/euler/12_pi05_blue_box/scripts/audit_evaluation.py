"""Audit complete trial records; this does not rerun physics or certify generalization."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from evaluation_metadata import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def probe_video(path):
    value = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ], text=True, timeout=120)
    streams = json.loads(value)["streams"]
    require(len(streams) == 1, "Expected one camera video stream")
    return streams[0]


def audit_trial(folder, result, settings, probe=probe_video):
    require(type(result["success"]) is bool, "Invalid success flag")
    require(result["policy_actions_replaced"] == 0, "Policy action replacement reported")
    require(result["image_mode"] == settings["image_mode"], "Image mode mismatch")
    require(result["control_hz"] == 50, "Control frequency mismatch")
    count = result["num_frames"]
    require(type(count) is int and 0 < count <= settings["max_steps"],
            "No action/video evidence, or invalid frame count")
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    required = (
        "state", "action", "target", "wall_clearance_m", "basket_floor_contact",
        "released", "object_speed", "stable_steps", "time", "complete_step",
    )
    for name in required:
        value = arrays[name]
        require(len(value) == count and np.isfinite(value).all(),
                f"Invalid trajectory field: {name}")
        if name not in {"state", "action", "target"}:
            require(value.shape == (count,), f"Invalid scalar field shape: {name}")
    require(arrays["state"].shape == arrays["action"].shape == (count, 9),
            "Expected nine state/action dimensions")
    require(arrays["target"].shape == (count, 3), "Invalid target position shape")
    if count > 1:
        dt = np.diff(arrays["time"])
        require(np.all(dt > 0) and np.all(dt <= 0.020001), "Invalid control timestamps")
        require(np.allclose(dt[arrays["complete_step"][1:].astype(bool)], 0.02,
                            atol=1e-6, rtol=0), "Complete actions are not at 50 Hz")
    with np.load(folder / "predicted_chunks.npz", allow_pickle=False) as data:
        chunks = data["actions"]
    needed = (count + 49) // 50
    allowed_counts = {needed}
    if count % 50 == 0 and not result["success"]:
        allowed_counts.add(needed + 1)
    require(chunks.ndim == 3 and chunks.shape[0] in allowed_counts
            and chunks.shape[1:] == (50, 9) and np.isfinite(chunks).all(),
            "Invalid prediction chunks")
    require(np.array_equal(chunks.reshape(-1, 9)[:count], arrays["action"]),
            "Recorded commands differ from policy predictions")
    closed, active, stable = False, False, 0
    lift = 0.0
    for i in range(count):
        action = arrays["action"][i]
        closed |= bool(np.all(action[7:] < 0))
        active |= closed and bool(np.all(action[7:] > 0.3))
        lift = max(lift, float(arrays["target"][i, 2]) - result["initial_target_z"])
        good = (lift > 0.08 and arrays["wall_clearance_m"][i] >= -0.001
                and arrays["basket_floor_contact"][i] and arrays["released"][i]
                and arrays["object_speed"][i] < 0.02 and active
                and arrays["complete_step"][i])
        stable = stable + 1 if good else 0
        require(arrays["stable_steps"][i] == stable, "Stable-placement counter mismatch")
    require(result["active_release"] == active, "Active release summary mismatch")
    require(result["stable_success_steps"] == stable, "Final stability summary mismatch")
    require(abs(result["max_lift_m"] - lift) < 1e-6, "Maximum lift summary mismatch")
    if result["success"]:
        require(stable >= 50 and result["reason"] == "stable_released_placement",
                "Success lacks sustained released placement")
    video = probe(folder / "cameras.mp4")
    require(int(video["nb_read_frames"]) == count, "Video/action frame count mismatch")
    require((int(video["width"]), int(video["height"])) == (1280, 480),
            "Expected paired 1280x480 camera video")
    require(Fraction(video["avg_frame_rate"]) == 50, "Video is not 50 Hz")
    require(result["video_frames"] == count, "Reported video frame count mismatch")
    for name in ("head_cam", "camera_right"):
        with Image.open(folder / f"initial_{name}.png") as image:
            image.load()
            require(image.size == (640, 480) and image.mode == "RGB",
                    "Invalid initial camera image")


def audit(root, probe=probe_video):
    settings, summary = read(root / "settings.json"), read(root / "summary.json")
    requested = settings["trials"]
    require(type(requested) is int and requested > 0, "Invalid requested trial count")
    require(summary["requested"] == requested, "Requested trial count mismatch")
    require(summary["complete"] is True and summary["completed"] == requested,
            "Evaluation is incomplete")
    require(len(summary["trials"]) == len(settings["seeds"]) == requested,
            "Incomplete trial list")
    expected = {f"trial_{i:03d}" for i in range(requested)}
    require({p.name for p in root.glob("trial_*") if p.is_dir()} == expected,
            "Missing or additional trial directories")
    errors, successes, manifest = [], 0, {}
    require((root / "provenance.json").is_file(), "Missing evaluation provenance")
    metadata = read(root / "provenance.json")
    require(bool(metadata.get("checkpoint_sha256", {}).get("model.safetensors")),
            "Missing actual weight hash")
    require(bool(metadata.get("environment")), "Missing environment versions")
    for i in range(requested):
        name = f"trial_{i:03d}"
        folder = root / name
        try:
            result = read(folder / "result.json")
            require(result == summary["trials"][i], "Per-trial result differs from summary")
            require(result["seed"] == settings["seeds"][i], "Trial seed mismatch")
            successes += int(result["success"] is True)
            audit_trial(folder, result, settings, probe)
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            errors.append({"trial": name, "issue": str(error) if isinstance(error, ValueError)
                           else type(error).__name__})
    require(summary["successes"] == successes, "Success count mismatch")
    rate = successes / requested
    require(abs(summary["success_rate"] - rate) < 1e-12, "Success rate mismatch")
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix in {".json", ".npz", ".mp4", ".png"}):
            manifest[path.relative_to(root).as_posix()] = sha256(path)
    return {
        "audit_passed": not errors, "requested": requested, "successes": successes,
        "reported_success_rate": rate, "errors": errors, "files_sha256": manifest,
        "scope": ("Record consistency and logged success conditions only; "
                  "no independent physics replay, manual video review, "
                  "visual-package certification or guaranteed future success."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, destination = args.evaluation.resolve(), args.output.resolve()
    if destination.exists():
        parser.error("Audit output already exists")
    if root == destination or root in destination.parents:
        parser.error("Save audit output outside the evaluation directory")
    try:
        report = audit(root)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        report = {"audit_passed": False, "issue": (
            str(error) if isinstance(error, ValueError) else type(error).__name__
        )}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=2))
    if not report["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
