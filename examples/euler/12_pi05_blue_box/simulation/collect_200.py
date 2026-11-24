"""Collect independent successful V9 episodes with synchronized video."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from collect import DualCollector, save_json
from task_config import ROOT


def verify_episode(folder, result):
    with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
        count = len(trajectory["action"])
        if (trajectory["action"].shape != (count, 9)
                or trajectory["state"].shape != (count, 9)
                or count != result["num_frames"]
                or not np.isfinite(trajectory["state"]).all()
                or not np.isfinite(trajectory["action"]).all()):
            raise ValueError("Invalid state/action data")
        required = {"open", "pregrasp", "approach", "close", "lift",
                    "transfer", "place", "release", "retreat", "settle"}
        if not required.issubset(set(trajectory["phase"].tolist())):
            raise ValueError("Episode is missing required collection phases")
    video = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
        "-of", "json", str(folder / "cameras.mp4"),
    ], text=True))["streams"][0]
    if (int(video["nb_read_frames"]), video["width"], video["height"], video["r_frame_rate"]) != (
        count, 1280, 480, "50/1",
    ):
        raise ValueError("Video/action alignment failed")
    return {"verified": True, "frames": count,
            "trajectory_sha256": hashlib.sha256((folder / "trajectory.npz").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v9")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=240)
    args = parser.parse_args()
    if args.count < 1 or args.max_attempts < args.count:
        parser.error("Require count >= 1 and max-attempts >= count")
    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    collector = None
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "simulation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for folder in sorted(raw.glob("episode_*")):
            if not (folder / "result.json").is_file():
                raise RuntimeError("An incomplete episode requires inspection")
            result = json.loads((folder / "result.json").read_text())
            if not result.get("recorded"):
                raise RuntimeError("Cannot mix offline diagnostics with recorded collection")
            if result["success"]:
                verify_episode(folder, result)
            records.append({"episode": folder.name, **result})
        accepted = sum(bool(row["success"]) for row in records)
        if accepted >= args.count:
            print(f"Already collected {accepted} accepted episodes")
            return
        failures = 0
        try:
            collector = DualCollector(False, 0.11, -0.04)
            start = max((int(row["episode"].split("_")[-1]) for row in records), default=-1) + 1
            for attempt in range(start, args.max_attempts):
                folder = raw / f"episode_{attempt:03d}"
                result = collector.run_episode(folder)
                if result["success"]:
                    save_json(folder / "validation.json", verify_episode(folder, result))
                    accepted += 1
                    failures = 0
                else:
                    failures += 1
                records.append({"episode": folder.name, **result})
                save_json(output / "status.json", {
                    "accepted": accepted, "target": args.count,
                    "complete": accepted >= args.count, "trials": records,
                })
                print("COLLECTION_PROGRESS", accepted, args.count, flush=True)
                if accepted >= args.count:
                    return
                if failures >= 2:
                    raise RuntimeError("Two failed demonstrations; inspect before resuming")
            raise RuntimeError("Attempt budget exhausted")
        finally:
            if collector is not None:
                collector.close()


if __name__ == "__main__":
    main()
