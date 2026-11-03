"""Collect a saved episode plan with disk limits and fail-closed resume."""

import argparse
import fcntl
import json
import math
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .recording import RAW_FORMAT, gripper_mapping

HOST_WIDE_COLLECTION_LOCK = Path("/tmp/orcalab-medication-collection.lock")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def native_metadata(meta):
    return (
        meta.get("format") == RAW_FORMAT
        and meta.get("action_dim") == 16
        and meta.get("gripper_mapping") == gripper_mapping()
    )


def ensure_native_dataset(root):
    """Do not append to legacy or differently calibrated datasets."""
    contract = {"format": RAW_FORMAT, "action_dim": 16, "state_dim": 30, "gripper_mapping": gripper_mapping()}
    path = root / "dataset_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Expected native16 dataset; use a new dataset directory")
    for metadata in (root / "raw").glob("*/episode_*/episode/metadata.json"):
        if not native_metadata(json.loads(metadata.read_text())):
            raise ValueError(f"Expected native16 dataset; legacy/incompatible recording: {metadata}")
    if not path.exists():
        save_json(path, contract)


def ordered_rows(plan):
    rows = plan["episodes"]
    ids, seeds = set(), set()
    for row in rows:
        if not re.fullmatch(r"episode_\d{4}", row["episode_id"]) or row["episode_id"] in ids:
            raise ValueError("Invalid or repeated episode ID")
        if row["split"] not in ("train", "validation"):
            raise ValueError("Unknown split")
        if not isinstance(row["seed"], int) or row["seed"] < 0 or row["seed"] in seeds:
            raise ValueError("Invalid or repeated layout seed")
        if row["task_index"] not in (0, 1, 2):
            raise ValueError("Invalid task index")
        ids.add(row["episode_id"])
        seeds.add(row["seed"])
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    ordered = []
    while train or validation:
        ordered.extend(train[:4])
        del train[:4]
        ordered.extend(validation[:1])
        del validation[:1]
    return ordered


def valid_episode(output):
    try:
        result = json.loads((output / "result.json").read_text())
        meta = json.loads((output / "episode/metadata.json").read_text())
        return (
            result.get("success") is True
            and meta.get("raw_valid") is True
            and meta.get("frame_count", 0) > 0
            and native_metadata(meta)
        )
    except (OSError, ValueError):
        return False


def run_episode(plan, row, output, root, maximum_bytes, reserve_bytes):
    command = [
        sys.executable,
        "-u",
        "-m",
        "examples.euler.medication_sorting.main",
        "--mode",
        "run",
        "--execute",
        "--record",
        "--object",
        "all",
        "--output",
        str(output),
        "--layout-mode",
        plan["layout_mode"],
        "--seed",
        str(row["seed"]),
        "--task-index",
        str(row["task_index"]),
        "--xy-jitter",
        str(plan["initial_xy_jitter_m"]),
        "--yaw-jitter-deg",
        str(plan["initial_yaw_jitter_deg"]),
    ]
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    process = None
    try:
        with (logs / f"{row['episode_id']}.log").open("x") as log:
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[3], stdout=log, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                time.sleep(5)
                used, free = tree_bytes(root), shutil.disk_usage(root).free
                save_json(
                    root / "live.json",
                    {
                        "episode_id": row["episode_id"],
                        "pid": process.pid,
                        "elapsed_seconds": time.monotonic() - started,
                        "dataset_bytes": used,
                        "free_bytes": free,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if free < reserve_bytes or used >= maximum_bytes or time.monotonic() - started > 3600:
                    raise RuntimeError("Collection hit disk budget, free-space reserve, or one-hour episode timeout")
            return process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def collect(plan_path, maximum_bytes, reserve_bytes, estimate_bytes):
    if min(maximum_bytes, reserve_bytes, estimate_bytes) <= 0:
        raise ValueError("Capacity limits must be positive")
    root = plan_path.resolve().parent
    plan = json.loads(plan_path.read_text())
    if plan.get("action_dim", 16) != 16:
        raise ValueError("Legacy collection plan; use a new native16 plan/directory")
    rows = ordered_rows(plan)
    status = {
        "status": "running",
        "completed": [],
        "active": None,
        "maximum_bytes": maximum_bytes,
        "reserve_bytes": reserve_bytes,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "training_ready": False,
    }
    with (root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_native_dataset(root)
        try:
            for row in rows:
                output = root / "raw" / row["split"] / row["episode_id"]
                if output.exists():
                    if not valid_episode(output):
                        raise RuntimeError(f"Existing incomplete/failed episode requires review: {output}")
                else:
                    used, free = tree_bytes(root), shutil.disk_usage(root).free
                    if used + estimate_bytes > maximum_bytes or free < reserve_bytes + estimate_bytes:
                        status.update(status="capacity_reached", dataset_bytes=used, free_bytes=free)
                        break
                    output.parent.mkdir(parents=True, exist_ok=True)
                    status["active"] = row["episode_id"]
                    save_json(root / "collection_status.json", status)
                    print(f"COLLECT {row['episode_id']} split={row['split']} seed={row['seed']}", flush=True)
                    code = run_episode(plan, row, output, root, maximum_bytes, reserve_bytes)
                    if code or not valid_episode(output):
                        raise RuntimeError(f"Episode failed motion/recording validation: {output}; exit={code}")
                measured_bytes = tree_bytes(output)
                estimate_bytes = max(estimate_bytes, int(measured_bytes * 1.25))
                status["completed"].append({**row, "status": "complete", "path": str(output), "bytes": measured_bytes})
                status["active"] = None
                save_json(root / "collection_status.json", status)
            else:
                status["status"] = "complete"
        except (Exception, KeyboardInterrupt) as exc:
            status.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        finally:
            status.update(
                finished_at=datetime.now(timezone.utc).isoformat(),
                dataset_bytes=tree_bytes(root),
                free_bytes=shutil.disk_usage(root).free,
            )
            save_json(root / "collection_status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-gb", type=float, default=25)
    parser.add_argument("--reserve-gb", type=float, default=4)
    parser.add_argument("--episode-budget-gb", type=float, default=2.5)
    args = parser.parse_args()
    if not args.execute:
        parser.error("Live simulation control requires --execute")
    limits = (args.max_gb, args.reserve_gb, args.episode_budget_gb)
    if any(not math.isfinite(n) or n <= 0 for n in limits):
        parser.error("Capacity limits must be finite and positive")
    with HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = collect(args.plan, *(int(n * 1_000_000_000) for n in limits))
    if status["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
