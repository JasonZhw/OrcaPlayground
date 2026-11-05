"""Start collection, inspect a scene, or test motion from one command-line entry."""

import argparse
import fcntl
import math
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .collect_dataset import HOST_WIDE_COLLECTION_LOCK, ensure_native_dataset, tree_bytes, valid_episode


def next_episode_number(root):
    """Keep IDs unique across splits, including incomplete or failed attempts."""
    numbers = []
    for path in (root / "raw").glob("*/episode_*"):
        match = re.fullmatch(r"episode_(\d+)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=-1) + 1


def run_monitored(command, root, maximum_bytes, reserve_bytes):
    """Stream the child output while retaining the lock until the child is reaped."""
    process = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3])
        while process.poll() is None:
            time.sleep(5)
            used, free = tree_bytes(root), shutil.disk_usage(root).free
            elapsed = time.monotonic() - started
            if free < reserve_bytes or used >= maximum_bytes or elapsed > 3600:
                raise RuntimeError(
                    "采集中止：触及数据集容量上限、磁盘预留空间或一小时超时。已有文件保留，请检查并复位仿真后再运行。"
                )
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


def collect_one(
    root,
    split="train",
    endpoint="127.0.0.1:50051",
    task_index=None,
    reserve_bytes=4_000_000_000,
    episode_budget_bytes=2_500_000_000,
    max_bytes=25_000_000_000,
):
    if split not in ("train", "val") or task_index not in (None, 0, 1, 2):
        raise ValueError("split must be train/val and task_index must be 0, 1, or 2")
    if any(not math.isfinite(value) or value <= 0 for value in (reserve_bytes, episode_budget_bytes, max_bytes)):
        raise ValueError("Capacity limits must be finite and positive")
    root = Path(root).resolve()
    with HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True, exist_ok=True)
        ensure_native_dataset(root)
        used, free = tree_bytes(root), shutil.disk_usage(root).free
        if used + episode_budget_bytes > max_bytes:
            raise RuntimeError(
                f"数据集容量不足：已用 {used / 1e9:.2f} GB，上限 {max_bytes / 1e9:.2f} GB，"
                f"本条预计需要 {episode_budget_bytes / 1e9:.2f} GB。"
            )
        if free < reserve_bytes + episode_budget_bytes:
            raise RuntimeError(
                f"磁盘空间不足：可用 {free / 1e9:.2f} GB，"
                f"本次至少需要 {(reserve_bytes + episode_budget_bytes) / 1e9:.2f} GB（含预留空间）。"
            )
        number = next_episode_number(root)
        output = root / "raw" / split / f"episode_{number:04d}"
        output.parent.mkdir(parents=True, exist_ok=True)
        task_index = number % 3 if task_index is None else task_index
        command = [
            sys.executable,
            "-u",
            "-m",
            "examples.euler.medication_sorting.main",
            "--mode",
            "run",
            "--endpoint",
            endpoint,
            "--execute",
            "--record",
            "--object",
            "all",
            "--output",
            str(output),
            "--layout-mode",
            "current",
            "--xy-jitter",
            "0",
            "--yaw-jitter-deg",
            "0",
            "--task-index",
            str(task_index),
        ]
        print(f"开始采集 {output.name}，split={split}，task_index={task_index}。", flush=True)
        print("使用当前已保存布局，自动规划四件药品的抓放轨迹并同步录制。", flush=True)
        code = run_monitored(command, root, max_bytes, reserve_bytes)
        if code or not valid_episode(output):
            raise RuntimeError(
                f"采集失败或数据校验未通过：{output}（exit={code}）。"
                "原始文件已保留；先检查 result.json 和仿真状态，复位后再调整并保存布局。"
                "不要把失败记录作为成功示范。重新运行会创建新编号。"
            )
    print(f"采集及完整性校验通过：{output}", flush=True)
    print(
        "本条已结束。请先检查三路视频和入框结果，再复位场景，"
        "小幅调整药品 X/Y 位置或旋转角度并保存布局，然后重新运行同一条命令采集下一条。",
        flush=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "inspect", "run"), default="collect")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--task-index", type=int, choices=range(3))
    parser.add_argument("--max-gb", type=float, default=25)
    parser.add_argument("--reserve-gb", type=float, default=4)
    parser.add_argument("--episode-budget-gb", type=float, default=2.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--object", choices=("all", "calcium", "melatonin", "herbbox", "pharyngitis"), default="all")
    parser.add_argument("--layout-mode", choices=("current", "baseline"), default="current")
    parser.add_argument("--reference-layout", type=Path, help="Frozen initial_layout JSON for randomized trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xy-jitter", type=float, default=0)
    parser.add_argument("--yaw-jitter-deg", type=float, default=0)
    args = parser.parse_args()
    if args.mode == "collect":
        if not args.execute or args.dataset_root is None:
            parser.error("Collection requires --execute and --dataset-root")
        if (
            args.output
            or args.record
            or args.reference_layout
            or args.object != "all"
            or args.layout_mode != "current"
            or args.seed
            or args.xy_jitter
            or args.yaw_jitter_deg
        ):
            parser.error("Collection records all objects in the saved layout; use --mode run for trial options")
        if any(not math.isfinite(v) or v <= 0 for v in (args.reserve_gb, args.episode_budget_gb, args.max_gb)):
            parser.error("Capacity limits must be finite and positive")
    else:
        if args.output is None or args.dataset_root is not None:
            parser.error("inspect/run require --output, not --dataset-root")
        if args.mode == "run" and not args.execute:
            parser.error("Motion requires --execute")
        if args.mode == "inspect" and (args.execute or args.record):
            parser.error("Inspection does not execute or record")
        args.task_index = 0 if args.task_index is None else args.task_index
    try:
        if args.mode == "collect":
            collect_one(
                args.dataset_root,
                args.split,
                args.endpoint,
                args.task_index,
                int(args.reserve_gb * 1e9),
                int(args.episode_budget_gb * 1e9),
                max_bytes=int(args.max_gb * 1e9),
            )
        else:
            from .workflow import run_trial

            run_trial(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"未完成：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("已中断；已有文件保留，请检查并复位仿真后再运行。", file=sys.stderr, flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
