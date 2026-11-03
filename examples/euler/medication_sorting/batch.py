"""Sequential, auditable randomized trials; every episode gets a fresh reset."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--xy-jitter", type=float, default=0.005)
    parser.add_argument("--yaw-jitter-deg", type=float, default=3.0)
    parser.add_argument("--layout-mode", choices=["baseline", "current"], default="baseline")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue to the next reset after a failed trial")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Batch execution requires --execute")
    if args.episodes < 1 or args.start_seed < 0:
        parser.error("episodes must be positive and start-seed nonnegative")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i in range(args.episodes):
        seed = args.start_seed + i
        out = args.output / f"episode_{i:04d}_seed_{seed}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "examples.euler.medication_sorting.main",
            "--mode",
            "run",
            "--execute",
            "--object",
            "all",
            "--output",
            str(out),
            "--layout-mode",
            args.layout_mode,
            "--seed",
            str(seed),
            "--xy-jitter",
            str(args.xy_jitter),
            "--yaw-jitter-deg",
            str(args.yaw_jitter_deg),
            "--task-index",
            str(i % 3),
        ]
        if args.record:
            command.append("--record")
        print(f"TRIAL {i + 1}/{args.episodes}: seed={seed}, output={out}", flush=True)
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[3], check=False)
        result_file = out / "result.json"
        report = (
            json.loads(result_file.read_text())
            if result_file.exists()
            else {"success": False, "reason": "No result file"}
        )
        if args.record and not report.get("recording", {}).get("raw_valid", False):
            report["success"] = False
            report.setdefault("reason", "Missing or invalid recorded episode")
        rows.append({"seed": seed, "directory": str(out), "exitcode": result.returncode, "result": report})
        (args.output / "batch.json").write_text(json.dumps(rows, indent=2))
        if (result.returncode or not report["success"]) and not args.keep_going:
            break
    if any(row["exitcode"] or not row["result"]["success"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
