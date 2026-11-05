"""Run smoke, reload, formal ACT training and final reload, with an exclusive lock."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--verified-smoke", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        smoke = json.loads(args.verified_smoke.read_text())
        assert smoke["passed"] and smoke["shape"] == [50, 16]
        assert smoke["finite"] and Path(smoke["checkpoint"]).is_dir()
        assert json.loads((root / "smoke10_v1_metrics.jsonl").read_text().splitlines()[-1])["step"] == 10
        write_json(root / "process.json", {"pid": os.getpid(), "started_unix": time.time()})
        stages = [
            ("training", ["train", "--steps", "20000"]),
            ("final_reload", ["reload", "--checkpoint",
             str(root / "act20k_v1/checkpoints/020000/pretrained_model"),
             "--report", str(root / "final_reload.json")]),
        ]
        try:
            for phase, extra in stages:
                cmd = [sys.executable, "-u", "-m", "examples.euler.medication_sorting.act.run",
                       *extra, "--dataset", str(args.dataset), "--output", str(root / "act20k_v1")]
                with (root / f"{phase}.log").open("x") as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    write_json(root / "status.json", {"phase": phase, "pid": os.getpid(),
                               "child_pid": child.pid, "command": cmd, "time_unix": time.time()})
                    code = child.wait()
                (root / f"{phase}.exitcode").write_text(f"{code}\n")
                if code:
                    raise RuntimeError(f"{phase} failed: exit {code}; inspect its log before recovery")
            assert json.loads((root / "final_reload.json").read_text())["passed"]
            write_json(root / "status.json", {"phase": "complete", "steps": 20000,
                                              "time_unix": time.time(), "simulation_tested": False})
            (root / "pipeline.exitcode").write_text("0\n")
        except Exception as error:
            write_json(root / "status.json", {"phase": "blocked", "error": str(error),
                                              "time_unix": time.time()})
            (root / "pipeline.exitcode").write_text("1\n")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
