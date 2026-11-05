"""Local-only NPZ inference transport; run in the isolated trained JAX environment."""

import argparse
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from .adapter import MedicationInputs
from .guards import digest
from .run import parser, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    r = a.run_root.resolve()
    args = parser().parse_args(
        [
            "--openpi-dir",
            str(r / "openpi"),
            "--data-home",
            "/root/gpufree-data/datasets/lerobot",
            "--assets-dir",
            str(r / "assets"),
            "--checkpoint-dir",
            str(r / "checkpoints"),
            "--base-params",
            str(r / "base/pi05_base/params"),
            "infer",
            "--checkpoint",
            str(a.checkpoint),
            "--output",
            str(r / "unused_online.npy"),
        ]
    )
    config, _ = prepare(args)
    norm = a.checkpoint / "assets/local/g1_med_sort_v2/norm_stats.json"
    if digest(norm) != digest(config.assets_dirs / "local/g1_med_sort_v2/norm_stats.json"):
        raise ValueError("Checkpoint normalization differs from verified training assets")
    from openpi.policies.policy_config import create_trained_policy

    policy = create_trained_policy(config, a.checkpoint)
    metadata = dict(
        checkpoint=str(a.checkpoint.resolve()),
        norm_sha256=digest(norm),
        horizon=50,
        action_dim=16,
        fps=50,
        closure="medication_closure_v1",
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps(metadata).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if self.path != "/infer" or not 0 < size < 4_000_000:
                    raise ValueError("Invalid request")
                self.connection.settimeout(120)
                with np.load(io.BytesIO(self.rfile.read(size)), allow_pickle=False) as data:
                    raw = {key: data[key] for key in data.files}
                raw["prompt"] = str(raw["prompt"].item())
                MedicationInputs()(raw)
                started = time.monotonic()
                actions = np.asarray(policy.infer(raw)["actions"])
                if actions.shape != (50, 16) or not np.isfinite(actions).all():
                    raise ValueError("Invalid model prediction")
                seconds = time.monotonic() - started
                out = io.BytesIO()
                np.savez(out, actions=actions, infer_seconds=seconds)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(out.getvalue())
                print(json.dumps(dict(infer_seconds=seconds, finite=True)), flush=True)
            except Exception as exc:
                self.send_error(400, str(exc))
                print(f"INFERENCE_FAILED {type(exc).__name__}: {exc}", flush=True)

    print(json.dumps(metadata), flush=True)
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
