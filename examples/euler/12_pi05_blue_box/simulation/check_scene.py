"""Capture initial synchronized RGB for manual scene acceptance; no policy actions."""

import argparse
import fcntl
import json
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_metadata import sha256
from camera_capture import Cameras
from environment import DualArmEnv
from task_config import PROJECT, ROOT
from build_model import OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if not OUTPUT.is_file():
        parser.error("Run prepare-scene first")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "simulation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (7070, 7072):
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Camera port {port} is already in use")
        env, cameras = None, None
        try:
            env = DualArmEnv()
            env.restore()
            cameras = Cameras(env)
            images = cameras.capture()
            args.output.mkdir(parents=True)
            report = {
                "scene_sha256": sha256(OUTPUT),
                "layout_sha256": sha256(PROJECT / "assets/scene_v9.json"),
                "target_position": env.observe()["target"].tolist(),
                "cameras": {}, "manual_visual_review": "required",
                "visual_resource_contents_verified": False,
                "scope": "Valid RGB and scene mapping do not prove matching textures or object appearance.",
            }
            valid = True
            for name, image in images.items():
                good = (image.shape == (480, 640, 3) and image.dtype == np.uint8
                        and float(image.std()) >= 2)
                valid &= good
                report["cameras"][name] = {
                    "shape": list(image.shape), "dtype": str(image.dtype),
                    "std": float(image.std()), "valid_rgb": bool(good),
                }
                if image.dtype == np.uint8 and image.shape == (480, 640, 3):
                    Image.fromarray(image).save(args.output / f"{name}.png")
            report["rgb_checks_passed"] = bool(valid)
            (args.output / "scene_check.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8",
            )
            print(json.dumps(report, indent=2))
            if not valid:
                raise SystemExit(1)
        finally:
            try:
                if cameras:
                    cameras.close()
            finally:
                if env:
                    env.close()


if __name__ == "__main__":
    main()
