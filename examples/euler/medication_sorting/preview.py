"""Make a review MP4 from a validated three-camera expert episode."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    metadata = json.loads((args.episode / "metadata.json").read_text())
    if not metadata.get("raw_valid"):
        parser.error("Review video requires a successful, validated episode")
    if args.output.exists():
        parser.error("Output must be a new file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-n", "-filter_complex_threads", "2"]
    for name in ["head_cam", "camera_left", "camera_right"]:
        command += ["-i", str(args.episode / metadata["cameras"][name]["path"])]
    shapes = [metadata["cameras"][name]["native_shape"] for name in ["head_cam", "camera_left", "camera_right"]]
    if any(len(shape) != 3 or shape[2] != 3 or min(shape[:2]) <= 0 for shape in shapes):
        parser.error("Expected native RGB camera dimensions")
    height = max(shape[0] for shape in shapes)
    width = sum(shape[1] for shape in shapes)
    fps = float(metadata["fps"])
    if not math.isfinite(fps) or fps <= 0:
        parser.error("Recorded fps must be finite and positive")
    filters = []
    for i, label in enumerate(["Head", "Left wrist", "Right wrist"]):
        filters.append(
            f"[{i}:v]setpts=(PTS-STARTPTS)/{args.speed},setsar=1,pad=iw:{height}:0:0:black,"
            f"drawtext=text={label!r}:x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]"
        )
    filters += [
        f"[v0][v1][v2]hstack=inputs=3,pad={width}:{height + 32}:0:32:black,"
        f"drawtext=text='Expert motion planning | {args.speed:g}x speed':x=10:y=6:fontsize=20:fontcolor=white[out]",
    ]
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-an",
        "-r",
        f"{fps * args.speed:g}",
        "-vsync",
        "0",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    print(args.output)


if __name__ == "__main__":
    main()
