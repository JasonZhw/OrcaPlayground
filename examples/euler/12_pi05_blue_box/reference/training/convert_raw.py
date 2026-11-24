#!/usr/bin/env python
"""把松应原始采集记录（raw/episode_*/cameras.mp4 + trajectory.npz + result.json）转成 lerobot v3.0 数据集。
做法照对方 V7 包的 scripts/dual_arm_v4/convert.py：只收 result.json 里 success 的轨迹；录像 1280×480 是头部（左半）与右腕（右半）
两路 640×480 拼成，逐帧解码后各自用 openpi 的 resize_with_pad 缩到 224×224；录像帧数必须等于轨迹行数（每个动作前拍一对图），
50Hz；状态、动作取 trajectory.npz 的 state、action（各 9 维）。键名直接写成 lerobot 认的。
用法：python convert_raw.py <raw 目录（含 episode_*）> <输出目录>"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi_client.image_tools import resize_with_pad

TASK = "Pick up the blue Intel processor box and place it in the wooden basket."
W, H = 1280, 480
raw, out = Path(sys.argv[1]), Path(sys.argv[2])

img = {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}
vec = lambda n: {"dtype": "float32", "shape": (9,), "names": [n]}  # noqa: E731
ds = LeRobotDataset.create(
    repo_id=out.name, root=out, fps=50, robot_type="unitree_g1_fixed_torso_dual_arm", use_videos=False,
    features={"observation.images.image": img, "observation.images.wrist_image": img,
              "observation.state": vec("state"), "action": vec("action")},
)
kept, skipped = [], []
for folder in sorted(raw.glob("episode_*")):
    result = json.loads((folder / "result.json").read_text())
    if not result.get("success"):
        skipped.append(folder.name)
        continue
    with np.load(folder / "trajectory.npz", allow_pickle=False) as t:
        states, actions = t["state"].astype(np.float32), t["action"].astype(np.float32)
        times = t["time"]
    assert states.shape == actions.shape == (result["num_frames"], 9), folder
    assert np.isfinite(states).all() and np.isfinite(actions).all(), folder
    assert np.allclose(np.diff(times), 0.02, atol=1e-6), folder
    assert result["video_hz"] == 50 and result["control_hz"] == 50, folder
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(folder / "cameras.mp4"),
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
    for i in range(len(states)):
        payload = proc.stdout.read(W * H * 3)
        if len(payload) != W * H * 3:
            raise RuntimeError(f"录像比轨迹短：{folder} 第 {i} 帧")
        pair = np.frombuffer(payload, dtype=np.uint8).reshape(H, W, 3)
        ds.add_frame({"observation.images.image": resize_with_pad(pair[:, :640], 224, 224),
                      "observation.images.wrist_image": resize_with_pad(pair[:, 640:], 224, 224),
                      "observation.state": states[i], "action": actions[i], "task": TASK})
    if proc.stdout.read(1):
        raise RuntimeError(f"录像比轨迹长：{folder}")
    if proc.wait(timeout=60):
        raise RuntimeError(f"解码失败：{folder}")
    ds.save_episode()
    kept.append({"episode": folder.name, "frames": len(states)})
    print("CONVERTED", len(kept), folder.name, len(states), flush=True)
ds.finalize()
summary = {"episodes": len(kept), "frames": sum(k["frames"] for k in kept), "skipped_not_success": skipped, "kept": kept}
Path(f"{out}.summary.json").write_text(json.dumps(summary, indent=1))
print("CONVERSION_COMPLETE", summary["episodes"], summary["frames"], "skipped", len(skipped))
