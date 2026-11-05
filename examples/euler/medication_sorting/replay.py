"""Replay successful recorded measured states in OrcaLab, without physics or a policy.

The preceding trace row's post-step qpos is observation_t. Using the current
row's qpos_after would shift the scene by one control tick. All 30 recorded
robot joints and timestamps must agree before any rendering is allowed.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import re
import time
from pathlib import Path

import grpc
import numpy as np
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

from .collect_dataset import HOST_WIDE_COLLECTION_LOCK
from .execution import Grippers
from .recording import STATE_NAMES
from .workflow import Scene, prepare_model


def load_recording(directory):
    metadata = json.loads((directory / "episode/metadata.json").read_text())
    if not all(metadata.get(k) is True for k in ("success", "complete", "raw_valid")):
        raise ValueError("Only complete, validated successful recordings can be replayed")
    if metadata.get("state_names") != list(STATE_NAMES) or metadata.get("fps") != 50:
        raise ValueError("Unexpected joint layout or recording frequency")
    scene = Scene(directory / "scene.xml")
    names = metadata["actuator_joint_names"]
    grippers = Grippers(names)
    joint_ids = []
    for side in ("left", "right"):
        if list(scene.arms[side].joint_names) != metadata["arms"][side]:
            raise ValueError("Recorded arm joint names do not match the scene")
        joint_ids.extend(scene.arms[side].qpos_ids)
        joint_ids.extend(int(scene.m.jnt_qposadr[scene.m.joint(names[i]).id]) for i in grippers.indices[side])
    trace = json.loads((directory / "trace.json").read_text())
    lookup = {row["index"]: i for i, row in enumerate(trace)}
    if len(lookup) != len(trace):
        raise ValueError("Duplicate trace indices")
    with np.load(directory / "episode/episode.npz", allow_pickle=False) as data:
        states, times = data["state"].copy(), data["sim_time"].copy()
        indices = data["frame_index"].copy()
    if states.shape != (metadata["frame_count"], 30) or len(times) != len(states):
        raise ValueError("Recording length/shape mismatch")
    positions = [lookup[int(i)] for i in indices]
    if any(i < 1 for i in positions):
        raise ValueError("Missing preceding full scene state")
    qpos = np.array([trace[i - 1]["qpos_after"] for i in positions])
    prior_times = np.array([trace[i - 1]["time"] for i in positions])
    trace_states = np.array([trace[i]["state"] for i in positions])
    if qpos.shape != (len(states), scene.m.nq) or not np.isfinite(qpos).all():
        raise ValueError("Invalid full scene state")
    if not np.isfinite(states).all() or not np.isfinite(times).all():
        raise ValueError("Nonfinite recording")
    if not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-8):
        raise ValueError("Recording is not a continuous 50 Hz sequence")
    state_error = float(np.max(abs(qpos[:, joint_ids] - states)))
    time_error = float(np.max(abs(prior_times - times)))
    if state_error > 1e-10 or time_error > 1e-8 or not np.array_equal(states, trace_states):
        raise ValueError("Recorded states and full scene trace are misaligned")
    report = dict(
        mode="recorded_state_playback",
        source=str(directory),
        frames=len(states),
        duration_seconds=float(times[-1] - times[0]),
        fps=50,
        measured_joint_count=30,
        max_joint_alignment_error_rad=state_error,
        max_time_alignment_error_seconds=time_error,
        policy_used=False,
        physics_stepped=False,
        task_success_reevaluated=False,
        objects="recorded poses, not newly simulated contacts",
    )
    report["source_sha256"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("episode/episode.npz", "episode/metadata.json", "trace.json", "scene.xml")
    }
    return scene, qpos, times, indices, report


def model_signature(model):
    # Studio regenerates UUID suffixes when exporting object joints. Keep the
    # stable name, owning body, address, type, axis and anchor in the contract.
    return (
        model.nq,
        [
            (
                re.sub(r"_[0-9A-Fa-f]{8}(?:_[0-9A-Fa-f]{4}){3}_[0-9A-Fa-f]{12}$", "", model.joint(i).name),
                model.body(int(model.jnt_bodyid[i])).name,
                int(model.jnt_type[i]),
                int(model.jnt_qposadr[i]),
                tuple(model.jnt_axis[i]),
                tuple(model.jnt_pos[i]),
            )
            for i in range(model.njnt)
        ],
    )


async def play(args, scene, qpos, times, indices, report):
    channel = grpc.aio.insecure_channel(args.endpoint)
    bridge = OrcaStudioBridge(GrpcServiceStub(channel))
    count = len(qpos) if args.frames is None else min(args.frames, len(qpos))
    rendered = 0
    started = None
    try:
        source = await asyncio.wait_for(bridge.load_model_xml(), 20)
        current = Scene(prepare_model(source, args.output / "scene_before.xml"))
        if model_signature(current.m) != model_signature(scene.m):
            raise ValueError("Live scene joint names/order differ from the recorded scene")
        await asyncio.wait_for(bridge.pause_simulation(), 20)
        started = time.monotonic()
        for i in range(count):
            target = started + float(times[i] - times[0]) / args.speed
            await asyncio.sleep(max(0, target - time.monotonic()))
            await asyncio.wait_for(bridge.render(qpos[i], float(times[i]), simulate_index=int(indices[i])), 20)
            rendered += 1
            if i % 250 == 0 or rendered == count:
                status = dict(rendered_frames=rendered, requested_frames=count, wall_seconds=time.monotonic() - started)
                (args.output / "progress.json").write_text(json.dumps(status, indent=2))
                print(json.dumps(status), flush=True)
        report["completed"] = True
    except BaseException as exc:
        report.update(completed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(
            rendered_frames=rendered,
            requested_frames=count,
            speed=args.speed,
            wall_seconds=None if started is None else time.monotonic() - started,
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        await channel.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True, help="Raw episode root containing trace.json and episode/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endpoint", default="127.0.0.1:50051")
    p.add_argument("--speed", type=float, default=1)
    p.add_argument("--frames", type=int, help="Optional prefix length for a short check")
    p.add_argument("--check-only", action="store_true", help="Validate locally without connecting to OrcaLab")
    args = p.parse_args()
    if not np.isfinite(args.speed) or not 0 < args.speed <= 4 or (args.frames is not None and args.frames < 1):
        p.error("Speed must be in (0,4], frames must be positive")
    args.episode = args.episode.resolve()
    scene, qpos, times, indices, report = load_recording(args.episode)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    if args.check_only:
        print(json.dumps(report, indent=2))
        return
    with HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(play(args, scene, qpos, times, indices, report))


if __name__ == "__main__":
    main()
