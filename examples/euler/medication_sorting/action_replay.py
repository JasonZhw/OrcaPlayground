"""Execute recorded actions through the original controller and measure tracking.

Initial layout is restored once; settle/alignment actions are replayed to recover
initial velocities. No recorded state is written during the action sequence.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import time
from pathlib import Path

import grpc
import numpy as np
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

from .action_pipeline import process_recorded_actions
from .collect_dataset import HOST_WIDE_COLLECTION_LOCK
from .execution import SortingEnv
from .replay import load_recording, model_signature
from .workflow import DESTINATIONS, Scene, placement_satisfied, prepare_model


class MeasuredEnv(SortingEnv):
    """Observe the public step boundary without altering the original controller."""

    def begin_measurement(self):
        self.torque_peak = np.zeros(self.model.nu)
        self.limit_hits = np.zeros(self.model.nu, dtype=int)
        self.microsteps = 0

    def do_simulation(self, ctrl, n_frames):
        super().do_simulation(ctrl, n_frames)
        if hasattr(self, "microsteps"):
            self.torque_peak = np.maximum(self.torque_peak, np.abs(self.ctrl).ravel())
            self.limit_hits += np.isclose(ctrl, self.limits[:, 0], atol=1e-9, rtol=0) | np.isclose(
                ctrl, self.limits[:, 1], atol=1e-9, rtol=0
            )
            self.microsteps += 1


def command_action(env, scene, action):
    action = np.asarray(action, dtype=float)
    if (
        action.shape != (16,)
        or not np.isfinite(action).all()
        or np.any(action[[7, 15]] < 0)
        or np.any(action[[7, 15]] > 1)
    ):
        raise ValueError("Expected 16 finite absolute targets and closures in [0,1]")
    command = env.command.copy()
    for side, offset in [("left", 0), ("right", 8)]:
        ids = [env.joint_names.index(n) for n in scene.arms[side].joint_names]
        command[ids] = action[offset : offset + 7]
        env.grippers.set(side, float(action[offset + 7]))
    env.step(command)


async def run(args, env):
    scene, reference, times, indices, validation = load_recording(args.episode)
    meta = json.loads((args.episode / "episode/metadata.json").read_text())
    if meta["control"]["arm_position_feedback_gain"] != 12:
        raise ValueError("This baseline requires the recorded gain 12 controller")
    trace = json.loads((args.episode / "trace.json").read_text())
    lookup = {r["index"]: i for i, r in enumerate(trace)}
    start = lookup[int(indices[0])]
    with np.load(args.episode / "episode/episode.npz", allow_pickle=False) as data:
        actions = data["action"].copy()
    if not np.array_equal(actions, np.array([trace[lookup[int(i)]]["action"] for i in indices])):
        raise ValueError("Raw actions differ from execution trace")
    args.output.mkdir(parents=True, exist_ok=False)
    channel = bridge = None
    report = dict(
        completed=False,
        mode="recorded_action_physics_replay",
        policy_used=False,
        physics_stepped=True,
        source=str(args.episode),
        gain_multiplier=12,
        scene_reset="initial layout once, then replay original settle/alignment actions",
        data_fingerprints=validation["source_sha256"],
        controller_sha256=hashlib.sha256(Path(__file__).with_name("execution.py").read_bytes()).hexdigest(),
        processing=args.processing,
        latency_steps=args.latency_steps if args.processing != "direct" else None,
    )
    before, after, torques, clipped = [], [], [], []
    arm_names = [n for side in ("left", "right") for n in scene.arms[side].joint_names]
    arm_ids = [env.joint_names.index(n) for n in arm_names]
    state_ids = []
    for side in ("left", "right"):
        state_ids.extend(scene.arms[side].qpos_ids)
        state_ids.extend(env.qids[env.grippers.indices[side]])
    count = len(actions) if args.frames is None else min(args.frames, len(actions))
    wall = time.monotonic()
    try:
        if args.render:
            channel = grpc.aio.insecure_channel(args.endpoint)
            bridge = OrcaStudioBridge(GrpcServiceStub(channel))
            source = await asyncio.wait_for(bridge.load_model_xml(), 20)
            live = Scene(prepare_model(source, args.output / "scene_before.xml"))
            if model_signature(live.m) != model_signature(scene.m):
                raise ValueError("Live scene differs from recording")
            await asyncio.wait_for(bridge.pause_simulation(), 20)
        q = env.data.qpos.ravel().copy()
        poses = json.loads((args.episode / "initial_layout.json").read_text())["poses"]
        for key, pose in poses.items():
            address = scene.free_joint_address(key)
            q[address : address + 7] = pose["position"] + pose["quaternion"]
        env.set_joint_qpos(q)
        env.set_joint_qvel(np.zeros(env.model.nv))
        env.mj_forward()
        env.step(env.command)
        for row in trace[:start]:
            command_action(env, scene, row["action"])
        report["initial_state_max_error_rad"] = float(
            np.max(abs(env.data.qpos.ravel()[state_ids] - reference[0, state_ids]))
        )
        if report["initial_state_max_error_rad"] > 1e-6:
            raise ValueError("Preroll failed to reproduce recorded starting joints")
        requested, executed = actions.copy(), actions.copy()
        if args.processing != "direct":
            limits, previous = [], []
            for side in ("left", "right"):
                binding = scene.arms[side]
                ids = [env.joint_names.index(n) for n in binding.joint_names]
                limits.extend(
                    np.c_[np.maximum(binding.lower, env.limits[ids, 0]), np.minimum(binding.upper, env.limits[ids, 1])]
                )
                limits.append([0, 1])
                previous.extend(env.command[ids])
                previous.append(env.grippers.closure[side])
            requested, executed, events = process_recorded_actions(
                actions, previous, limits, args.processing, args.latency_steps
            )
            (args.output / "queue_events.json").write_text(json.dumps(events, indent=2))
            report["processing_note"] = (
                "Oracle future actions; fixed simulation-time return latency; no image/network/model"
            )
            report["online_sha256"] = hashlib.sha256(
                Path(__file__).with_name("pi05").joinpath("online.py").read_bytes()
            ).hexdigest()
        report["max_requested_vs_recorded"] = float(np.max(abs(requested - actions)))
        report["max_executed_vs_recorded_per_dimension"] = np.max(abs(executed - actions), axis=0).tolist()
        env.begin_measurement()
        playback_start = time.monotonic()
        for i, action in enumerate(executed[:count]):
            before.append(env.data.qpos.ravel()[state_ids].copy())
            command_action(env, scene, action)
            after.append(env.data.qpos.ravel()[state_ids].copy())
            torques.append(np.asarray(env.ctrl).ravel().copy())
            clipped.append(env.last_control.copy())
            if bridge is not None:
                await asyncio.wait_for(
                    bridge.render(env.data.qpos.ravel().copy(), float(env.data.time), simulate_index=int(indices[i])),
                    20,
                )
                await asyncio.sleep(max(0, playback_start + (i + 1) * 0.02 - time.monotonic()))
            if i % 500 == 0 or i + 1 == count:
                progress = dict(steps=i + 1, total=count, wall_seconds=time.monotonic() - playback_start)
                (args.output / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        before, after = np.asarray(before), np.asarray(after)
        arm_columns = list(range(7)) + list(range(15, 22))
        targets = executed[:count, list(range(7)) + list(range(8, 15))]
        error = after[:, arm_columns] - targets
        state_difference = before - reference[:count, state_ids]
        joint_reports = []
        for j, name in enumerate(arm_names):
            joint_reports.append(
                dict(
                    joint=name,
                    rmse_rad=float(np.sqrt(np.mean(error[:, j] ** 2))),
                    p95_abs_rad=float(np.quantile(abs(error[:, j]), 0.95)),
                    max_abs_rad=float(np.max(abs(error[:, j]))),
                    peak_torque_nm=float(env.torque_peak[arm_ids[j]]),
                    control_limit_fraction=float(env.limit_hits[arm_ids[j]] / env.microsteps),
                )
            )
        scene.set_qpos(env.data.qpos.ravel().copy())
        report.update(
            completed=True,
            steps=count,
            recorded_frames=len(actions),
            wall_seconds=time.monotonic() - wall,
            simulation_seconds=count * 0.02,
            arm_post_step_tracking=joint_reports,
            reference_state_max_abs_error_rad=float(np.max(abs(state_difference))),
            reference_state_rmse_rad=float(np.sqrt(np.mean(state_difference**2))),
            placements={k: placement_satisfied(scene, k, d) for k, d in DESTINATIONS.items()},
            physics_microsteps=env.microsteps,
            actuator_peak_torque_nm=env.torque_peak.tolist(),
            actuator_control_limit_fraction=(env.limit_hits / env.microsteps).tolist(),
            actuator_joint_names=env.joint_names,
            torque_sampling="peak measured at each 2 ms physics step; timeseries sampled at 20 ms",
            gripper_note="Linkage/contact targets are not interpreted as independent joint position tracking",
        )
        np.savez_compressed(
            args.output / "tracking.npz",
            action=actions[:count],
            requested_action=requested[:count],
            executed_action=executed[:count],
            state_before=before,
            state_after=after,
            reference_state=reference[:count, state_ids],
            torque=np.array(torques),
            control=np.array(clipped),
            sim_time=times[:count],
        )
        np.save(args.output / "final_qpos.npy", env.data.qpos.ravel())
    except BaseException as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", steps=len(after), wall_seconds=time.monotonic() - wall)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        if channel is not None:
            await channel.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int)
    p.add_argument(
        "--processing", choices=("direct", "queue", "inference", "arm_only", "gripper_only"), default="direct"
    )
    p.add_argument("--latency-steps", type=int, default=11)
    p.add_argument("--render", action="store_true", help="Display simulated states in OrcaLab at 50 Hz")
    p.add_argument("--endpoint", default="127.0.0.1:50051")
    args = p.parse_args()
    if args.frames is not None and args.frames < 1:
        p.error("frames must be positive")
    args.episode = args.episode.resolve()
    with HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Euler initializes/closes its own loop; do so outside the renderer loop.
        env = MeasuredEnv(args.episode / "scene.xml")
        try:
            asyncio.run(run(args, env))
        finally:
            env.close()


if __name__ == "__main__":
    main()
