"""Bounded Euler simulation using live native images and the medication policy service.

No demonstration collection, training, or physical robot connection. Reports are
evaluation diagnostics. Rendering is synchronous; image decode/inference overlaps
execution. This is latency-trimmed chunking, not RTC or guaranteed real-time control.
"""

import argparse
import asyncio
import fcntl
import io
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import grpc
import numpy as np
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

from ..collect_dataset import HOST_WIDE_COLLECTION_LOCK
from ..execution import Executor, SortingEnv
from ..recording import CAMERAS, TASK_TEXTS
from ..workflow import DESTINATIONS, Scene, placement_satisfied, prepare_model
from .fixed_bins import FixedBinRenderBridge, prepare_fixed_bins
from .online import ActionConditioner, ActionQueue, bounded_closures, state_vector


def receiver_factory(name, port, size):
    from orca_gym.recorder.video_recorder import VideoRecorder

    return VideoRecorder(name, port, size)


class NativeCameras:
    def __init__(self, bridge, loop, factory=receiver_factory):
        self.bridge, self.loop = bridge, loop
        self.factory = factory
        self.original, self.receivers, self.timestamps = {}, {}, {}

    def call(self, coroutine):
        return self.loop.run_until_complete(asyncio.wait_for(coroutine, 20))

    def start(self):
        if not set(CAMERAS) <= set(self.call(self.bridge.get_camera_names())):
            raise ValueError("Three native medication cameras required")
        for name, port in zip(CAMERAS, (7070, 7074, 7072)):
            prop = self.call(self.bridge.get_camera_properties(name))
            if (prop.width, prop.height) != (640, 480):
                raise ValueError(f"{name}: native resolution changed")
            self.original[name] = {
                k: getattr(prop, k)
                for k in (
                    "capture_rgb",
                    "capture_depth",
                    "capture_normal",
                    "capture_object_color",
                    "color_port",
                    "streaming_enabled",
                )
            }
            self.call(self.bridge.set_streaming_enabled(name, False))
            self.call(
                self.bridge.set_camera_properties(
                    name,
                    capture_rgb=True,
                    capture_depth=False,
                    capture_normal=False,
                    capture_object_color=False,
                    color_port=port,
                )
            )
            receiver = self.factory(name, port, 32)
            self.receivers[name] = receiver
            self.call(self.bridge.set_streaming_enabled(name, True))
            receiver.start()

    def frames(self, index):
        """May run in a worker: exact indexed IDR only, never latest-frame fallback."""
        frames, stamps = {}, {}
        deadline = time.monotonic() + 10
        for name, receiver in self.receivers.items():
            entry = None
            while entry is None and time.monotonic() < deadline:
                entry = receiver.get_frame(index)
                if entry is None:
                    time.sleep(0.002)
            if entry is None or entry.simulate_index != index or not entry.is_keyframe:
                raise RuntimeError(f"{name}: missing indexed IDR {index}")
            stamp = int(entry.timestamp_ns)
            if stamp <= self.timestamps.get(name, -1):
                raise ValueError("Stale camera timestamp")
            decoder = av.CodecContext.create("h264", "r")
            decoded = list(decoder.decode(av.Packet(entry.nal_data))) + list(decoder.decode(None))
            if len(decoded) != 1:
                raise ValueError("Expected one decoded IDR")
            rgb = decoded[0].to_ndarray(format="rgb24")
            if rgb.shape != (480, 640, 3) or rgb.std() < 1:
                raise ValueError(f"{name}: invalid/blank native image")
            frames[name], stamps[name] = rgb, stamp
        self.timestamps.update(stamps)
        return frames, stamps

    def close(self):
        errors = []
        for receiver in self.receivers.values():
            try:
                receiver.stop()
            except Exception as exc:
                errors.append(str(exc))
        for name, original in self.original.items():
            try:
                self.call(self.bridge.set_streaming_enabled(name, False))
                fields = {k: v for k, v in original.items() if k != "streaming_enabled"}
                self.call(self.bridge.set_camera_properties(name, **fields))
                self.call(self.bridge.set_streaming_enabled(name, original["streaming_enabled"]))
            except Exception as exc:
                errors.append(str(exc))
        return errors


def predict(url, cameras, index, state, task, output, captured):
    started = time.monotonic()
    images, stamps = cameras.frames(index)
    # Render RPC acknowledgment precedes GPU readback. Do not overwrite its
    # state/index until every camera has delivered the requested frame.
    captured.set()
    raw = {"observation.state": state, "prompt": np.asarray(task)}
    for native, slot in zip(CAMERAS, ("head", "left_wrist", "right_wrist")):
        raw[f"observation.images.{slot}"] = images[native]
    payload = io.BytesIO()
    np.savez(payload, **raw)
    request = urllib.request.Request(url + "/infer", payload.getvalue(), method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        with np.load(io.BytesIO(response.read()), allow_pickle=False) as result:
            actions, infer_seconds = result["actions"].copy(), float(result["infer_seconds"])
    np.save(output / f"prediction_{index}.npy", actions)
    if not (output / "initial_images.npz").exists():
        np.savez_compressed(output / "initial_images.npz", **images, state=state)
    return actions, dict(
        index=index,
        camera_timestamps_ns=stamps,
        infer_seconds=infer_seconds,
        capture_and_infer_seconds=time.monotonic() - started,
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(args.policy_url, timeout=5) as response:
        metadata = json.load(response)
    if (metadata.get("horizon"), metadata.get("action_dim"), metadata.get("fps"), metadata.get("closure")) != (
        50,
        16,
        50,
        "medication_closure_v1",
    ) or metadata.get("checkpoint") != str(args.checkpoint):
        raise ValueError("Policy endpoint contract/checkpoint mismatch")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    channel = grpc.aio.insecure_channel(args.endpoint, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)])
    bridge = OrcaStudioBridge(GrpcServiceStub(channel))
    render_bridge = bridge
    bin_mapping = None
    cameras = NativeCameras(bridge, loop)
    pool = ThreadPoolExecutor(max_workers=1)
    captured = threading.Event()
    future = None
    env = None
    trace, predictions = [], []
    report = dict(
        completed=False,
        policy=metadata,
        requested_steps=args.steps,
        model_steps=0,
        task_success_verified=False,
        dataset_collected=False,
        real_robot=False,
        blend_steps=args.blend_steps,
        max_delta=args.max_delta,
        request_at_remaining=args.request_at_remaining,
        bin_physics="static_world_fixtures" if args.fixed_bins else "original_free_bodies",
        closure_policy="prediction_saturation_within_0.1; execution_uses_separate_open_close_slew_limits",
        conditioning=dict(
            arm_tau_seconds=args.arm_smoothing_seconds,
            close_speed_per_second=args.closure_close_speed,
            open_speed_per_second=args.closure_open_speed,
            timestep_seconds=0.02,
            raw_and_executed_arm_delta_guard=args.max_delta,
        ),
    )
    try:
        source = cameras.call(bridge.load_model_xml())
        model_path = prepare_model(source, args.output / ("scene_original.xml" if args.fixed_bins else "scene.xml"))
        scene = Scene(model_path)
        if args.fixed_bins:
            bin_mapping = prepare_fixed_bins(
                model_path, args.output / "scene.xml", [scene.names[k] for k in ("left_bin", "right_bin")]
            )
            render_bridge = FixedBinRenderBridge(bridge, bin_mapping)
            scene = Scene(args.output / "scene.xml")
            report["fixed_bins"] = bin_mapping.bins
            report["qpos_dimensions"] = dict(physics=bin_mapping.physics_nq, renderer=bin_mapping.source_nq)
        env = SortingEnv(args.output / "scene.xml", endpoint=args.endpoint)
        cameras.call(bridge.pause_simulation())
        initial = Executor(env, scene, render_bridge, loop, args.output)
        initial.hold(75, "initial_settle")
        initial.align_initial_command()
        arms = [scene.arms[a].qpos_ids for a in ("left", "right")]
        grips = [env.qids[env.grippers.indices[a]] for a in ("left", "right")]
        servo_ids = [[env.joint_names.index(n) for n in scene.arms[a].joint_names] for a in ("left", "right")]
        bounds, previous = [], []
        for arm, ids in zip(("left", "right"), servo_ids):
            binding = scene.arms[arm]
            bounds.extend(
                np.c_[np.maximum(binding.lower, env.limits[ids, 0]), np.minimum(binding.upper, env.limits[ids, 1])]
            )
            bounds.append([0, 1])
            previous.extend(env.command[ids])
            previous.append(env.grippers.closure[arm])
        bounds, previous = np.asarray(bounds), np.asarray(previous)
        conditioner = ActionConditioner(
            previous,
            bounds,
            max_arm_delta=args.max_delta,
            dt=0.02,
            arm_tau=args.arm_smoothing_seconds,
            close_speed=args.closure_close_speed,
            open_speed=args.closure_open_speed,
        )
        np.save(args.output / "initial_qpos.npy", env.data.qpos.ravel())
        cameras.start()
        for warm in range(12):
            cameras.call(
                render_bridge.render(
                    env.data.qpos.ravel().copy(), float(env.data.time), simulate_index=1000 + warm, request_idr=True
                )
            )
            time.sleep(0.08)

        def submit(step, index):
            q = env.data.qpos.ravel().copy()
            state = state_vector(q, arms, grips)
            captured.clear()
            cameras.call(render_bridge.render(q, float(env.data.time), simulate_index=index, request_idr=True))
            return pool.submit(
                predict, args.policy_url, cameras, index, state, TASK_TEXTS[args.task_index], args.output, captured
            )

        # Compile/warm the real camera pipeline while simulation time is frozen.
        for index in (2000, 2001):
            _, metric = submit(0, index).result(timeout=130)
            metric["warmup"] = True
            predictions.append(metric)
        future = submit(0, 3000)
        actions, metric = future.result(timeout=130)
        actions, metric["closure_values_saturated"] = bounded_closures(actions)
        metric.update(observed_step=0, received_step=0, discarded=0)
        predictions.append(metric)
        queue = ActionQueue()
        queue.replace(actions, 0, 0)
        future = None
        observed = 0
        started = time.monotonic()
        for step in range(args.steps):
            tick = time.monotonic()
            if future is not None and future.done():
                actions, metric = future.result()
                actions, metric["closure_values_saturated"] = bounded_closures(actions)
                age = queue.replace(actions, observed, step, blend_steps=args.blend_steps)
                metric.update(observed_step=observed, received_step=step, discarded=age)
                predictions.append(metric)
                future = None
            requested = queue.pop(step)
            try:
                action = conditioner.step(requested)
            except ValueError:
                report["rejected_action"] = dict(step=step, requested=requested.tolist(), previous=previous.tolist())
                raise
            before = env.data.qpos.ravel().copy()
            command = env.command.copy()
            for side, arm, ids in zip((0, 8), ("left", "right"), servo_ids):
                command[ids] = action[side : side + 7]
                env.grippers.set(arm, action[side + 7])
            env.step(command)
            previous = action
            report["model_steps"] = step + 1
            if (step + 1) % 250 == 0:
                progress = dict(
                    model_steps=step + 1,
                    sim_seconds=(step + 1) * 0.02,
                    wall_seconds=time.monotonic() - started,
                    queue_remaining=queue.remaining(step + 1),
                )
                (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
            if step + 1 < args.steps and future is None and queue.remaining(step + 1) <= args.request_at_remaining:
                observed = step + 1
                future = submit(observed, 3000 + observed)
            elif step % 2 == 0 and captured.is_set():
                cameras.call(
                    render_bridge.render(
                        env.data.qpos.ravel().copy(), float(env.data.time), simulate_index=100000 + step
                    )
                )
            elapsed = time.monotonic() - tick
            trace.append(
                dict(
                    step=step,
                    sim_time=float(env.data.time),
                    wall_time=time.monotonic() - started,
                    action=action.tolist(),
                    requested_action=requested.tolist(),
                    state=state_vector(before, arms, grips).tolist(),
                    state_after=state_vector(env.data.qpos.ravel(), arms, grips).tolist(),
                    arm_servo_control=env.last_control[np.asarray(servo_ids).ravel()].tolist(),
                    tick_seconds=elapsed,
                    queue_remaining=queue.remaining(step + 1),
                )
            )
            time.sleep(max(0.0, 0.02 - elapsed))
        report.update(completed=True, wall_seconds=time.monotonic() - started)
        scene.set_qpos(env.data.qpos.ravel().copy())
        report["placements"] = {key: placement_satisfied(scene, key, dest) for key, dest in DESTINATIONS.items()}
        report["all_four_in_bins"] = all(report["placements"].values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if future is not None:
            try:
                _, pending_metric = future.result()
                report["pending_request_not_executed"] = pending_metric
            except Exception as exc:
                report["pending_request_error"] = f"{type(exc).__name__}: {exc}"
        report["camera_restore_errors"] = cameras.close()
        if env is not None:
            np.save(args.output / "final_qpos.npy", env.data.qpos.ravel())
            if bin_mapping is not None:
                np.save(args.output / "final_render_qpos.npy", bin_mapping.expand(env.data.qpos.ravel()))
                scene.set_qpos(env.data.qpos.ravel().copy())
                report["final_bin_positions"] = {
                    k: scene.d.xpos[scene.bodies[k]].tolist() for k in ("left_bin", "right_bin")
                }
            cameras.call(render_bridge.render(env.data.qpos.ravel().copy(), float(env.data.time)))
            env.close()
        (args.output / "trace.json").write_text(json.dumps(trace))
        (args.output / "predictions.json").write_text(json.dumps(predictions, indent=2))
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        loop.run_until_complete(channel.close())
        loop.close()
        print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="127.0.0.1:50051")
    p.add_argument("--policy-url", default="http://127.0.0.1:8765")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--request-at-remaining", type=int, default=30)
    p.add_argument("--max-delta", type=float, default=0.08)
    p.add_argument("--blend-steps", type=int, choices=range(0, 21), default=10)
    p.add_argument("--task-index", type=int, choices=range(3), default=0)
    p.add_argument("--arm-smoothing-seconds", type=float, default=0.06)
    p.add_argument("--closure-close-speed", type=float, default=0.625)
    p.add_argument("--closure-open-speed", type=float, default=1.6)
    p.add_argument(
        "--fixed-bins", action="store_true", help="Evaluate with two static collidable bins; preserve source scene"
    )
    args = p.parse_args()
    if not 1 <= args.steps <= 15000 or not 1 <= args.request_at_remaining <= 49:
        p.error("Bounded 1..15000 steps and 1..49 queued steps required")
    with HOST_WIDE_COLLECTION_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
