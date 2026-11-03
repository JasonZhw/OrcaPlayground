"""Indexed native-camera raw episodes; no simulation or LeRobot conversion.

For every row the caller must freeze physics, snapshot state/action, await
``bridge.render(qpos, sim_time, simulate_index=index, request_idr=True)``, call
``append(..., index, ...)``, then execute the recorded action. Never render the
same index twice. Camera wall-clock timestamps are retained separately from
simulation time; equality between those clock domains is not assumed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

SCALAR_GRIPPER_VERSION = "medication_closure_v1"
CLOSURE_KNOTS = np.array([0.0, 0.4, 0.7, 1.0])
AD_KNOTS = np.array([[0.0, -1.0], [0.0, 0.97], [0.28, 0.9], [0.28, 1.45]])


def gripper_mapping():
    return {"version": SCALAR_GRIPPER_VERSION, "closure_knots": CLOSURE_KNOTS.tolist(), "ad_knots": AD_KNOTS.tolist()}


RAW_FORMAT = "medication_sorting_raw_v2"

TASK_TEXTS = (
    "Use the left arm to place medicine bottles in the purple bin on the left, "
    "and the right arm to place medicine boxes in the blue bin on the right.",
    "Sort the medicines: move bottles to the left purple bin with the left arm "
    "and boxes to the right blue bin with the right arm.",
    "Pick up each medicine bottle with the left arm and put it in the left purple bin; "
    "use the right arm to put each medicine box in the right blue bin.",
)
CAMERAS = ("head_cam", "camera_left", "camera_right")
STATE_NAMES = tuple(
    f"{side}_{kind}_{i}"
    for side in ("left", "right")
    for kind, count in (("arm", 7), ("gripper", 8))
    for i in range(count)
)
ACTION_NAMES = tuple(
    name for side in ("left", "right") for name in (*(f"{side}_arm_{i}" for i in range(7)), f"{side}_gripper_closure")
)


def _vector(value, size, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _receiver(camera_name, color_port, max_buffer_frames):
    from orca_gym.recorder.video_recorder import VideoRecorder

    return VideoRecorder(camera_name, color_port, max_buffer_frames)


class RawEpisodeRecorder:
    """Synchronous recorder for a bridge owned by the caller's asyncio loop.

    The loop may be stopped, or running on a different thread; calling from its
    own running thread is rejected to avoid deadlock. Native sockets are local.
    Videos contain lossless RGB decodes at native dimensions, not resized crops.
    Raw validity is distinct from training readiness: conversion is still needed.
    """

    def __init__(self, directory, task=TASK_TEXTS[0], fps=50.0, metadata=None, timeout=15.0, receiver_factory=None):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty English task instruction")
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("fps and timeout must be finite and positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "videos").mkdir()
        self.task, self.fps, self.timeout = task, float(fps), float(timeout)
        self.extra = json.loads(json.dumps(metadata or {}, allow_nan=False))
        self.receiver_factory = receiver_factory or _receiver
        self.bridge = self.loop = None
        self.original, self.receivers, self.properties = {}, {}, {}
        self.writers, self.streams, self.shapes = {}, {}, {}
        self.camera_rows = {name: [] for name in CAMERAS}
        self.rows, self.errors = [], []
        self.started = self.closed = False
        self.result = None

    def _run(self, coroutine):
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is self.loop:
            coroutine.close()
            raise RuntimeError("Call synchronous recorder outside the bridge loop thread")
        if self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            try:
                return future.result(self.timeout)
            except TimeoutError:
                future.cancel()
                raise
        return self.loop.run_until_complete(asyncio.wait_for(coroutine, self.timeout))

    def start(self, bridge, loop):
        if self.started or self.closed:
            raise RuntimeError("Recorder already started or closed")
        self.bridge, self.loop = bridge, loop
        try:
            names = self._run(bridge.get_camera_names())
            missing = set(CAMERAS) - set(names)
            if missing:
                raise ValueError(f"Missing native cameras: {sorted(missing)}")
            for name, port in zip(CAMERAS, (7070, 7074, 7072)):
                prop = self._run(bridge.get_camera_properties(name))
                self.original[name] = {
                    field: getattr(prop, field)
                    for field in (
                        "capture_rgb",
                        "capture_depth",
                        "capture_normal",
                        "capture_object_color",
                        "color_port",
                        "streaming_enabled",
                    )
                }
                self.properties[name] = {
                    field: getattr(prop, field)
                    for field in ("width", "height", "vertical_fov", "near_clip", "far_clip", "gamma")
                }
                self._run(bridge.set_streaming_enabled(name, False))
                self._run(
                    bridge.set_camera_properties(
                        name,
                        capture_rgb=True,
                        capture_depth=False,
                        capture_normal=False,
                        capture_object_color=False,
                        color_port=port,
                    )
                )
                self._run(bridge.set_streaming_enabled(name, True))
                receiver = self.receiver_factory(name, port, 32)
                self.receivers[name] = receiver
                receiver.start()
            self.started = True
        except Exception as exc:
            self.errors.append(f"camera_start: {exc}")
            self._restore()
            raise

    def _matched_frames(self, index):
        import av

        entries = {}
        deadline = time.monotonic() + self.timeout
        while len(entries) < len(CAMERAS):
            for name, receiver in self.receivers.items():
                if name not in entries:
                    entry = receiver.get_frame(index)
                    if entry is not None:
                        entries[name] = entry
            if len(entries) < len(CAMERAS):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Missing indexed RGB {index}: {sorted(set(CAMERAS) - entries.keys())}")
                time.sleep(0.005)
        images = {}
        for name, entry in entries.items():
            if entry.simulate_index != index or not entry.is_keyframe:
                raise ValueError(f"{name}: wrong frame index or missing requested IDR")
            stamp = entry.timestamp_ns
            if (
                isinstance(stamp, bool)
                or not isinstance(stamp, (int, np.integer))
                or not 0 <= stamp <= np.iinfo(np.int64).max
            ):
                raise ValueError(f"{name}: invalid native timestamp")
            if self.camera_rows[name] and stamp <= self.camera_rows[name][-1]["timestamp_ns"]:
                raise ValueError(f"{name}: native timestamp did not increase")
            decoder = av.CodecContext.create("h264", "r")
            frames = list(decoder.decode(av.Packet(entry.nal_data))) + list(decoder.decode(None))
            if len(frames) != 1:
                raise ValueError(f"{name}: indexed packet decoded {len(frames)} frames")
            rgb = frames[0].to_ndarray(format="rgb24")
            prop = self.properties[name]
            if rgb.shape != (prop["height"], prop["width"], 3):
                raise ValueError(f"{name}: native video dimensions differ from camera properties")
            if float(rgb.std()) < 1.0:
                raise ValueError(f"{name}: blank or nearly uniform camera frame")
            images[name] = rgb
        return images, entries

    def append(self, state30, action16, sim_time, frame_index, stage):
        if not self.started or self.closed:
            raise RuntimeError("Recorder must be started and not closed")
        try:
            if len(self.rows) % 50 == 0 and shutil.disk_usage(self.directory).free < 1_000_000_000:
                raise OSError("Less than 1 GB free; stopping recording before the disk fills")
            state, action = _vector(state30, 30, "state30"), _vector(action16, 16, "action16")
            if np.any((action[[7, 15]] < 0) | (action[[7, 15]] > 1)):
                raise ValueError("Gripper closure must be in [0, 1]")
            sim_time = float(sim_time)
            if not np.isfinite(sim_time) or sim_time < 0:
                raise ValueError("sim_time must be finite and nonnegative")
            if (
                isinstance(frame_index, (bool, np.bool_))
                or not isinstance(frame_index, (int, np.integer))
                or not 0 <= frame_index <= 2**31 - 1
            ):
                raise ValueError("frame_index must be a nonnegative int32")
            if not isinstance(stage, str) or not stage:
                raise ValueError("stage must be a nonempty string")
            if self.rows:
                if frame_index <= self.rows[-1][3]:
                    raise ValueError("frame_index must increase; never reuse a render index")
                if not np.isclose(sim_time - self.rows[-1][2], 1 / self.fps, atol=1e-7, rtol=0):
                    raise ValueError("sim_time must advance by 1/fps")
            images, entries = self._matched_frames(int(frame_index))
            # Validate every camera before writing any of the row. A later IO
            # failure still invalidates the episode and is checked at close.
            for name in CAMERAS:
                self._write_frame(name, images[name], len(self.rows))
                self.camera_rows[name].append(
                    {
                        "row_index": len(self.rows),
                        "frame_index": int(frame_index),
                        "timestamp_ns": int(entries[name].timestamp_ns),
                        "sim_time": sim_time,
                        "is_keyframe": True,
                        "rgb_sha256": hashlib.sha256(images[name].tobytes()).hexdigest(),
                    }
                )
            self.rows.append((state, action, sim_time, int(frame_index), stage))
        except Exception as exc:
            self.errors.append(f"append: {exc}")
            raise

    def _write_frame(self, name, rgb, row_index):
        import av

        if name not in self.writers:
            writer = av.open(str(self.directory / "videos" / f"{name}.mkv"), "w")
            self.writers[name] = writer
            stream = writer.add_stream("ffv1", rate=Fraction(str(self.fps)))
            stream.width, stream.height = rgb.shape[1], rgb.shape[0]
            stream.pix_fmt = "bgr0"
            self.streams[name], self.shapes[name] = stream, list(rgb.shape)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = row_index
        frame.time_base = Fraction(str(1 / self.fps)).limit_denominator(1_000_000)
        for packet in self.streams[name].encode(frame):
            self.writers[name].mux(packet)

    def _restore(self):
        for name, receiver in self.receivers.items():
            try:
                receiver.stop()
            except Exception as exc:
                self.errors.append(f"receiver_stop {name}: {exc}")
        self.receivers.clear()
        for name, original in self.original.items():
            try:
                self._run(self.bridge.set_streaming_enabled(name, False))
                self._run(
                    self.bridge.set_camera_properties(
                        name, **{k: v for k, v in original.items() if k != "streaming_enabled"}
                    )
                )
                self._run(self.bridge.set_streaming_enabled(name, original["streaming_enabled"]))
            except Exception as exc:
                self.errors.append(f"camera_restore {name}: {exc}")
        self.original.clear()

    def close(self, success, reason):
        """Finalize even failed runs. The caller supplies physical task success."""
        import av

        if self.closed:
            return self.result
        if not isinstance(success, bool) or not isinstance(reason, str):
            raise ValueError("success must be bool and reason must be str")
        for name, writer in self.writers.items():
            try:
                for packet in self.streams[name].encode(None):
                    writer.mux(packet)
            except Exception as exc:
                self.errors.append(f"video_flush {name}: {exc}")
            finally:
                try:
                    writer.close()
                except Exception as exc:
                    self.errors.append(f"video_close {name}: {exc}")
        self._restore()
        count, cameras = len(self.rows), {}
        if not count:
            self.errors.append("empty_episode")
        for name in CAMERAS:
            video_path = self.directory / "videos" / f"{name}.mkv"
            decoded_count = 0
            try:
                with av.open(str(video_path)) as container:
                    for frame in container.decode(video=0):
                        if [frame.height, frame.width, 3] != self.shapes[name]:
                            raise ValueError("resolution changed")
                        if decoded_count >= len(self.camera_rows[name]):
                            raise ValueError("video/robot/frame-metadata count mismatch")
                        digest = hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest()
                        if digest != self.camera_rows[name][decoded_count]["rgb_sha256"]:
                            raise ValueError("video pixels differ from indexed native RGB")
                        decoded_count += 1
                if decoded_count != count or len(self.camera_rows[name]) != count:
                    raise ValueError("video/robot/frame-metadata count mismatch")
            except Exception as exc:
                self.errors.append(f"video_validate {name}: {exc}")
            sidecar = f"videos/{name}.frames.json"
            (self.directory / sidecar).write_text(json.dumps(self.camera_rows[name], indent=2) + "\n")
            cameras[name] = dict(
                path=f"videos/{name}.mkv",
                frame_metadata=sidecar,
                decoded_frame_count=decoded_count,
                native_shape=self.shapes.get(name),
                properties=self.properties.get(name),
                encoding="FFV1 lossless RGB decode of native H264",
                resized=False,
            )
        arrays = {
            "state": np.asarray([r[0] for r in self.rows], dtype=np.float64).reshape(count, 30),
            "action": np.asarray([r[1] for r in self.rows], dtype=np.float64).reshape(count, 16),
            "sim_time": np.asarray([r[2] for r in self.rows], dtype=np.float64),
            "frame_index": np.asarray([r[3] for r in self.rows], dtype=np.int64),
            "stage": np.asarray([r[4] for r in self.rows], dtype=str),
        }
        np.savez_compressed(self.directory / "episode.npz", **arrays)
        self.result = dict(self.extra)
        self.result.update(
            format=RAW_FORMAT,
            task=self.task,
            task_variants=list(TASK_TEXTS) if self.task in TASK_TEXTS else [self.task],
            success=success,
            reason=reason,
            complete=success and bool(count) and not self.errors,
            raw_valid=success and bool(count) and not self.errors,
            training_ready=False,
            training_readiness_reason="Raw episode requires reviewed dataset conversion",
            frame_count=count,
            fps=self.fps,
            state_dim=30,
            action_dim=16,
            state_names=list(STATE_NAMES),
            action_names=list(ACTION_NAMES),
            units={"state": "radians", "arm_action": "radians", "gripper_action": "unitless closure [0,1]"},
            state_layout="left_arm7,left_measured_gripper8,right_arm7,right_measured_gripper8",
            action_layout="left_arm7,left_closure,right_arm7,right_closure",
            gripper_mapping=gripper_mapping(),
            cameras=cameras,
            camera_order=list(CAMERAS),
            errors=list(self.errors),
            synchronization={
                "alignment": "pre-step observation_t paired with action_t",
                "method": "exact simulate_index lookup; requested IDR decoded independently",
                "timestamp_clock": "native camera header; separate from sim_time",
                "video_time": "constant control cadence; original camera timestamps in sidecars",
                "physics_frozen_during_capture": "caller obligation",
                "latest_frame_fallback": False,
            },
        )
        (self.directory / "metadata.json").write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        self.closed = True
        return self.result
