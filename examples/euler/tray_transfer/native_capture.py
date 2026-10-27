"""Capture indexed native head/right RGB while Euler physics remains paused."""

import time

import av
import numpy as np
from PIL import Image, ImageOps


class NativeCameraPair:
    """Use public camera APIs and decode the exact requested IDR packets."""

    camera_names = {"head": "head_cam", "right_wrist": "camera_right"}

    def __init__(self, env):
        self.env = env
        self.manager = env.get_recorder_manager()
        self.original = {}
        self.recorders = {}
        self.last_index = None
        self.frame_metadata = []
        if not set(self.camera_names.values()) <= set(env.get_camera_names()):
            raise ValueError("Current scene must expose head_cam and camera_right")
        try:
            for (key, name), port in zip(self.camera_names.items(), (7070, 7072)):
                current = self.manager.get_camera_properties(name)
                prop = current
                self.original[name] = {
                    field: getattr(prop, field) for field in (
                        "capture_rgb", "capture_depth", "capture_normal",
                        "capture_object_color", "color_port",
                    )
                }
                env.start_streaming(
                    name, capture_rgb=True, capture_depth=False,
                    capture_normal=False, capture_object_color=False,
                    color_port=port, max_buffer_frames=32,
                )
                self.recorders[key] = self.manager.get_recorder(name, "color")
                if self.recorders[key] is None:
                    raise RuntimeError(f"No receiver for {name}")
            self.properties = {
                key: {field: getattr(self.manager.get_camera_properties(name), field)
                      for field in ("width", "height", "vertical_fov", "near_clip", "far_clip", "gamma")}
                for key, name in self.camera_names.items()
            }
        except Exception:
            self.close()
            raise

    def capture(self, simulate_index, timeout=15.0):
        if not 0 <= simulate_index <= 2**31 - 1:
            raise ValueError("Camera index must fit nonnegative int32")
        if self.last_index is not None and simulate_index <= self.last_index:
            raise ValueError("Camera index must increase")
        if any(rec.get_frame(simulate_index) is not None for rec in self.recorders.values()):
            raise RuntimeError("Requested camera index already exists; refusing stale frame")
        self.env.render(simulate_index=simulate_index, request_idr=True)
        entries = {}
        deadline = time.monotonic() + timeout
        while len(entries) < 2:
            for key, recorder in self.recorders.items():
                if key in entries:
                    continue
                entry = recorder.get_frame(simulate_index)
                if entry is not None:
                    if entry.simulate_index != simulate_index or not entry.is_keyframe:
                        raise RuntimeError(f"{key}: mismatched index or missing requested keyframe")
                    entries[key] = entry
            if len(entries) < 2:
                if time.monotonic() >= deadline:
                    missing = set(self.recorders) - entries.keys()
                    raise TimeoutError(f"Missing native RGB at {simulate_index}: {missing}")
                time.sleep(0.005)
        images, metadata = {}, {}
        for key, entry in entries.items():
            decoder = av.CodecContext.create("h264", "r")
            frames = list(decoder.decode(av.Packet(entry.nal_data)))
            frames.extend(decoder.decode(None))
            if len(frames) != 1:
                raise RuntimeError(f"{key}: exact packet decoded {len(frames)} frames")
            rgb = frames[0].to_ndarray(format="rgb24")
            if float(rgb.std()) < 1.0:
                raise RuntimeError(f"{key}: native camera returned a blank frame")
            # Preserve native aspect ratio/FOV; pad, never crop the task scene.
            images[key] = np.asarray(ImageOps.pad(
                Image.fromarray(rgb), (224, 224), method=Image.Resampling.BILINEAR, color=(0, 0, 0),
            )).copy()
            metadata[key] = {
                "simulate_index": entry.simulate_index, "timestamp_ns": entry.timestamp_ns,
                "is_keyframe": entry.is_keyframe, "native_shape": list(rgb.shape),
            }
        self.last_index = simulate_index
        self.frame_metadata.append(metadata)
        return images, metadata

    def close(self):
        errors = []
        for name, original in self.original.items():
            try:
                self.manager.stop_recorder(name)
                self.manager.set_streaming_enabled(name, False)
                self.manager.set_camera_properties(name, **original)
                self.manager.remove_recorder(name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        self.original.clear()
        if errors:
            print("Camera cleanup: " + "; ".join(errors), flush=True)


def record_control_step(env, control, arm_indices, gripper_indices, recorder, cameras, phase):
    """Record observation_t and the latched nominal action, then execute it."""
    from .episode_recording import encode_action

    indices = np.array([*arm_indices, *gripper_indices])
    state = env.data.qpos.ravel()[env.actuator_qpos_indices[indices]].copy()
    sim_time = float(env.data.time)
    effective = env.right_gripper_hold.apply(control)
    action = encode_action(effective[arm_indices], effective[gripper_indices])
    simulate_index = 1_000_000 + len(recorder.rows)
    images, frame_info = cameras.capture(simulate_index)
    if float(env.data.time) != sim_time or not np.array_equal(
        env.data.qpos.ravel()[env.actuator_qpos_indices[indices]], state,
    ):
        raise RuntimeError("Physics/state changed during camera capture")
    recorder.add(
        state, action, effective[indices], images, sim_time, phase, simulate_index,
        {key: value["simulate_index"] for key, value in frame_info.items()},
        requested_action15=np.asarray(control)[indices],
    )
    env.step(control)
    if not np.array_equal(env.last_command, effective):
        raise RuntimeError("Executed nominal command differs from recorded action")
