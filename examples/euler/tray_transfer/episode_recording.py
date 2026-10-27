"""Lossless, pre-step episode storage without simulator dependencies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

TASK_TEXTS = (
    "Pick up the smallest box on the table, which is light blue, and place it in the wooden tray on the left.",
    "Place the light blue box, the smallest one on the table, into the wooden tray on the left.",
    "Move the smallest box on the table, the light blue one, into the wooden tray on the left.",
)
CAMERAS = ("head", "right_wrist")
TOLERANCE = 1e-7


def _vector(value, size, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def decode_action(action9):
    """Expand absolute arm targets and (a, d) into all 15 nominal targets."""
    action = _vector(action9, 9, "action9")
    a, d = action[7:]
    return np.concatenate((action[:7], [0, 0, a, d, 0, 0, -a, d]))


def encode_action(arm7, gripper8):
    """Compress only commands that obey the current mirrored gripper mapping."""
    arm = _vector(arm7, 7, "arm7")
    gripper = _vector(gripper8, 8, "gripper8")
    action = np.concatenate((arm, gripper[[2, 3]]))
    if not np.allclose(decode_action(action)[7:], gripper, atol=TOLERANCE, rtol=0):
        raise ValueError("gripper8 does not match [0, 0, a, d, 0, 0, -a, d]")
    return action


class EpisodeRecorder:
    """Store caller-supplied observation_t/action_t pairs; never step a simulator."""

    def __init__(self, directory, task, arm_joint_names, gripper_joint_names,
                 fps=50.0, metadata=None):
        if task not in TASK_TEXTS:
            raise ValueError("task must be one of TASK_TEXTS")
        self.arm_names, self.gripper_names = list(arm_joint_names), list(gripper_joint_names)
        names = self.arm_names + self.gripper_names
        if (len(self.arm_names) != 7 or len(self.gripper_names) != 8
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != 15):
            raise ValueError("expected 7 arm and 8 gripper joint names, all unique")
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.extra_metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        self.directory, self.task = Path(directory), task
        self.directory.mkdir(parents=True, exist_ok=False)
        for camera in CAMERAS:
            (self.directory / "frames" / camera).mkdir(parents=True)
        self.rows = []
        self.finished = False

    def add(self, state15, action9, raw_action15, images, sim_time, phase,
            simulate_index, camera_indices, requested_action15=None):
        if self.finished:
            raise RuntimeError("episode has already finished")
        state = _vector(state15, 15, "state15")
        action = _vector(action9, 9, "action9")
        raw = _vector(raw_action15, 15, "raw_action15")
        requested = _vector(raw if requested_action15 is None else requested_action15,
                            15, "requested_action15")
        if not np.allclose(decode_action(action), raw, atol=TOLERANCE, rtol=0):
            raise ValueError("action9 does not reconstruct raw_action15")
        time = float(sim_time)
        if not np.isfinite(time):
            raise ValueError("sim_time must be finite")
        if self.rows and not np.isclose(time - self.rows[-1][4], 1 / self.fps,
                                       atol=TOLERANCE, rtol=0):
            raise ValueError("sim_time must advance by exactly 1/fps")
        if (isinstance(simulate_index, (bool, np.bool_))
                or not isinstance(simulate_index, (int, np.integer)) or simulate_index < 0):
            raise ValueError("simulate_index must be a nonnegative integer")
        if self.rows and simulate_index <= self.rows[-1][6]:
            raise ValueError("simulate_index must increase")
        if set(camera_indices) != set(CAMERAS) or any(
                isinstance(camera_indices[c], (bool, np.bool_))
                or not isinstance(camera_indices[c], (int, np.integer))
                or camera_indices[c] != simulate_index for c in CAMERAS):
            raise ValueError("both camera indices must equal simulate_index")
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a nonempty string")
        if set(images) != set(CAMERAS):
            raise ValueError("images must contain exactly head and right_wrist")
        for camera in CAMERAS:
            pixels = np.asarray(images[camera])
            if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8:
                raise ValueError(f"{camera} must be 224x224 RGB uint8")
            if float(pixels.std()) < 1.0:
                raise ValueError(f"{camera} image is black or nearly uniform")
        paths = [self.directory / "frames" / c / f"{len(self.rows):06d}.png" for c in CAMERAS]
        try:
            for camera, path in zip(CAMERAS, paths):
                Image.fromarray(images[camera]).save(path, format="PNG")
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            raise
        self.rows.append((state, action, raw, requested, time, phase, int(simulate_index),
                          [int(camera_indices[c]) for c in CAMERAS], requested_action15 is not None))

    def finish(self, success: bool, result: dict) -> dict:
        if self.finished:
            raise RuntimeError("episode has already finished")
        if not isinstance(success, bool) or not isinstance(result, dict):
            raise ValueError("success must be bool and result must be dict")
        if success and not self.rows:
            raise ValueError("an empty episode cannot succeed")
        count = len(self.rows)
        arrays = {name: np.asarray([row[i] for row in self.rows], dtype=np.float64).reshape(count, size)
                  for i, (name, size) in enumerate(
                      (("state", 15), ("action", 9), ("raw_action", 15), ("requested_action", 15)))}
        arrays.update(sim_time=np.asarray([r[4] for r in self.rows], dtype=np.float64),
                      frame_index=np.arange(count, dtype=np.int64),
                      phase=np.asarray([r[5] for r in self.rows], dtype=str),
                      simulate_index=np.asarray([r[6] for r in self.rows], dtype=np.int64),
                      camera_indices=np.asarray([r[7] for r in self.rows], dtype=np.int64).reshape(count, 2),
                      requested_action_provided=np.asarray([r[8] for r in self.rows], dtype=bool))
        native_names = self.extra_metadata.get("camera_names", {"head": "head_cam", "right_wrist": "camera_right"})
        metadata = dict(self.extra_metadata)
        metadata.update(
            task=self.task, success=success, complete=success, training_ready=False, result=result,
            frame_layout_version=1, fps=self.fps, frame_count=count,
            arm_joint_names=self.arm_names, gripper_joint_names=self.gripper_names,
            state_names=self.arm_names + self.gripper_names,
            action_names=self.arm_names + ["gripper_a", "gripper_d"],
            state_dim=15, action_dim=9, raw_action_dim=15, units="radians",
            state_description="Measured arm7 + gripper8; not reconstructed from commands",
            action_description="Absolute arm7 targets + a,d; gripper=[0,0,a,d,0,0,-a,d]",
            requested_action_default="raw_action when requested_action_provided is false",
            cameras={c: {"native_name": native_names.get(c), "path": f"frames/{c}/{{frame_index:06d}}.png",
                         "encoding": "PNG", "color_space": "RGB", "dtype": "uint8", "shape": [224, 224, 3],
                         "frame_count": count} for c in CAMERAS},
            camera_order=list(CAMERAS),
            synchronization={"alignment": "observation_t -> action_t (caller supplies pre-step observations)",
                             "camera_indices_equal_simulate_index": True, "regular_sim_time": True,
                             "time_tolerance_seconds": TOLERANCE, "dt_seconds": 1 / self.fps},
            observed_ranges={name: {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
                             if count else {"min": None, "max": None}
                             for name, values in arrays.items() if name in ("state", "action", "raw_action")},
            range_description="Observed episode values, not actuator safety limits",
        )
        serialized = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
        np.savez_compressed(self.directory / "episode.npz", **arrays)
        (self.directory / "metadata.json").write_text(serialized + "\n", encoding="utf-8")
        self.finished = True
        return metadata
