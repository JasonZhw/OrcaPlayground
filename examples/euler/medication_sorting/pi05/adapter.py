"""The same native dual-arm adapter is used for training and inference."""

from dataclasses import dataclass

import numpy as np

STATE_DIM = 30
ACTION_DIM = 16
MODEL_DIM = 32
ACTION_HORIZON = 50
CAMERAS = {"head": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}
STATE_NAMES = [
    f"{side}_{kind}_{i}"
    for side in ("left", "right")
    for kind, size in [("arm", 7), ("gripper", 8)]
    for i in range(size)
]
ACTION_NAMES = [
    n for side in ("left", "right") for n in [*[f"{side}_arm_{i}" for i in range(7)], f"{side}_gripper_closure"]
]


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite {shape}, got {array.shape}")
    return array


def rgb_image(value, name):
    image = np.asarray(value)
    if image.shape == (3, 480, 640):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (480, 640, 3):
        raise ValueError(f"{name}: expected native 640x480 RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"{name}: floating pixels must be finite in [0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{name}: expected uint8 or normalized floating pixels")
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class MedicationInputs:
    def __call__(self, data):
        state = finite_array(data["observation.state"], (STATE_DIM,), "state")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty original task prompt is required")
        out = {
            "state": state,
            "image": {slot: rgb_image(data[f"observation.images.{cam}"], cam) for cam, slot in CAMERAS.items()},
            "image_mask": {slot: np.True_ for slot in CAMERAS.values()},
            "prompt": prompt,
        }
        if "action" in data:
            out["actions"] = finite_array(data["action"], (ACTION_HORIZON, ACTION_DIM), "action")
        return out


@dataclass(frozen=True)
class MedicationOutputs:
    """Applied after unnormalization; validation/control limits belong to executor."""

    def __call__(self, data):
        actions = finite_array(data["actions"], (ACTION_HORIZON, MODEL_DIM), "actions")
        return {"actions": actions[:, :ACTION_DIM]}


def validate_info(info, report):
    """Reject incompatible datasets before OpenPI starts a loader or a model."""
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 50:
        raise ValueError("Expected LeRobot v2.1 at 50Hz")
    if report.get("passed") is not True:
        raise ValueError("Formal export validation must pass")
    for key in ("total_episodes", "total_frames"):
        if not isinstance(info.get(key), int) or info[key] <= 0 or report.get(key) != info[key]:
            raise ValueError(f"Validation and dataset disagree on {key}")
    for key, want in [("state_dim", 30), ("action_dim", 16), ("fps", 50), ("crop", None), ("resize", None)]:
        if key not in report or report[key] != want:
            raise ValueError(f"Wrong validation contract: {key}")
    features = info.get("features", {})
    for key, names in [("observation.state", STATE_NAMES), ("action", ACTION_NAMES)]:
        f = features.get(key, {})
        if f.get("shape") != [len(names)] or f.get("names") != names or f.get("dtype") != "float32":
            raise ValueError(f"Wrong joint order/shape/dtype: {key}")
    for cam in CAMERAS:
        f = features.get(f"observation.images.{cam}", {})
        if f.get("shape") != [480, 640, 3] or f.get("dtype") != "video":
            raise ValueError(f"Wrong camera contract: {cam}")
