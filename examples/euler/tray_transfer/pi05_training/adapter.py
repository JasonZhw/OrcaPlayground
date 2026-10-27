"""NumPy-only adapter shared by OpenPI training and offline inference."""

from dataclasses import dataclass

import numpy as np

STATE_DIM = 15
ACTION_DIM = 9
MODEL_DIM = 32
ACTION_HORIZON = 50


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values; got {array.shape}")
    return array


def rgb_image(value, name):
    """Accept LeRobot CHW floats in [0,1] or an inference RGB uint8 image."""
    image = np.asarray(value)
    if image.shape == (3, 224, 224):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (224, 224, 3):
        raise ValueError(f"{name} must be 224x224 RGB, HWC or CHW; got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"{name} float pixels must be finite and in [0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{name} must contain uint8 or float pixels; got {image.dtype}")
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class TrayInputs:
    """Retain measured state15 and absolute action9; do not compute deltas."""

    def __call__(self, data):
        state = finite_array(data["observation.state"], (STATE_DIM,), "observation.state")
        head = rgb_image(data["observation.images.head"], "observation.images.head")
        wrist = rgb_image(data["observation.images.right_wrist"], "observation.images.right_wrist")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty task string")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": np.zeros_like(head),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": prompt,
        }
        if "action" in data:
            inputs["actions"] = finite_array(data["action"], (ACTION_HORIZON, ACTION_DIM), "action")
        return inputs


@dataclass(frozen=True)
class TrayOutputs:
    """Run after OpenPI unnormalization; return arm7 + gripper(a,d)."""

    def __call__(self, data):
        actions = finite_array(data["actions"], (ACTION_HORIZON, MODEL_DIM), "actions")
        return {"actions": actions[:, :ACTION_DIM]}
