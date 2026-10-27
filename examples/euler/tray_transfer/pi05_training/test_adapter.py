"""CPU-only contract tests; OpenPI, Torch, and model weights are not required."""

import unittest

import numpy as np
from adapter import TrayInputs, TrayOutputs


def sample():
    head = np.zeros((3, 224, 224), dtype=np.float32)
    head[0] = 1.0
    wrist = np.zeros((3, 224, 224), dtype=np.float32)
    wrist[1] = 1.0
    return {
        "observation.state": np.arange(15, dtype=np.float32),
        "observation.images.head": head,
        "observation.images.right_wrist": wrist,
        "action": np.arange(450, dtype=np.float32).reshape(50, 9),
        "prompt": "Place the light blue box on the tray.",
    }


class AdapterTests(unittest.TestCase):
    def test_preserves_measured_state_and_absolute_actions(self):
        raw = sample()
        adapted = TrayInputs()(raw)
        np.testing.assert_array_equal(adapted["state"], raw["observation.state"])
        np.testing.assert_array_equal(adapted["actions"], raw["action"])
        self.assertEqual(adapted["state"].shape, (15,))
        self.assertEqual(adapted["actions"].shape, (50, 9))
        self.assertEqual(adapted["prompt"], raw["prompt"])

    def test_camera_order_and_missing_left_mask(self):
        adapted = TrayInputs()(sample())
        self.assertEqual(adapted["image"]["base_0_rgb"].shape, (224, 224, 3))
        np.testing.assert_array_equal(adapted["image"]["base_0_rgb"][0, 0], [255, 0, 0])
        np.testing.assert_array_equal(adapted["image"]["right_wrist_0_rgb"][0, 0], [0, 255, 0])
        self.assertFalse(adapted["image"]["left_wrist_0_rgb"].any())
        self.assertEqual(
            adapted["image_mask"],
            {"base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": True},
        )

    def test_inference_accepts_hwc_uint8_without_action(self):
        raw = sample()
        raw.pop("action")
        raw["observation.images.head"] = np.full((224, 224, 3), 127, dtype=np.uint8)
        adapted = TrayInputs()(raw)
        self.assertNotIn("actions", adapted)
        self.assertTrue((adapted["image"]["base_0_rgb"] == 127).all())

    def test_rejects_wrong_state_action_or_horizon(self):
        for key, shape in [("observation.state", (9,)), ("action", (50, 7)), ("action", (10, 9))]:
            with self.subTest(key=key, shape=shape):
                raw = sample()
                raw[key] = np.zeros(shape, dtype=np.float32)
                with self.assertRaisesRegex(ValueError, key.replace(".", r"\.")):
                    TrayInputs()(raw)

    def test_rejects_bad_images(self):
        for image in [
            np.zeros((3, 64, 64), dtype=np.float32),
            np.full((3, 224, 224), 255.0, dtype=np.float32),
            np.full((3, 224, 224), np.nan, dtype=np.float32),
        ]:
            with self.subTest(shape=image.shape):
                raw = sample()
                raw["observation.images.head"] = image
                with self.assertRaisesRegex(ValueError, "observation.images.head"):
                    TrayInputs()(raw)

    def test_rejects_nonfinite_state_and_missing_prompt(self):
        raw = sample()
        raw["observation.state"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            TrayInputs()(raw)
        raw = sample()
        raw.pop("prompt")
        with self.assertRaisesRegex(ValueError, "prompt"):
            TrayInputs()(raw)

    def test_inference_keeps_nine_absolute_targets(self):
        padded = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
        actions = TrayOutputs()({"actions": padded})["actions"]
        self.assertEqual(actions.shape, (50, 9))
        np.testing.assert_array_equal(actions, padded[:, :9])

    def test_rejects_wrong_model_output_width(self):
        with self.assertRaisesRegex(ValueError, "actions"):
            TrayOutputs()({"actions": np.zeros((50, 7), dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
