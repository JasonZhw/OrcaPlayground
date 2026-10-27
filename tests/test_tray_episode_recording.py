"""Real file tests for the small standalone episode writer."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from examples.euler.tray_transfer.episode_recording import TASK_TEXTS, EpisodeRecorder, decode_action, encode_action


class EpisodeRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "episode"
        self.arm = np.arange(7, dtype=float) / 10
        self.gripper = np.array([0, 0, 0.28, 1.35, 0, 0, -0.28, 1.35])
        self.action = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.28, 1.35])
        self.raw = np.concatenate([self.arm, self.gripper])
        # Actual measured gripper positions need not obey the target mapping.
        self.state = np.arange(15, dtype=float) / 100
        pixels = np.broadcast_to(np.arange(224, dtype=np.uint8)[None, :, None], (224, 224, 3)).copy()
        self.images = {"head": pixels, "right_wrist": pixels[:, ::-1].copy()}

    def recorder(self, **kwargs):
        return EpisodeRecorder(self.directory, TASK_TEXTS[0], [f"arm_{i}" for i in range(7)],
                               [f"gripper_{i}" for i in range(8)], **kwargs)

    def add(self, recorder, **kwargs):
        args = dict(state15=self.state, action9=self.action, raw_action15=self.raw,
                    images=self.images, sim_time=4.0, phase="approach", simulate_index=200,
                    camera_indices={"head": 200, "right_wrist": 200})
        args.update(kwargs)
        recorder.add(**args)

    def test_action_round_trip_preserves_release_and_hold_targets(self):
        for a, d in [(0.0, -1.0), (0.28, 0.9), (0.28, 1.35)]:
            gripper = np.array([0, 0, a, d, 0, 0, -a, d])
            encoded = encode_action(self.arm, gripper)
            np.testing.assert_allclose(encoded, [0, .1, .2, .3, .4, .5, .6, a, d])
            np.testing.assert_array_equal(decode_action(encoded), np.concatenate([self.arm, gripper]))

    def test_action_encoding_rejects_nonmirrored_or_nonfinite_targets(self):
        for index, value in [(0, .01), (6, -.27), (7, 1.2), (3, float("nan"))]:
            gripper = self.gripper.copy()
            gripper[index] = value
            with self.assertRaises(ValueError):
                encode_action(self.arm, gripper)
        with self.assertRaises(ValueError):
            decode_action([0] * 8 + [float("inf")])

    def test_records_provided_prestep_values_and_lossless_pngs_incrementally(self):
        recorder = self.recorder(metadata={"layout_id": "current", "success": "untrusted"})
        expected_state = self.state.copy()
        expected_head = self.images["head"].copy()
        requested = self.raw.copy()
        requested[0] = .125
        self.add(recorder, requested_action15=requested)
        head_path = self.directory / "frames/head/000000.png"
        self.assertTrue(head_path.exists(), "PNG must be persisted before finish")
        self.state[:] = 99
        self.action[:] = 99
        requested[:] = 99
        self.images["head"][:] = 0
        metadata = recorder.finish(True, {"placed": True})
        with np.load(self.directory / "episode.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["state"][0], expected_state)
            np.testing.assert_array_equal(data["raw_action"][0], self.raw)
            np.testing.assert_allclose(data["action"][0], [0, .1, .2, .3, .4, .5, .6, .28, 1.35])
            self.assertEqual(data["requested_action"][0, 0], .125)
            self.assertEqual(data["phase"].tolist(), ["approach"])
            self.assertEqual(data["simulate_index"].tolist(), [200])
            self.assertEqual(data["camera_indices"].tolist(), [[200, 200]])
            self.assertEqual(data["frame_index"].tolist(), [0])
        with Image.open(head_path) as saved:
            self.assertEqual(saved.mode, "RGB")
            np.testing.assert_array_equal(np.asarray(saved), expected_head)
        self.assertTrue(metadata["complete"])
        self.assertIs(metadata["success"], True)
        self.assertFalse(metadata["training_ready"])
        self.assertEqual(metadata["frame_count"], 1)
        self.assertEqual(metadata["task"], TASK_TEXTS[0])

    def test_camera_mismatch_is_rejected_before_writing_a_frame(self):
        recorder = self.recorder()
        with self.assertRaises(ValueError):
            self.add(recorder, camera_indices={"head": 199, "right_wrist": 200})
        self.assertEqual(list(self.directory.glob("frames/*/*.png")), [])
        self.assertEqual(recorder.finish(False, {"reason": "camera lag"})["frame_count"], 0)

    def test_time_must_advance_regularly_and_static_images_are_allowed(self):
        recorder = self.recorder()
        self.add(recorder)
        for invalid_time in [4.0, 3.98, 4.04, float("nan")]:
            with self.assertRaises(ValueError):
                self.add(recorder, sim_time=invalid_time, simulate_index=201,
                         camera_indices={"head": 201, "right_wrist": 201})
        self.add(recorder, sim_time=4.02, simulate_index=201,
                 camera_indices={"head": 201, "right_wrist": 201})
        self.assertEqual(recorder.finish(False, {"reason": "test stop"})["frame_count"], 2)

    def test_rejects_invalid_images_and_mismatched_raw_actions(self):
        recorder = self.recorder()
        for invalid in [np.zeros((224, 224, 3), np.uint8), self.images["head"].astype(float),
                        self.images["head"][:100]]:
            with self.assertRaises(ValueError):
                self.add(recorder, images={"head": invalid, "right_wrist": self.images["right_wrist"]})
        wrong_raw = self.raw.copy()
        wrong_raw[1] += .01
        with self.assertRaises(ValueError):
            self.add(recorder, raw_action15=wrong_raw)
        self.assertEqual(list(self.directory.glob("frames/*/*.png")), [])

    def test_failed_finish_preserves_rows_and_explicitly_marks_incomplete(self):
        recorder = self.recorder()
        self.add(recorder)
        recorder.finish(False, {"reason": "wrong object"})
        metadata = json.loads((self.directory / "metadata.json").read_text())
        self.assertFalse(metadata["success"])
        self.assertFalse(metadata["complete"])
        self.assertFalse(metadata["training_ready"])
        self.assertEqual(metadata["result"], {"reason": "wrong object"})
        with np.load(self.directory / "episode.npz", allow_pickle=False) as data:
            self.assertEqual(data["state"].shape, (1, 15))
            self.assertEqual(data["requested_action_provided"].tolist(), [False])
        with self.assertRaises(RuntimeError):
            self.add(recorder)

    def test_empty_failure_has_valid_array_shapes_and_never_overwrites(self):
        recorder = self.recorder()
        with self.assertRaises(FileExistsError):
            self.recorder()
        self.assertFalse(recorder.finish(False, {"reason": "initialization"})["complete"])
        with np.load(self.directory / "episode.npz", allow_pickle=False) as data:
            self.assertEqual(data["state"].shape, (0, 15))
            self.assertEqual(data["action"].shape, (0, 9))
            self.assertEqual(data["raw_action"].shape, (0, 15))


if __name__ == "__main__":
    unittest.main()
