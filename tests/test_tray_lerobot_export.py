"""Source-contract tests; no simulator or LeRobot installation required."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.euler.tray_transfer.episode_recording import TASK_TEXTS, EpisodeRecorder
from examples.euler.tray_transfer.lerobot_export import load_episode


class TrayExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "episode_0000"
        recorder = EpisodeRecorder(self.root, TASK_TEXTS[0], [f"arm_{i}" for i in range(7)],
                                   [f"grip_{i}" for i in range(8)])
        pixels = np.broadcast_to(np.arange(224, dtype=np.uint8)[None, :, None], (224, 224, 3)).copy()
        self.actions = []
        for i in range(3):
            action = np.array([i / 10] * 7 + [.28, 1.45])
            self.actions.append(action)
            recorder.add(np.arange(15) / 10 + i, action,
                         np.r_[action[:7], 0, 0, .28, 1.45, 0, 0, -.28, 1.45],
                         {"head": pixels, "right_wrist": pixels[:, ::-1]},
                         15 + i / 50, "approach", 100 + i,
                         {"head": 100 + i, "right_wrist": 100 + i},
                         requested_action15=np.zeros(15))
        recorder.finish(True, {"passed": True})

    def test_keeps_prestep_measured_state_and_executed_absolute_action(self):
        metadata, arrays = load_episode(self.root)
        self.assertEqual(metadata["task"], TASK_TEXTS[0])
        np.testing.assert_array_equal(arrays["action"], self.actions)
        np.testing.assert_array_equal(arrays["state"][2], np.arange(15) / 10 + 2)
        np.testing.assert_allclose(arrays["timestamp"], [0, .02, .04])

    def test_failed_recording_cannot_enter_training_export(self):
        path = self.root / "metadata.json"
        meta = json.loads(path.read_text())
        meta["success"] = False
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "successful"):
            load_episode(self.root)

    def test_camera_desynchronization_is_rejected(self):
        path = self.root / "episode.npz"
        with np.load(path) as source:
            arrays = dict(source)
        arrays["camera_indices"][1, 0] -= 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "camera"):
            load_episode(self.root)

    def test_missing_image_is_rejected_before_export(self):
        (self.root / "frames/right_wrist/000001.png").unlink()
        with self.assertRaisesRegex(ValueError, "frames"):
            load_episode(self.root)


if __name__ == "__main__":
    unittest.main()
