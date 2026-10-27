"""Recording must pair pre-step observations with the command actually executed."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.euler.tray_transfer.episode_recording import TASK_TEXTS, EpisodeRecorder
from examples.euler.tray_transfer.gripper_control import RIGHT_GRIPPER_JOINTS, RightGripperHold
from examples.euler.tray_transfer.native_capture import record_control_step


class CaptureStepTest(unittest.TestCase):
    def test_records_before_motion_and_uses_latched_command(self):
        names = [f"arm{i}" for i in range(7)] + list(RIGHT_GRIPPER_JOINTS)
        hold = RightGripperHold(names)
        hold.latch_command([0, 0, .28, 1.35, 0, 0, -.28, 1.35])
        env = SimpleNamespace(
            data=SimpleNamespace(qpos=np.arange(15, dtype=float), time=1.0),
            actuator_qpos_indices=np.arange(15), right_gripper_hold=hold,
        )
        def step(control):
            env.last_command = hold.apply(control)
            env.data.qpos += 10
            env.data.time += .02
        env.step = step
        pixels = np.zeros((224, 224, 3), dtype=np.uint8)
        pixels[30:60] = 255
        cameras = SimpleNamespace(capture=lambda index: (
            {key: pixels for key in ("head", "right_wrist")},
            {key: {"simulate_index": index} for key in ("head", "right_wrist")},
        ))
        requested = np.zeros(15)
        requested[[10, 14]] = -1
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(Path(tmp) / "episode", TASK_TEXTS[0], names[:7], names[7:])
            record_control_step(env, requested, list(range(7)), list(range(7, 15)), recorder, cameras, "carry")
            recorder.finish(False, {})
            with np.load(recorder.directory / "episode.npz") as data:
                np.testing.assert_array_equal(data["state"][0], np.arange(15))
                np.testing.assert_allclose(data["action"][0, 7:], [.28, 1.35])
                np.testing.assert_array_equal(data["requested_action"][0], requested)
                self.assertEqual(data["sim_time"][0], 1.0)
            self.assertEqual(env.data.time, 1.02)


if __name__ == "__main__":
    unittest.main()
