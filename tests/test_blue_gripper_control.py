"""Regression checks for the official joint2 actuator command convention."""

import json
import unittest
from pathlib import Path

import numpy as np

from examples.euler.tray_transfer import gripper_control
from examples.euler.tray_transfer.gripper_control import (
    OPEN,
    REFERENCE_CLOSE,
    right_gripper_command,
)


class RightGripperHoldTests(unittest.TestCase):
    def make_hold(self):
        self.assertTrue(hasattr(gripper_control, "RightGripperHold"), "Missing persistent hold controller")
        self.names = ("arm_a",) + tuple(reversed(gripper_control.RIGHT_GRIPPER_JOINTS)) + ("arm_b",)
        return gripper_control.RightGripperHold(self.names)

    def test_arm_updates_cannot_reopen_held_gripper(self):
        hold = self.make_hold()
        hold.latch(1.25)
        for opening in [-1., 0., 2.]:
            incoming = np.full(10, opening)
            incoming[[0, 9]] = [.3, -.2]
            original = incoming.copy()
            command = hold.apply(incoming)
            np.testing.assert_array_equal(command[[0, 9]], [.3, -.2])
            np.testing.assert_array_equal(command[1:9], right_gripper_command(self.names[1:9], 1.25))
            np.testing.assert_array_equal(incoming, original)

    def test_only_explicit_release_allows_new_gripper_command(self):
        hold = self.make_hold()
        incoming = np.arange(10, dtype=float) / 10
        np.testing.assert_array_equal(hold.apply(incoming), incoming)
        hold.latch(1.25)
        with self.assertRaises(ValueError):
            hold.latch(-1.)
        self.assertTrue(hold.active)
        hold.release()
        self.assertFalse(hold.active)
        np.testing.assert_array_equal(hold.apply(incoming), incoming)

    def test_calibrated_finger_angles_remain_held_in_actuator_name_order(self):
        hold = self.make_hold()
        targets = np.array([0., 0., .28, .90, 0., 0., -.28, .90])
        hold.latch_command(targets)
        targets[:] = 0
        requested = np.zeros(10)
        requested[[0, 9]] = [.4, -.3]
        actual = hold.apply(requested)
        np.testing.assert_array_equal(actual[1:9], [.90, -.28, 0, 0, .90, .28, 0, 0])
        np.testing.assert_array_equal(actual[[0, 9]], [.4, -.3])
        with self.assertRaises(ValueError):
            hold.latch(.90)

    def test_invalid_calibrated_command_does_not_replace_active_hold(self):
        hold = self.make_hold()
        original = np.array([0., 0., .28, .90, 0., 0., -.28, .90])
        hold.latch_command(original)
        for bad in [np.zeros(8), np.ones(7), [0, 0, np.nan, .9, 0, 0, 0, .9]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                hold.latch_command(bad)
            np.testing.assert_array_equal(hold.held_command, original)


class RightGripperCommandTests(unittest.TestCase):
    def setUp(self):
        self.names = [
            "unitree_g1_1_idx71_gripper_r_inner_joint1",
            "unitree_g1_1_idx72_gripper_r_inner_joint3",
            "unitree_g1_1_idx73_gripper_r_inner_joint4",
            "unitree_g1_1_idx79_gripper_r_inner_joint2",
            "unitree_g1_1_idx81_gripper_r_outer_joint1",
            "unitree_g1_1_idx82_gripper_r_outer_joint3",
            "unitree_g1_1_idx83_gripper_r_outer_joint4",
            "unitree_g1_1_idx89_gripper_r_outer_joint2",
        ]

    def test_reference_open_and_close_drive_joint2_not_joint1(self):
        self.assertEqual(OPEN, -1.0)
        self.assertEqual(REFERENCE_CLOSE, 2.0)
        np.testing.assert_array_equal(
            right_gripper_command(self.names, OPEN), [0, 0, 0, -1, 0, 0, 0, -1],
        )
        closed = right_gripper_command(self.names, REFERENCE_CLOSE)
        np.testing.assert_array_equal(closed, [0, 0, 0, 2, 0, 0, 0, 2])
        self.assertEqual(np.count_nonzero(closed), 2)

    def test_joint_names_determine_output_order(self):
        shuffled = [self.names[i] for i in [7, 4, 2, 0, 3, 5, 1, 6]]
        np.testing.assert_array_equal(
            right_gripper_command(shuffled, .5), [.5, 0, 0, 0, .5, 0, 0, 0],
        )
        self.assertEqual(shuffled[0], self.names[7])

    def test_unknown_missing_duplicate_and_foreign_names_are_rejected(self):
        invalid = [
            self.names[:-1],
            self.names + [self.names[0]],
            self.names[:-1] + [self.names[0]],
            self.names[:-1] + ["unknown"],
            [name.replace("unitree_g1_1_", "unitree_g1_2_") for name in self.names],
            self.names[:-1] + [None],
        ]
        for names in invalid:
            with self.subTest(names=names), self.assertRaises(ValueError):
                right_gripper_command(names, OPEN)

    def test_nonfinite_out_of_range_and_nonscalar_drive_are_rejected(self):
        for drive in [np.nan, np.inf, -np.inf, -1.00001, 2.00001, True, "1", None, [1]]:
            with self.subTest(drive=drive), self.assertRaises(ValueError):
                right_gripper_command(self.names, drive)

    def test_zero_drive_is_a_valid_actuator_input(self):
        np.testing.assert_array_equal(right_gripper_command(self.names, 0), np.zeros(8))

    def test_shipped_plan_does_not_close_by_zeroing_all_eight_targets(self):
        path = Path(__file__).resolve().parents[1] / "examples/euler/tray_transfer/blue_pick_plan.json"
        plan = json.loads(path.read_text())
        gripper = {p["name"]: p for p in plan["phases"] if p["joints"] == "gripper"}
        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(gripper["open"]["drive"], [0, OPEN])
        self.assertEqual(gripper["close"]["drive"], [OPEN, REFERENCE_CLOSE])
        for phase in gripper.values():
            self.assertNotIn("q", phase)
            for drive in phase["drive"]:
                right_gripper_command(plan["gripper_joint_names"], drive)


if __name__ == "__main__":
    unittest.main(verbosity=2)
