"""Returning the arm must preserve the placed object and finish at home."""

import unittest
from copy import deepcopy

from examples.euler.tray_transfer.return_home import evaluate_home_return
from tests.test_blue_placement_result import stable_placement


def home_trace():
    rows = stable_placement()
    for i, row in enumerate(rows):
        row.update(phase="return_home" if i < 3 else "home_hold",
                   right_arm_qpos=[0.] * 7, right_arm_qvel=[0.] * 7,
                   forbidden_contacts=[], gripper={"hold_active": False,
                       "nominal_command": [0., 0., 0., -1., 0., 0., 0., -1.]})
    return rows


class ReturnHomeTests(unittest.TestCase):
    def test_inside_mode_returns_home_despite_rail_contact_and_object_motion(self):
        rows = home_trace()
        for row in rows:
            row.update(object_linear_speed_m_s=.03, object_angular_speed_rad_s=.2,
                       tray_floor_normal_force_n=0., external_support_force_n=1.)
            row["object_bounds_in_tray_max"][0] = .2505
        self.assertFalse(evaluate_home_return(rows, [0.] * 7)["passed"])
        result = evaluate_home_return(rows, [0.] * 7, acceptance="released_inside_tray")
        self.assertTrue(result["passed"], result)
        for change in ("outside", "not_home", "moving_arm"):
            invalid = deepcopy(rows)
            if change == "outside":
                invalid[-1]["object_bounds_in_tray_max"][0] = .252
            elif change == "not_home":
                invalid[-1]["right_arm_qpos"][0] = .1
            else:
                invalid[-1]["right_arm_qvel"][0] = .1
            with self.subTest(change=change):
                self.assertFalse(evaluate_home_return(
                    invalid, [0.] * 7, acceptance="released_inside_tray",
                )["passed"])

    def test_returned_open_arm_and_stable_object_pass(self):
        rows = home_trace()
        before = deepcopy(rows)
        result = evaluate_home_return(rows, [0.] * 7)
        self.assertTrue(result["passed"], result)
        self.assertEqual(rows, before)

    def test_arm_not_home_or_still_moving_cannot_pass(self):
        for field, value in (("right_arm_qpos", .1), ("right_arm_qvel", .1)):
            rows = home_trace()
            rows[-1][field][0] = value
            self.assertFalse(evaluate_home_return(rows, [0.] * 7)["passed"])

    def test_home_uses_the_same_configured_tray_margin(self):
        rows = home_trace()
        for row in rows:
            row["object_bounds_in_tray_min"][0] = .20
            row["object_bounds_in_tray_max"][0] = .24
        self.assertFalse(evaluate_home_return(rows, [0.] * 7)["passed"])
        self.assertTrue(evaluate_home_return(rows, [0.] * 7, edge_clearance_m=0.)["passed"])
        rows[-1]["object_bounds_in_tray_max"][0] = .251
        self.assertFalse(evaluate_home_return(rows, [0.] * 7, edge_clearance_m=0.)["passed"])

    def test_return_cannot_hide_disturbed_object_or_contact(self):
        for change in ("box", "collision", "grip", "close_command"):
            rows = home_trace()
            if change == "box":
                rows[-1]["object_bounds_in_tray_max"][0] = .4
            elif change == "collision":
                rows[0]["forbidden_contacts"] = [{"bodies": ["arm", "tray"]}]
            elif change == "grip":
                rows[0]["gripper"]["hold_active"] = True
            else:
                rows[0]["gripper"]["nominal_command"][3] = 1.35
            self.assertFalse(evaluate_home_return(rows, [0.] * 7)["passed"])

    def test_missing_hold_or_nonfinite_state_is_not_success(self):
        rows = home_trace()
        self.assertFalse(evaluate_home_return(rows[:3], [0.] * 7)["passed"])
        rows[-1]["right_arm_qpos"][0] = float("nan")
        self.assertFalse(evaluate_home_return(rows, [0.] * 7)["passed"])


if __name__ == "__main__":
    unittest.main()
