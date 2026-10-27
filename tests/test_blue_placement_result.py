"""Placement requires a released, supported object throughout the final hold."""

import copy
import json
import unittest

import numpy as np

from examples.euler.tray_transfer.placement_result import evaluate_placement


def stable_placement():
    # The 123 mm tall part protrudes above the 98 mm tray wall, which is allowed.
    return [
        {
            "phase": "settle_placed", "time_s": float(t),
            "object_bounds_in_tray_min": [-.0235, -.067, .021],
            "object_bounds_in_tray_max": [.0235, .067, .144],
            "tray_inner_xy_min": [-.25, -.165], "tray_inner_xy_max": [.25, .165],
            "tray_floor_top_z": .021,
            "object_linear_speed_m_s": .001, "object_angular_speed_rad_s": .01,
            "gripper_target_normal_force_n": 0., "tray_floor_normal_force_n": 8.,
            "external_support_force_n": 0.,
        }
        for t in np.linspace(2., 3.2, 61)
    ]


class BluePlacementResultTests(unittest.TestCase):
    def test_released_inside_mode_accepts_motion_and_rail_contact_without_hold(self):
        rows = stable_placement()[-2:]
        for row in rows:
            row.update(object_linear_speed_m_s=.05, object_angular_speed_rad_s=.4,
                       tray_floor_normal_force_n=0., external_support_force_n=1.,
                       gripper={"hold_active": False,
                                "nominal_command": [0., 0., 0., -1., 0., 0., 0., -1.]})
            row["object_bounds_in_tray_max"][0] = .2505
            row["object_bounds_in_tray_min"][2] = .033
        self.assertFalse(evaluate_placement(rows)["passed"])
        result = evaluate_placement(rows, acceptance="released_inside_tray")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["success_criterion"], "released_inside_tray")
        for change in ("outside", "high", "gripped", "latched", "closed", "missing", "nan", "gap"):
            invalid = copy.deepcopy(rows)
            last = invalid[-1]
            if change == "outside":
                last["object_bounds_in_tray_max"][0] = .2511
            elif change == "high":
                last["object_bounds_in_tray_min"][2] = .0411
            elif change == "gripped":
                last["gripper_target_normal_force_n"] = .1
            elif change == "latched":
                last["gripper"]["hold_active"] = True
            elif change == "closed":
                last["gripper"]["nominal_command"][3] = 1.45
            elif change == "missing":
                del last["gripper"]
            elif change == "gap":
                last["time_s"] += .1
            else:
                last["object_linear_speed_m_s"] = float("nan")
            with self.subTest(change=change):
                self.assertFalse(evaluate_placement(invalid, acceptance="released_inside_tray")["passed"])

    def test_released_part_above_rim_passes_without_mutating_trace(self):
        rows = stable_placement()
        original = copy.deepcopy(rows)
        result = evaluate_placement(rows)
        self.assertTrue(result["passed"], result)
        self.assertTrue(json.loads(json.dumps(result, allow_nan=False))["passed"])
        self.assertEqual(rows, original)

    def test_one_sample_still_held_by_gripper_fails(self):
        rows = stable_placement()
        rows[30]["gripper_target_normal_force_n"] = .1
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_tipped_part_spanning_walls_fails(self):
        rows = stable_placement()
        for row in rows:
            row["object_bounds_in_tray_min"] = [-.26, -.02, .021]
            row["object_bounds_in_tray_max"] = [.26, .02, .068]
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_center_inside_does_not_excuse_collider_outside_clearance(self):
        rows = stable_placement()
        for row in rows:
            row["object_bounds_in_tray_min"][0] = .20
            row["object_bounds_in_tray_max"][0] = .24
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_table_support_cannot_replace_tray_floor_support(self):
        rows = stable_placement()
        for row in rows:
            row["tray_floor_normal_force_n"] = 0.
            row["external_support_force_n"] = 8.
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_shared_support_with_table_fails_even_with_floor_contact(self):
        rows = stable_placement()
        rows[30]["external_support_force_n"] = .1
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_flight_over_tray_without_floor_contact_fails(self):
        rows = stable_placement()
        rows[30]["tray_floor_normal_force_n"] = .5
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_floating_or_penetrating_bottom_fails(self):
        for bottom in (.027, .018):
            with self.subTest(bottom=bottom):
                rows = stable_placement()
                rows[30]["object_bounds_in_tray_min"][2] = bottom
                self.assertFalse(evaluate_placement(rows)["passed"])

    def test_motion_at_linear_or_angular_limit_fails(self):
        for field, value in (
            ("object_linear_speed_m_s", .01), ("object_angular_speed_rad_s", .1),
        ):
            with self.subTest(field=field):
                rows = stable_placement()
                rows[30][field] = value
                self.assertFalse(evaluate_placement(rows)["passed"])

    def test_empty_short_or_sparse_evidence_fails(self):
        rows = stable_placement()
        for trace in ([], rows[-20:], [rows[-1]], [rows[0], rows[-1]]):
            with self.subTest(samples=len(trace)):
                self.assertFalse(evaluate_placement(trace)["passed"])

    def test_gap_inside_hold_cannot_be_hidden_by_many_other_samples(self):
        rows = stable_placement()
        del rows[25:30]
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_separate_settle_segments_are_not_combined(self):
        rows = stable_placement()
        rows[35]["phase"] = "release"
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_old_hold_does_not_pass_when_final_phase_is_not_settled(self):
        rows = stable_placement()
        rows[-1]["phase"] = "retreat"
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_contact_before_final_window_is_not_a_placement_failure(self):
        rows = stable_placement()
        rows[0]["phase"] = "release"
        rows[0]["gripper_target_normal_force_n"] = 20.
        self.assertTrue(evaluate_placement(rows)["passed"])

    def test_hold_window_can_be_bracketed_between_sample_times(self):
        rows = stable_placement()
        self.assertTrue(evaluate_placement(rows, minimum_hold_s=1.01)["passed"])

    def test_repeated_or_backwards_time_fails(self):
        for time in (2.58, 2.57):
            with self.subTest(time=time):
                rows = stable_placement()
                rows[30]["time_s"] = time
                self.assertFalse(evaluate_placement(rows)["passed"])

    def test_missing_nonfinite_or_malformed_readings_fail_closed(self):
        for field, value in (
            ("time_s", float("nan")),
            ("object_bounds_in_tray_min", [-.02, float("inf"), .021]),
            ("object_bounds_in_tray_max", [0., 0.]),
            ("tray_inner_xy_min", [-.25, float("nan")]),
            ("tray_floor_top_z", None),
            ("object_linear_speed_m_s", -.1),
            ("external_support_force_n", -.1),
            ("tray_floor_normal_force_n", float("inf")),
            ("gripper_target_normal_force_n", False),
        ):
            with self.subTest(field=field, value=value):
                rows = stable_placement()
                rows[30][field] = value
                result = evaluate_placement(rows)
                self.assertFalse(result["passed"])
                json.dumps(result, allow_nan=False)
        rows = stable_placement()
        del rows[30]["object_angular_speed_rad_s"]
        self.assertFalse(evaluate_placement(rows)["passed"])

    def test_inverted_bounds_or_collapsed_interior_fail(self):
        for field, value in (
            ("object_bounds_in_tray_min", [.03, -.067, .021]),
            ("tray_inner_xy_max", [-.25, .165]),
        ):
            rows = stable_placement()
            rows[30][field] = value
            self.assertFalse(evaluate_placement(rows)["passed"])
        self.assertFalse(evaluate_placement(stable_placement(), edge_clearance_m=.2)["passed"])

    def test_invalid_acceptance_parameters_fail_closed(self):
        for kwargs in (
            {"minimum_hold_s": 0.}, {"minimum_hold_s": float("nan")},
            {"edge_clearance_m": -.01}, {"edge_clearance_m": float("inf")},
            {"max_sample_gap_s": 0.},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(evaluate_placement(stable_placement(), **kwargs)["passed"])


if __name__ == "__main__":
    unittest.main()
