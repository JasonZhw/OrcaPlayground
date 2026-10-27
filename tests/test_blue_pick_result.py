"""A tilted or supported object must never be labeled a successful lift."""

import copy
import json
import unittest

import numpy as np

from examples.euler.tray_transfer.blue_pick import TIPS, evaluate_hold


def stable_hold():
    return [{"phase": "lift_hold", "time_s": float(t), "blue_position": [0., 0., .1],
             "relative_position": [0., 0., .15], "relative_rotation": np.eye(3).tolist(),
             "lowest_collider_clearance_m": .08,
             "contacts": [{"other_body": tip, "normal_force_n": 10., "distance_m": -.0005}
                          for tip in TIPS]}
            for t in np.linspace(0., 1., 51)]


class BluePickResultTests(unittest.TestCase):
    def test_stable_unsupported_bilateral_lift_passes(self):
        result = evaluate_hold(stable_hold(), [0., 0., 0.])
        self.assertTrue(result["passed"])
        self.assertTrue(json.loads(json.dumps(result))["passed"])

    def test_origin_rise_with_collider_on_table_fails(self):
        rows = stable_hold()
        for row in rows:
            row["lowest_collider_clearance_m"] = 0.
        self.assertFalse(evaluate_hold(rows, [0., 0., 0.])["passed"])

    def test_single_supported_or_lost_contact_sample_fails(self):
        for mode in ("support", "lost_contact", "deep_penetration"):
            with self.subTest(mode=mode):
                rows = stable_hold()
                if mode == "support":
                    rows[25]["contacts"].append({"other_body": "table", "normal_force_n": 2., "distance_m": 0.})
                elif mode == "lost_contact":
                    rows[25]["contacts"].pop()
                else:
                    rows[25]["contacts"][0]["distance_m"] = -.01
                self.assertFalse(evaluate_hold(rows, [0., 0., 0.])["passed"])

    def test_relative_slip_or_rotation_fails(self):
        baseline = stable_hold()
        for key, value in (
            ("relative_position", [.02, 0., .15]),
            ("relative_rotation", [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
        ):
            rows = copy.deepcopy(baseline)
            rows[-1][key] = value
            self.assertFalse(evaluate_hold(rows, [0., 0., 0.])["passed"])

    def test_many_small_external_supports_are_summed(self):
        rows = stable_hold()
        for row in rows:
            row["contacts"].extend(
                {"other_body": "table", "normal_force_n": .09, "distance_m": 0.} for _ in range(80)
            )
        result = evaluate_hold(rows, [0., 0., 0.])
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["metrics"]["max_total_external_support_n"], 7.2)

    def test_short_or_missing_hold_fails(self):
        for rows in ([], stable_hold()[-20:]):
            self.assertFalse(evaluate_hold(rows, [0., 0., 0.])["passed"])

    def test_earlier_deep_contact_cannot_be_hidden_by_good_final_hold(self):
        rows = stable_hold()
        closing = copy.deepcopy(rows[0])
        closing.update(phase="close", time_s=-1.)
        closing["contacts"][0]["distance_m"] = -.01
        result = evaluate_hold([closing, *rows], [0., 0., 0.])
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["trajectory_metrics"]["max_target_penetration_m"], .01)
        self.assertEqual(result["trajectory_metrics"]["peak_penetration_phase"], "close")


if __name__ == "__main__":
    unittest.main()
