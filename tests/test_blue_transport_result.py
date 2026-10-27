"""Premature loss of grasp cannot be repaired by a later lucky tray landing."""

import copy
import json
import unittest

import numpy as np

from examples.euler.tray_transfer.transport_result import evaluate_transport

TIPS = ("right_inner_tip", "right_outer_tip")
PHASES = ("park_left", "clear_gray_before_transfer", "carry_and_lower", "place_hold")


def held_transfer():
    phases = ["lift_hold", "lift_hold", *[phase for phase in PHASES for _ in range(4)],
              "release", "settle_placed"]
    rows = [{
        "phase": phase, "time_s": i * .02,
        "relative_position": [0., 0., .15], "relative_rotation": np.eye(3).tolist(),
        "contacts": [{"other_body": tip, "normal_force_n": 2.} for tip in TIPS],
    } for i, phase in enumerate(phases)]
    for row in rows[-2:]:
        row["contacts"] = [{"other_body": "tray", "normal_force_n": 8.}]
        row["relative_position"] = [.4, 0., -.2]
    return rows


class BlueTransportResultTests(unittest.TestCase):
    def test_held_transfer_passes_and_release_support_is_outside_scope(self):
        rows = held_transfer()
        # The last lift_hold sample, not an earlier settling sample, is the baseline.
        rows[0]["relative_position"][0] = -.02
        original = copy.deepcopy(rows)
        result = evaluate_transport(rows, TIPS, PHASES)
        self.assertTrue(result["passed"], result)
        json.dumps(result, allow_nan=False)
        self.assertEqual(rows, original)

    def test_drop_then_lucky_tray_landing_cannot_pass(self):
        rows = held_transfer()
        rows[10]["contacts"] = []
        rows[10]["relative_position"] = [.5, 0., -.3]
        self.assertFalse(evaluate_transport(rows, TIPS, PHASES)["passed"])

    def test_losing_only_one_tip_even_once_fails(self):
        rows = held_transfer()
        rows[8]["contacts"].pop()
        self.assertFalse(evaluate_transport(rows, TIPS, PHASES)["passed"])

    def test_sliding_rotating_or_early_support_fails(self):
        angle = np.deg2rad(6.)
        for field, value in (
            ("relative_position", [.0051, 0., .15]),
            ("relative_rotation", [[np.cos(angle), -np.sin(angle), 0.],
                                   [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]),
            ("contacts", [{"other_body": tip, "normal_force_n": 2.} for tip in TIPS]
             + [{"other_body": "tray", "normal_force_n": .06},
                {"other_body": "table", "normal_force_n": .06}]),
        ):
            with self.subTest(field=field):
                rows = held_transfer()
                rows[12][field] = value
                self.assertFalse(evaluate_transport(rows, TIPS, PHASES)["passed"])

    def test_missing_reordered_or_unexpected_phase_fails(self):
        rows = held_transfer()
        cases = [
            [row for row in rows if row["phase"] != "release"],
            [row for row in rows if row["phase"] != "place_hold"],
            [row for row in rows if row["phase"] != "lift_hold"],
        ]
        reversed_phases = copy.deepcopy(rows)
        for row in reversed_phases:
            if row["phase"] == "park_left":
                row["phase"] = "clear_gray_before_transfer"
            elif row["phase"] == "clear_gray_before_transfer":
                row["phase"] = "park_left"
        cases.append(reversed_phases)
        unexpected = copy.deepcopy(rows)
        unexpected[10]["phase"] = "recover"
        cases.append(unexpected)
        for case in cases:
            with self.subTest(phases=[row["phase"] for row in case]):
                self.assertFalse(evaluate_transport(case, TIPS, PHASES)["passed"])

    def test_time_reversal_or_gap_including_release_boundary_fails(self):
        cases = []
        repeated = held_transfer()
        repeated[8]["time_s"] = repeated[7]["time_s"]
        cases.append(repeated)
        sparse = held_transfer()
        del sparse[7:10]
        cases.append(sparse)
        delayed_release = held_transfer()
        for row in delayed_release[-2:]:
            row["time_s"] += .1
        cases.append(delayed_release)
        for case in cases:
            with self.subTest(times=[row["time_s"] for row in case]):
                self.assertFalse(evaluate_transport(case, TIPS, PHASES)["passed"])

    def test_nonfinite_missing_or_invalid_readings_fail_closed(self):
        for field, value in (
            ("time_s", float("nan")),
            ("relative_position", [float("inf"), 0., .15]),
            ("relative_rotation", np.zeros((3, 3)).tolist()),
            ("contacts", [{"other_body": TIPS[0], "normal_force_n": float("nan")}]),
            ("contacts", [{"other_body": TIPS[0], "normal_force_n": -1.}]),
            ("contacts", [{"normal_force_n": 2.}]),
        ):
            with self.subTest(field=field, value=value):
                rows = held_transfer()
                rows[8][field] = value
                result = evaluate_transport(rows, TIPS, PHASES)
                self.assertFalse(result["passed"])
                json.dumps(result, allow_nan=False)
        rows = held_transfer()
        del rows[8]["relative_position"]
        self.assertFalse(evaluate_transport(rows, TIPS, PHASES)["passed"])

    def test_empty_trace_or_invalid_plan_contract_fails(self):
        for rows, tips, phases in (
            ([], TIPS, PHASES), (held_transfer(), (TIPS[0], TIPS[0]), PHASES),
            (held_transfer(), TIPS, ()), (held_transfer(), TIPS, (*PHASES, "release")),
        ):
            with self.subTest(tips=tips, phases=phases):
                self.assertFalse(evaluate_transport(rows, tips, phases)["passed"])


if __name__ == "__main__":
    unittest.main()
