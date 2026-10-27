"""Intentional release is allowed; an earlier drop cannot be hidden by placement."""

import unittest
from copy import deepcopy

from examples.euler.tray_transfer.scene_transfer_result import TRANSFER_PHASES, evaluate_scene_transfer
from tests.test_blue_pick_result import stable_hold
from tests.test_blue_placement_result import stable_placement


def stable_transfer():
    grip = stable_hold()
    placement = stable_placement()[0]
    rows = [{**deepcopy(placement), **row, "tray_floor_normal_force_n": 0.,
             "gripper_target_normal_force_n": 20.} for row in grip]
    for phase in TRANSFER_PHASES[1:]:
        for _ in range(61 if phase == "settle_placed" else 3):
            row = {**deepcopy(rows[-1]), "phase": phase, "time_s": rows[-1]["time_s"] + .02}
            if phase in ("release", "retreat", "settle_placed"):
                row.update(contacts=[{"other_body": "tray", "normal_force_n": 8., "distance_m": -.0005}],
                           gripper_target_normal_force_n=0., tray_floor_normal_force_n=8.)
            rows.append(row)
    return rows


class SceneTransferResultTests(unittest.TestCase):
    def test_release_does_not_invalidate_earlier_grasp(self):
        rows = stable_transfer()
        original = deepcopy(rows)
        result = evaluate_scene_transfer(rows, [0., 0., 0.])
        self.assertTrue(result["passed"], result)
        self.assertEqual(rows, original)
        self.assertFalse(result["training_data"])

    def test_overlap_and_grasp_transport_drift_are_diagnostic(self):
        rows = stable_transfer()
        rows[25]["relative_position"][0] += .03
        rows[25]["contacts"][0]["distance_m"] = -.008
        next(row for row in rows if row["phase"] == "transport_over_tray")["relative_position"][0] += .04
        result = evaluate_scene_transfer(rows, [0., 0., 0.])
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["grasp_quality"]["diagnostic_checks"]["relative_translation_under_5mm"])
        self.assertFalse(result["transport"]["diagnostic_checks"]["relative_translation_under_5mm"])
        self.assertAlmostEqual(result["transport"]["metrics"]["max_target_penetration_m"], .008)
        self.assertFalse(evaluate_scene_transfer(rows, [0., 0., 0.], contact_diagnostics=False)["passed"])

    def test_good_placement_cannot_hide_drop_during_grasp_or_transport(self):
        for phase in ("lift_hold", "transport_raise", "transport_over_tray", "transport_lower"):
            with self.subTest(phase=phase):
                rows = stable_transfer()
                row = next(row for row in rows[1:] if row["phase"] == phase)
                row["contacts"] = []
                result = evaluate_scene_transfer(rows, [0., 0., 0.])
                self.assertFalse(result["passed"])
                self.assertTrue(result["placement"]["passed"])
                self.assertFalse(result["transport"]["checks"]["bilateral_contact_until_release"])

    def test_strict_mode_rejects_transport_overlap_but_retention_mode_records_it(self):
        rows = stable_transfer()
        next(r for r in rows if r["phase"] == "transport_over_tray")["contacts"][0]["distance_m"] = -.003
        self.assertTrue(evaluate_scene_transfer(rows, [0., 0., 0.])["passed"])
        strict = evaluate_scene_transfer(rows, [0., 0., 0.], contact_diagnostics=False)
        self.assertFalse(strict["passed"])
        self.assertFalse(strict["transport"]["checks"]["target_penetration_under_2mm"])

    def test_touchdown_is_allowed_only_during_lowering_while_still_grasped(self):
        for phase, expected in (("transport_lower", True), ("transport_over_tray", False)):
            rows = stable_transfer()
            row = next(row for row in rows if row["phase"] == phase)
            row.update(tray_floor_normal_force_n=8., lowest_collider_clearance_m=0.)
            row["contacts"].append({"other_body": "tray", "normal_force_n": 8., "distance_m": -.0005})
            result = evaluate_scene_transfer(rows, [0., 0., 0.])
            self.assertEqual(result["passed"], expected, result)

    def test_other_support_during_lowering_fails(self):
        rows = stable_transfer()
        row = next(row for row in rows if row["phase"] == "transport_lower")
        row["contacts"].append({"other_body": "tray_rail", "normal_force_n": 2., "distance_m": -.0005})
        row["external_support_force_n"] = 2.
        self.assertFalse(evaluate_scene_transfer(rows, [0., 0., 0.])["passed"])

    def test_invalid_sequence_missing_evidence_and_sampling_gaps_fail(self):
        for change in ("phase", "missing", "nan", "gap", "rotation"):
            with self.subTest(change=change):
                rows = stable_transfer()
                row = next(row for row in rows if row["phase"] == "transport_over_tray")
                if change == "phase":
                    row["phase"] = "transport_lower"
                elif change == "missing":
                    del row["external_support_force_n"]
                elif change == "nan":
                    row["contacts"][0]["normal_force_n"] = float("nan")
                elif change == "rotation":
                    row["relative_rotation"][0][0] = 2.
                else:
                    for later in rows[rows.index(row):]:
                        later["time_s"] += .1
                self.assertFalse(evaluate_scene_transfer(rows, [0., 0., 0.])["passed"])

    def test_object_not_fully_in_tray_fails_placement(self):
        rows = stable_transfer()
        rows[-1]["object_bounds_in_tray_max"][0] = .3
        result = evaluate_scene_transfer(rows, [0., 0., 0.])
        self.assertFalse(result["passed"])
        self.assertTrue(result["transport"]["passed"])


if __name__ == "__main__":
    unittest.main()
