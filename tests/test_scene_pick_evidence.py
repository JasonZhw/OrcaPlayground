"""Offline evidence must authorize the exact diagnostic used for an online run."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.euler.tray_transfer import scene_pick
from tests.test_blue_pick_result import stable_hold


class ScenePickEvidenceTests(unittest.TestCase):
    def test_release_inside_mode_only_allows_contact_tolerance(self):
        sample = {
            "object_bounds_in_tray_min": [.20, -.04, .032],
            "object_bounds_in_tray_max": [.2505, .04, .125],
            "tray_inner_xy_min": [-.25, -.16], "tray_inner_xy_max": [.25, .16],
            "tray_floor_top_z": .02,
        }
        scene_pick.ensure_release_over_tray(sample, .02, acceptance="released_inside_tray")
        sample["object_bounds_in_tray_max"][0] = .2511
        with self.assertRaises(RuntimeError):
            scene_pick.ensure_release_over_tray(sample, .02, acceptance="released_inside_tray")

    def test_task_completion_keeps_strict_retention_failures_as_diagnostics(self):
        from tests.test_scene_transfer_result import stable_transfer
        from tests.test_tray_return_home import home_trace
        rows = stable_transfer()
        for row in rows:
            row["gripper"] = {"hold_active": False,
                              "nominal_command": [0., 0., 0., -1., 0., 0., 0., -1.]}
        next(row for row in rows if row["phase"] == "transport_lower")["contacts"] = []
        returning = home_trace()
        offset = rows[-1]["time_s"] - returning[0]["time_s"] + .02
        for row in returning:
            row["time_s"] += offset
        trace = rows + returning
        result = scene_pick.evaluate_transfer_with_home(
            trace, [0., 0., 0.], len(rows), [0.] * 7,
            {"edge_clearance_m": 0., "acceptance": "released_inside_tray"}, contact_diagnostics=True,
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["success_criterion"], "released_inside_tray_and_return_home")
        self.assertFalse(result["strict_quality_evaluation"]["passed"])
        self.assertTrue(result["return_home"]["passed"])
        trace[-1]["right_arm_qpos"][0] = .1
        self.assertFalse(scene_pick.evaluate_transfer_with_home(
            trace, [0., 0., 0.], len(rows), [0.] * 7,
            {"edge_clearance_m": 0., "acceptance": "released_inside_tray"}, contact_diagnostics=True,
        )["passed"])

    def test_release_only_opens_inside_tray_near_floor(self):
        sample = {
            "object_bounds_in_tray_min": [-.05, -.04, .025],
            "object_bounds_in_tray_max": [.05, .04, .125],
            "tray_inner_xy_min": [-.25, -.16], "tray_inner_xy_max": [.25, .16],
            "tray_floor_top_z": .02,
        }
        scene_pick.ensure_release_over_tray(sample, .02)
        for lower, upper in [([-.3, -.04, .025], [.05, .04, .125]),
                             ([-.05, -.04, .07], [.05, .04, .17]),
                             ([-.05, -.04, .01], [.05, .04, .11])]:
            with self.subTest(lower=lower), self.assertRaisesRegex(RuntimeError, "keeping the gripper closed"):
                scene_pick.ensure_release_over_tray(
                    {**sample, "object_bounds_in_tray_min": lower, "object_bounds_in_tray_max": upper}, .02
                )

    def test_retention_mode_reports_penetration_without_failing_a_stable_grasp(self):
        rows = stable_hold()
        rows[10]["contacts"][0]["distance_m"] = -.006
        result = scene_pick.evaluate_scene_pick(rows, [0., 0., 0.], contact_diagnostics=True)
        self.assertTrue(result["passed"])
        self.assertFalse(result["diagnostic_checks"]["trajectory_target_penetration_under_2mm"])
        self.assertAlmostEqual(result["trajectory_metrics"]["max_target_penetration_m"], .006)
        self.assertFalse(scene_pick.evaluate_scene_pick(rows, [0., 0., 0.])["passed"])

    def test_good_final_hold_cannot_hide_an_earlier_drop(self):
        rows = stable_hold()
        for row in rows:
            row["time_s"] += 10
        lifted = {**deepcopy(rows[0]), "phase": "lift", "time_s": 0.}
        dropped = {**deepcopy(lifted), "time_s": 1., "lowest_collider_clearance_m": 0., "contacts": []}
        result = scene_pick.evaluate_scene_pick([lifted, dropped, *rows], [0., 0., 0.], contact_diagnostics=True)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["retained_since_liftoff"])

    def test_retention_covers_early_liftoff_and_every_subsequent_phase(self):
        rows = stable_hold()
        for row in rows:
            row["time_s"] += 10
        lifted = {**deepcopy(rows[0]), "phase": "lift", "time_s": 0., "lowest_collider_clearance_m": .003}
        dropped = {**deepcopy(lifted), "phase": "grasp_hold", "time_s": 1.,
                   "lowest_collider_clearance_m": 0., "contacts": []}
        result = scene_pick.evaluate_scene_pick([lifted, dropped, *rows], [0., 0., 0.], contact_diagnostics=True)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["retained_since_liftoff"])
        self.assertEqual(result["retention_metrics"]["first_liftoff_time_s"], 0.)

    def test_contact_diagnostics_observes_deep_tip_contact_but_preserves_other_contact_stop(self):
        tip = {"other_body": scene_pick.TIPS[0], "distance_m": -.006}
        self.assertTrue(scene_pick.penetration_requires_stop([tip], contact_diagnostics=False))
        self.assertFalse(scene_pick.penetration_requires_stop([tip], contact_diagnostics=True))
        for other in ["table", "unitree_g1_1_gripper_r_inner_link2"]:
            with self.subTest(other=other):
                self.assertTrue(scene_pick.penetration_requires_stop(
                    [{**tip, "other_body": other}], contact_diagnostics=True
                ))

    def test_calibrated_targets_and_model_limits(self):
        targets = [[0, 0, 0, .97, 0, 0, 0, .97], [0, 0, .28, .9, 0, 0, -.28, .9]]
        limits = np.tile([-1., 2.], (8, 1))
        np.testing.assert_array_equal(
            scene_pick.gripper_phase_points({"servo_targets": targets}, .97, limits), targets
        )
        for invalid in [
            {"drive": [-1, "close"], "servo_targets": targets},
            {"servo_targets": targets[:1]},
            {"servo_targets": [[0] * 7] * 2},
            {"servo_targets": [[float("nan")] * 8] * 2},
            {"servo_targets": [[3.] * 8] * 2},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scene_pick.gripper_phase_points(invalid, .97, limits)

    def test_alignment_must_be_latched_before_lifting(self):
        phases = [
            {"name": "close", "joints": "gripper"},
            {"name": "align_fingers", "joints": "gripper"},
            {"name": "lift", "joints": "arm"},
        ]
        plan = {"phases": phases, "gripper_latch_after_phase": "align_fingers"}
        self.assertEqual(scene_pick.gripper_latch_phase(plan), "align_fingers")
        for name in ["close", "lift", "missing"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                scene_pick.gripper_latch_phase({**plan, "gripper_latch_after_phase": name})
        with self.assertRaisesRegex(ValueError, "unique"):
            scene_pick.gripper_latch_phase({**plan, "phases": [phases[0]] + phases})

    def matching_evidence(self):
        manifest = {
            "profile": "original", "changes": [], "online": False,
            "base_fixed": True, "waist_fixed": False, "gripper_hold_latched": True,
            "plan_sha256": "plan", "source_sha256": "scene", "asset_sha256": {"mesh": "asset"},
            "close_drive": 1.25, "target_body": "package",
            "contact_diagnostics": False,
            "control_config": dict(scene_pick.CONTROL_CONFIG),
            "implementation_sha256": {name: "sha256" for name in scene_pick.IMPLEMENTATION_FILES},
        }
        result = {"passed": True, "online": False, "target_body": "package",
                  "contact_diagnostics": False, "success_criterion": "strict_contact_diagnostic"}
        return result, manifest

    def test_result_and_manifest_must_use_the_same_success_criterion(self):
        result, manifest = self.matching_evidence()
        for key, value in [("contact_diagnostics", True), ("contact_diagnostics", 0),
                           ("contact_diagnostics", None), ("success_criterion", "retention"),
                           ("success_criterion", None)]:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "Offline evidence"):
                scene_pick.validate_offline_evidence({**result, key: value}, manifest, manifest)
        for key in ("contact_diagnostics", "success_criterion"):
            changed = dict(result)
            del changed[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "Offline evidence"):
                scene_pick.validate_offline_evidence(changed, manifest, manifest)

    def test_exact_matching_original_evidence_is_accepted(self):
        result, manifest = self.matching_evidence()
        scene_pick.validate_offline_evidence(result, manifest, deepcopy(manifest))

    def test_pick_evidence_cannot_stand_in_for_transfer_evidence(self):
        result, manifest = self.matching_evidence()
        transfer_manifest = {**manifest, "task_mode": "tray_transfer"}
        with self.assertRaisesRegex(ValueError, "task mode"):
            scene_pick.validate_offline_evidence(result, manifest, transfer_manifest)
        transfer_result = {**result, "task_mode": "tray_transfer",
                           "success_criterion": "transfer_strict_contact_diagnostic"}
        scene_pick.validate_offline_evidence(transfer_result, transfer_manifest, transfer_manifest)

    def test_explicit_preview_accepts_completed_quality_failure_without_marking_it_passed(self):
        result, manifest = self.matching_evidence()
        result.update(passed=False, checks={"penetration_under_2mm": False}, training_data=False)
        original = deepcopy(result)
        scene_pick.validate_offline_evidence(result, manifest, deepcopy(manifest), preview=True)
        self.assertEqual(result, original)
        with self.assertRaises(ValueError):
            scene_pick.validate_offline_evidence(result, manifest, deepcopy(manifest))

    def test_preview_still_rejects_aborted_and_mismatched_evidence(self):
        result, manifest = self.matching_evidence()
        result.update(passed=False, checks={"penetration_under_2mm": False})
        with self.assertRaisesRegex(ValueError, "completed"):
            scene_pick.validate_offline_evidence(
                {**result, "error": "Arm collision"}, manifest, manifest, preview=True
            )
        with self.assertRaisesRegex(ValueError, "completed"):
            scene_pick.validate_offline_evidence({"passed": False, "online": False}, manifest, manifest, preview=True)
        for key in ("plan_sha256", "source_sha256", "implementation_sha256", "control_config"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not match"):
                scene_pick.validate_offline_evidence(result, {**manifest, key: "changed"}, manifest, preview=True)

    def test_changed_or_missing_physics_control_or_code_is_rejected(self):
        result, expected = self.matching_evidence()
        changes = {
            "profile": "experimental-contact", "changes": [{"exclude": "finger"}], "online": True,
            "base_fixed": False, "waist_fixed": True, "gripper_hold_latched": False,
            "plan_sha256": "other-plan", "source_sha256": "other-scene", "asset_sha256": {},
            "close_drive": 2.0, "target_body": "other-package",
            "contact_diagnostics": True,
            "control_config": {**expected["control_config"], "position_feedback_gain": 1.0},
            "implementation_sha256": {**expected["implementation_sha256"], "blue_env.py": "changed"},
        }
        for key, changed in changes.items():
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    prior = deepcopy(expected)
                    if missing:
                        del prior[key]
                    else:
                        prior[key] = changed
                    with self.assertRaisesRegex(ValueError, "Offline evidence"):
                        scene_pick.validate_offline_evidence(result, prior, expected)
        for key, value in [("passed", False), ("passed", 1), ("online", True), ("online", None),
                           ("target_body", "other-package")]:
            with self.subTest(result_key=key, value=value), self.assertRaisesRegex(ValueError, "Offline evidence"):
                scene_pick.validate_offline_evidence({**result, key: value}, expected, expected)

    def test_archive_preserves_all_sources_and_rejects_missing_or_changed_code(self):
        sources = {name: f"# snapshot: {name}\n".encode() for name in scene_pick.IMPLEMENTATION_FILES}
        hashes = scene_pick.implementation_hashes(sources)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_pick.archive_implementation(root, sources)
            scene_pick.validate_code_archive(root, hashes)
            self.assertEqual((root / "run.py").read_bytes(), sources["scene_pick.py"])
            for name, content in sources.items():
                self.assertEqual((root / "code" / name).read_bytes(), content)
            altered = root / "code" / "gripper_control.py"
            altered.write_bytes(b"# different controller\n")
            with self.assertRaisesRegex(ValueError, "archive changed"):
                scene_pick.validate_code_archive(root, hashes)
            altered.unlink()
            with self.assertRaisesRegex(ValueError, "archive is incomplete"):
                scene_pick.validate_code_archive(root, hashes)

    def test_legacy_or_modified_physics_evidence_is_rejected_before_live_access(self):
        for profile, changes in [("original", []), ("experimental-contact", [{"friction": 2}])]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source.xml"
                source.write_text("<mujoco/>")
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                plan = root / "plan.json"
                plan.write_text(json.dumps({
                    "source_xml": str(source), "source_sha256": source_hash, "target_body": "package",
                    "phases": [{"name": "close", "joints": "gripper"}, {"name": "lift", "joints": "arm"}],
                }))
                prior = root / "prior"
                prior.mkdir()
                (prior / "result.json").write_text(json.dumps({"passed": True, "online": False}))
                (prior / "manifest.json").write_text(json.dumps({
                    "online": False, "profile": profile, "changes": changes,
                    "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                    "source_sha256": source_hash, "asset_sha256": {}, "close_drive": 1.25,
                }))
                argv = ["scene_pick", "--online", "--plan", str(plan), "--output", str(root / "run"),
                        "--offline-evidence", str(prior), "--close-drive", "1.25"]
                with patch("sys.argv", argv), patch(
                    "grpc.insecure_channel", side_effect=AssertionError("Rejected evidence reached live access"),
                ), self.assertRaisesRegex(ValueError, "Offline evidence"):
                    scene_pick.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
