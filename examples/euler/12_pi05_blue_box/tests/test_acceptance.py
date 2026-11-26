"""Reject incomplete evaluations, altered actions and unsupported success claims."""

from contextlib import redirect_stdout
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_evaluation import audit
import check_environment
from check_environment import check_versions
import evaluation_metadata


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class AcceptanceTests(unittest.TestCase):
    def fixture(self, root, success=True):
        count = 51
        settings = {"trials": 1, "seeds": [1000], "max_steps": 3000, "image_mode": "pad"}
        result = {
            "seed": 1000, "success": success, "num_frames": count,
            "policy_actions_replaced": 0, "image_mode": "pad", "control_hz": 50,
            "active_release": success, "stable_success_steps": 50 if success else 0,
            "reason": "stable_released_placement" if success else "timeout",
            "initial_target_z": 0.7, "max_lift_m": 0.1, "video_frames": count,
        }
        write(root / "settings.json", settings)
        write(root / "summary.json", {
            "requested": 1, "completed": 1, "complete": True,
            "trials": [result], "successes": int(success), "success_rate": float(success),
        })
        write(root / "provenance.json", {
            "checkpoint_sha256": {"model.safetensors": "fixture"},
            "environment": {"python": "test-fixture"},
        })
        folder = root / "trial_000"
        folder.mkdir()
        write(folder / "result.json", result)
        # The video decoder is injected except in the real FFmpeg integration test.
        (folder / "cameras.mp4").write_bytes(b"unit-test-media")
        for name in ("head_cam", "camera_right"):
            Image.new("RGB", (640, 480), (17, 91, 203)).save(folder / f"initial_{name}.png")
        chunks = np.zeros((2, 50, 9), dtype=np.float32)
        chunks[:, :, 7:] = -0.7
        if success:
            chunks[0, 1:, 7:] = 0.5
            chunks[1, :, 7:] = 0.5
        target = np.zeros((count, 3))
        target[:, 2] = 0.8
        np.savez(folder / "trajectory.npz",
                 state=np.zeros((count, 9), dtype=np.float32),
                 action=chunks.reshape(-1, 9)[:count], target=target,
                 wall_clearance_m=np.full(count, 0.01),
                 basket_floor_contact=np.ones(count, dtype=bool),
                 released=np.ones(count, dtype=bool), object_speed=np.zeros(count),
                 stable_steps=np.arange(count) if success else np.zeros(count),
                 time=np.arange(1, count + 1) * 0.02,
                 complete_step=np.ones(count, dtype=bool))
        np.savez(folder / "predicted_chunks.npz", actions=chunks)
        return folder

    def video(self, path):
        return {"nb_read_frames": "51", "width": 1280, "height": 480,
                "avg_frame_rate": "50/1"}

    def test_audits_success_and_keeps_failed_trial_in_denominator(self):
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, success)
                report = audit(root, self.video)
                self.assertTrue(report["audit_passed"], report)
                self.assertEqual(report["reported_success_rate"], float(success))
                self.assertEqual(report["requested"], 1)
                self.assertIn("trial_000/cameras.mp4", report["files_sha256"])

    def test_rejects_partial_evaluation_and_filtered_trial_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            summary = json.loads((root / "summary.json").read_text())
            summary["complete"] = False
            write(root / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit(root, self.video)
            summary["complete"] = True
            write(root / "summary.json", summary)
            (root / "trial_001").mkdir()
            with self.assertRaisesRegex(ValueError, "additional"):
                audit(root, self.video)

    def test_rejects_video_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            def wrong_video(path):
                return {**self.video(path), "nb_read_frames": "50"}
            report = audit(root, wrong_video)
            self.assertFalse(report["audit_passed"])
            self.assertIn("frame count mismatch", report["errors"][0]["issue"])

    def test_real_ffmpeg_video_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.fixture(root)
            subprocess.run([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                "-i", "color=c=blue:s=1280x480:r=50", "-frames:v", "51",
                "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
                str(folder / "cameras.mp4"),
            ], check=True, timeout=30)
            report = audit(root)
            self.assertTrue(report["audit_passed"], report)

    def test_keeps_extra_prediction_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.fixture(root, success=False)
            path = folder / "trajectory.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key][:50] for key in data.files}
            np.savez(path, **arrays)
            result = json.loads((folder / "result.json").read_text())
            result.update(num_frames=50, video_frames=50, reason="joint_jump_abort")
            write(folder / "result.json", result)
            summary = json.loads((root / "summary.json").read_text())
            summary["trials"] = [result]
            write(root / "summary.json", summary)
            def video(path):
                return {**self.video(path), "nb_read_frames": "50"}
            report = audit(root, video)
            self.assertTrue(report["audit_passed"], report)
            self.assertEqual(report["reported_success_rate"], 0)

    def test_rejects_changed_action_and_false_success(self):
        for field in ("action", "released"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = self.fixture(root)
                path = folder / "trajectory.npz"
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                arrays[field][0] = 0 if field == "action" else False
                if field == "released":
                    arrays[field][:] = False
                np.savez(path, **arrays)
                report = audit(root, self.video)
                self.assertFalse(report["audit_passed"])

    def clipped_fixture(self, root):
        folder = self.fixture(root)
        settings = json.loads((root / "settings.json").read_text())
        settings.update(action_processing="clip", expert_takeover=False,
                        action_limits=[[-0.7, 0.5]] * 9)
        write(root / "settings.json", settings)
        with np.load(folder / "predicted_chunks.npz", allow_pickle=False) as data:
            chunks = data["actions"].copy()
        chunks[0, 0, 7:] = -0.71
        np.savez(folder / "predicted_chunks.npz", actions=chunks)
        path = folder / "trajectory.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["action_raw"] = chunks.reshape(-1, 9)[:51].copy()
        limits = np.asarray(settings["action_limits"], dtype=float)
        arrays["action"] = np.clip(arrays["action_raw"], limits[:, 0], limits[:, 1])
        np.savez(path, **arrays)
        result = json.loads((folder / "result.json").read_text())
        result.update(action_processing="clip", expert_takeover=False, policy_actions_replaced=1)
        write(folder / "result.json", result)
        summary = json.loads((root / "summary.json").read_text())
        summary["trials"] = [result]
        write(root / "summary.json", summary)
        return folder

    def test_audits_clipping_and_rejects_tampered_raw_or_executed_actions(self):
        for field in (None, "action_raw", "action"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = self.clipped_fixture(root)
                if field:
                    path = folder / "trajectory.npz"
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {key: data[key] for key in data.files}
                    arrays[field][0, 7] = -0.6
                    np.savez(path, **arrays)
                report = audit(root, self.video)
                self.assertEqual(report["audit_passed"], field is None, report)

    def test_rejects_false_clipping_count_and_expert_takeover(self):
        for field, value in (("policy_actions_replaced", 0), ("expert_takeover", True),
                             ("action_processing", "raw")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = self.clipped_fixture(root)
                result = json.loads((folder / "result.json").read_text())
                result[field] = value
                write(folder / "result.json", result)
                summary = json.loads((root / "summary.json").read_text())
                summary["trials"] = [result]
                write(root / "summary.json", summary)
                self.assertFalse(audit(root, self.video)["audit_passed"])

    def test_profile_checks_missing_and_mismatched_packages(self):
        def version(name):
            if name == "lerobot":
                raise importlib.metadata.PackageNotFoundError(name)
            return "0.0.0"
        _, issues = check_versions("training", version)
        by_name = {item["package"]: item for item in issues}
        self.assertIsNone(by_name["lerobot"]["installed"])
        self.assertEqual(by_name["pillow"]["required"], "==12.3.0")
        _, issues = check_versions("simulation", version)
        by_name = {item["package"]: item for item in issues}
        self.assertEqual(by_name["pillow"]["required"], "==12.0.0")

    def test_strict_cli_reports_missing_packages_and_fails(self):
        issue = {"package": "lerobot", "required": "==0.6.1", "installed": None}
        output = io.StringIO()
        with patch.object(sys, "argv", ["check", "--strict-versions"]), \
                patch.object(check_environment, "check_versions", return_value=({}, [issue])), \
                redirect_stdout(output), self.assertRaises(SystemExit) as error:
            check_environment.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(json.loads(output.getvalue())["version_issues"], [issue])

    def test_dependency_conflict_fails_even_when_pinned_versions_match(self):
        output = io.StringIO()
        failed = subprocess.CompletedProcess([], 1, "incompatible dependencies", "")
        with patch.object(sys, "argv", ["check", "--profile", "training", "--strict-versions"]), \
                patch.object(check_environment, "check_versions", return_value=({}, [])), \
                patch.object(check_environment.subprocess, "run", return_value=failed), \
                redirect_stdout(output), self.assertRaises(SystemExit) as error:
            check_environment.main()
        self.assertEqual(error.exception.code, 1)
        self.assertFalse(json.loads(output.getvalue())["pip_check_passed"])

    def test_provenance_hashes_actual_files_without_deployment_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, tokenizer = root / "checkpoint", root / "tokenizer"
            checkpoint.mkdir()
            tokenizer.mkdir()
            sources = json.loads(
                (evaluation_metadata.ROOT / "configs/sources.json").read_text(encoding="utf-8")
            )
            for name in sources["checkpoint"]["files"]:
                (checkpoint / name).write_bytes(b"test-weight-file")
            write(tokenizer / "tokenizer_config.json", {"fixture": True})
            with patch.object(evaluation_metadata, "collect", return_value={"python": "fixture"}):
                report = evaluation_metadata.provenance(checkpoint, tokenizer)
            self.assertFalse(report["matches_published_model"])
            self.assertEqual(report["checkpoint_sha256"]["model.safetensors"],
                             evaluation_metadata.sha256(checkpoint / "model.safetensors"))
            self.assertIn("tokenizer_config.json", report["tokenizer_sha256"])
            self.assertIn("simulation/evaluate.py", report["source_files_sha256"])
            self.assertNotIn(str(root), json.dumps(report))


if __name__ == "__main__":
    unittest.main()
