"""Check the published recipe cannot silently become LoRA or change its schedule."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_train", REPO / "scripts/prepare_train.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.saved = json.loads(prepare.SAVED_CONFIG.read_text(encoding="utf-8"))

    def metadata(self):
        features = {}
        for key, shape in prepare.EXPECTED_FEATURES.items():
            features[key] = {"shape": shape, "dtype": "float32"}
            if key.startswith("observation.images."):
                features[key].update(dtype="image", names=["channels", "height", "width"])
        return {"total_episodes": 200, "total_frames": 277084, "features": features,
                "codebase_version": "v3.0", "fps": 50, "video_path": None}

    def test_only_paths_and_reporting_change(self):
        generated = prepare.make_config(
            self.saved, Path("/data/demo"), "owner/data", "/models/base", Path("/outputs/run"),
        )
        self.assertEqual(generated["optimizer"], self.saved["optimizer"])
        self.assertEqual(generated["scheduler"], self.saved["scheduler"])
        self.assertEqual(generated["steps"], 30000)
        self.assertEqual(generated["dataset"]["image_transforms"],
                         self.saved["dataset"]["image_transforms"])
        self.assertEqual(generated["dataset"]["eval_split"], 0.0)
        self.assertIsNone(generated["wandb"]["run_id"])
        self.assertFalse(generated["wandb"]["enable"])
        before = dict(self.saved["policy"])
        after = dict(generated["policy"])
        before.pop("pretrained_path")
        after.pop("pretrained_path")
        before.pop("pretrained_revision")
        after.pop("pretrained_revision")
        self.assertEqual(before, after)
        self.assertEqual(self.saved["dataset"]["root"], "/content/data/v9")

    def test_refuses_lora_and_shortened_schedule(self):
        self.saved["policy"]["use_peft"] = True
        with self.assertRaises(ValueError):
            prepare.make_config(self.saved, Path("/data"), "owner/data", "base", Path("/out"))
        self.saved["policy"]["use_peft"] = False
        self.saved["steps"] = 25000
        with self.assertRaises(ValueError):
            prepare.make_config(self.saved, Path("/data"), "owner/data", "base", Path("/out"))

    def test_metadata_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            info = self.metadata()
            path = root / "meta/info.json"
            path.write_text(json.dumps(info), encoding="utf-8")
            prepare.validate_dataset(root)
            info["features"]["observation.images.image"].update(
                shape=[224, 224, 3], names=["height", "width", "channels"],
            )
            path.write_text(json.dumps(info), encoding="utf-8")
            prepare.validate_dataset(root)
            info["features"]["action"]["shape"] = [7]
            path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare.validate_dataset(root)

    def test_cli_prepares_without_launch_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            info = self.metadata()
            (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
            command = [
                sys.executable, str(REPO / "scripts/prepare_train.py"),
                "--dataset-root", str(root), "--dataset-id", "owner/data",
                "--base-model", "/models/base", "--output-dir", str(root / "run"),
                "--config-out", str(root / "generated.json"),
                "--tokenizer", str(root / "tokenizer"),
            ]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertFalse((root / "run").exists())
            generated = json.loads((root / "generated.json").read_text(encoding="utf-8"))
            self.assertEqual(generated["steps"], 30000)
            self.assertEqual(generated["policy"]["pretrained_path"], "/models/base")
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_hub_base_revision_is_pinned(self):
        generated = prepare.make_config(
            self.saved, Path("/data"), "owner/data", "lerobot/pi05_base", Path("/out"),
        )
        self.assertEqual(generated["policy"]["pretrained_revision"],
                         "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0")

    def test_rejects_old_video_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            info = self.metadata()
            info["codebase_version"] = "v2.1"
            (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare.validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
