"""Verify tokenizer pinning preserves statistics and public inputs stay portable."""

import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingAdapterTests(unittest.TestCase):
    def test_pinning_preserves_existing_processor_overrides(self):
        train = load("train")
        received = {}
        def factory(*args, **kwargs):
            received.update(kwargs)
            return "pre", "post"
        stats = {"action": {"mean": [1.0]}}
        overrides = {"normalizer_processor": {"stats": stats},
                     "device_processor": {"device": "cuda"}}
        create = train.pinned_processor_factory(factory, Path("/models/tokenizer"))
        result = create(
            preprocessor_overrides=overrides,
            postprocessor_overrides={"unnormalizer_processor": {"stats": stats}},
            pretrained_revision="base-revision",
        )
        self.assertEqual(result, ("pre", "post"))
        self.assertIs(received["preprocessor_overrides"]["normalizer_processor"]["stats"], stats)
        self.assertEqual(received["preprocessor_overrides"]["device_processor"], {"device": "cuda"})
        self.assertEqual(received["preprocessor_overrides"]["tokenizer_processor"]["tokenizer_name"],
                         str(Path("/models/tokenizer")))
        self.assertNotIn("tokenizer_processor", overrides)
        self.assertEqual(received["pretrained_revision"], "base-revision")

    def test_public_files_pass_privacy_and_original_hash_checks(self):
        verifier = load("verify_public")
        report = verifier.validate_files()
        self.assertEqual(report["original_hashes"], 7)
        self.assertTrue(report["portable_xml_assets"])


if __name__ == "__main__":
    unittest.main()
