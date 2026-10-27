"""Integration check requiring the separate pinned LeRobot conversion environment."""
import importlib.util
import json
import unittest

from examples.euler.tray_transfer.lerobot_export import export
from examples.euler.tray_transfer.validate_lerobot import validate
from tests import test_tray_lerobot_export as fixtures


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "requires pinned LeRobot environment")
class LeRobotRoundtripTests(unittest.TestCase):
    def test_roundtrip_and_incomplete_manifest_rejected(self):
        fixture = fixtures.TrayExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        parent = fixture.root.parent
        manifest = parent / "manifest.json"
        manifest.write_text(json.dumps({"episodes": [{"path": fixture.root.name}]}))
        output = parent / "converted"
        export(manifest, parent, output, "local/test")
        self.assertTrue(validate(output, parent)["passed"])
        provenance_path = output / "provenance/export.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["manifest"]["episodes"].append({"path": "missing_episode"})
        provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate(output, parent)


if __name__ == "__main__":
    unittest.main()
