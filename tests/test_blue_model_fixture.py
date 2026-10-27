"""The torso fixture is explicit and must not alter the user's scene file."""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from examples.euler.tray_transfer.blue_pick import prepare_model, run


class WaistFixtureTests(unittest.TestCase):
    def test_original_has_no_waist_lock_and_source_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.xml'
            text = '<mujoco><worldbody><body name="torso">' + ''.join(
                f'<joint name="unitree_g1_1_waist_{axis}_joint"/>' for axis in ('yaw', 'roll', 'pitch')
            ) + '</body></worldbody></mujoco>'
            source.write_text(text)
            original = prepare_model(source, root / 'original.xml', 'original')
            locked = prepare_model(source, root / 'locked.xml', 'original', fix_waist=True)
            self.assertEqual(source.read_text(), text)
            self.assertEqual(original['changes'], [])
            self.assertIsNone(ET.parse(root / 'original.xml').find('equality'))
            constraints = ET.parse(root / 'locked.xml').findall('./equality/joint')
            self.assertEqual({c.get('joint1') for c in constraints}, {
                f'unitree_g1_1_waist_{axis}_joint' for axis in ('yaw', 'roll', 'pitch')
            })
            self.assertEqual(len(locked['changes']), 3)
            self.assertTrue(all(c.get('polycoef') == '0 0 0 0 0' for c in constraints))

    def test_fixture_rejects_missing_waist_joint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.xml'
            source.write_text('<mujoco><worldbody/></mujoco>')
            with self.assertRaisesRegex(ValueError, 'Waist fixture joint missing'):
                prepare_model(source, root / 'locked.xml', 'original', fix_waist=True)

    def test_setup_failure_records_explicit_fixture_and_failed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'schema_version': 1}))
            output = root / 'failed_run'
            with self.assertRaisesRegex(ValueError, 'explicit joint2'):
                run(root / 'unused.xml', 'original', output, plan, fix_waist=True)
            result = json.loads((output / 'result.json').read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['training_data'])
            self.assertTrue(result['waist_fixed_test_fixture'])
            self.assertIn('explicit joint2', result['error'])


if __name__ == '__main__':
    unittest.main()
