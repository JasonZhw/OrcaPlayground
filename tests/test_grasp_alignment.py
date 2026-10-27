"""Contact-coordinate regressions for expert-grasp calibration."""

import unittest

import numpy as np


class GraspAlignmentTests(unittest.TestCase):
    def analyze(self, *args):
        from examples.euler.tray_transfer import grasp_alignment

        return grasp_alignment.analyze_contacts(*args)

    def test_rotated_translated_object_preserves_local_grasp_error(self):
        position = np.array([30.0, -7.0, 0.65])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        local = np.array([[-0.0527, -0.018, 0.0675], [-0.0527, 0.018, 0.0841]])
        contacts = [
            dict(other_body=tip, normal_force_n=1.0, position_world_m=(position + rotation @ point).tolist())
            for tip, point in zip(("inner", "outer"), local)
        ]
        result = self.analyze(
            position, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], [0, 0, 0.0517], contacts, ("inner", "outer")
        )
        self.assertTrue(result["bilateral_loaded_contact"])
        np.testing.assert_allclose(result["midpoint_minus_com_body_m"], [-0.0527, 0, 0.0241], atol=1e-12)
        np.testing.assert_allclose(result["translation_to_com_world_m"], [0, 0.0527, -0.0241], atol=1e-12)
        self.assertAlmostEqual(result["horizontal_offset_world_m"], 0.0527)
        self.assertAlmostEqual(result["contact_height_difference_world_m"], 0.0166)

    def test_each_finger_gets_its_own_force_weighted_contact_center(self):
        contacts = [
            dict(other_body=tip, normal_force_n=force, position_world_m=point)
            for tip, force, point in [
                ("inner", 1.0, [-0.02, -0.02, 0.05]),
                ("inner", 3.0, [0.02, -0.02, 0.05]),
                ("outer", 1.0, [0.01, 0.02, 0.05]),
                ("table", 10.0, [0, 0, 0]),
            ]
        ]
        result = self.analyze([0, 0, 0], [1, 0, 0, 0], [0, 0, 0.05], contacts, ("inner", "outer"))
        np.testing.assert_allclose(result["midpoint_minus_com_body_m"], [0.01, 0, 0])

    def test_unloaded_collision_is_not_evidence_of_bilateral_grasp(self):
        contacts = [
            dict(other_body="inner", normal_force_n=1.0, position_world_m=[0, -0.02, 0]),
            dict(other_body="outer", normal_force_n=0.0, position_world_m=[0, 0.02, 0]),
        ]
        result = self.analyze([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], contacts, ("inner", "outer"))
        self.assertFalse(result["bilateral_loaded_contact"])
        self.assertNotIn("translation_to_com_world_m", result)


if __name__ == "__main__":
    unittest.main()
