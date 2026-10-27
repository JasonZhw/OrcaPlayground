"""Bad contact-force evidence must not become a released-placement success."""

import unittest
from types import SimpleNamespace

import numpy as np

from examples.euler.tray_transfer.placement_sample import sample_placement


class ContactEvidenceEnv:
    """A stationary part with valid floor support and one suspect gripper contact."""

    model = SimpleNamespace(nv=1)
    data = SimpleNamespace(qvel=np.zeros(1))

    def __init__(self, gripper_force):
        self.gripper_force = gripper_force

    def read_pose(self, body):
        return np.zeros(3), np.array([1., 0., 0., 0.])

    def query_contact_simple(self):
        return [{"geom1": 0, "geom2": 1}, {"geom1": 0, "geom2": 2}]

    def query_contact_force(self, indices):
        return {0: np.array([8., 0., 0., 0., 0., 0.]),
                1: np.array([self.gripper_force, 0., 0., 0., 0., 0.])}

    def mj_jacBody(self, jacp, jacr, obj):
        pass


class BluePlacementSampleTests(unittest.TestCase):
    def test_nonfinite_gripper_force_is_rejected_even_with_valid_floor_support(self):
        geometry = {
            "object_body": "blue", "body_name": "tray", "floor_id": 1, "floor_top": 0.,
            "inner_xy_min": [-.25, -.165], "inner_xy_max": [.25, .165],
            "geom_bodies": {0: "blue", 1: "tray", 2: "right_gripper_tip"},
            "object_geoms": [{
                "geom_type": 6, "geom_size": [.02, .05, .06],
                "geom_pos": [0., 0., .06], "geom_quat": [1., 0., 0., 0.],
                "geom_contype": 1, "geom_conaffinity": 1,
            }],
        }
        for force in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(gripper_force=force), self.assertRaises(ValueError):
                sample_placement(ContactEvidenceEnv(force), geometry)


if __name__ == "__main__":
    unittest.main()
