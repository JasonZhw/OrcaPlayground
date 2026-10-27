"""Pose targets remain explicit and continuous during a held-object wrist turn."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from examples.euler.tray_transfer.plan_tray_transfer import TransportPlanner


class CartesianPosePlanner(TransportPlanner):
    """Six independent pose coordinates isolate IK and path interpolation."""

    def __init__(self):
        self.end = 0
        self.data = SimpleNamespace(xpos=np.zeros((1, 3)), xmat=np.eye(3).reshape(1, 9))
        self.end_rotation = np.eye(3)
        self.low = np.full(6, -2.)
        self.high = np.full(6, 2.)
        self.max_position_error = 0.
        self.max_rotation_error = 0.
        self.max_joint_increment = 0.
        self.checks = 0

    def fk(self, q):
        self.data.xpos[0] = q[:3]
        self.data.xmat[0] = Rotation.from_rotvec(q[3:]).as_matrix().reshape(9)

    def collisions(self, q, *, carried=True, target_fixed=None):
        self.checks += 1
        return []


class TestTrayTransportPlanner(unittest.TestCase):
    def test_rotation_is_interpolated_from_the_measured_pose_without_a_joint_jump(self):
        planner = CartesianPosePlanner()
        target = Rotation.from_euler("y", -5., degrees=True).as_matrix()
        phase, final_q = planner.segment(
            "transport_raise", np.zeros(6), np.zeros(3), np.zeros(3), 2.,
            start_rotation=np.eye(3), end_rotation=target)
        points = np.asarray(phase["q"])
        np.testing.assert_array_equal(points[0], np.zeros(6))
        self.assertLess(np.max(np.abs(np.diff(points, axis=0))), .005)
        np.testing.assert_allclose(final_q[:3], 0., atol=1e-8)
        np.testing.assert_allclose(Rotation.from_rotvec(final_q[3:]).as_matrix(), target,
                                   atol=1e-5)
        self.assertEqual(planner.checks, 4 * (len(points) - 1))

    def test_choosing_an_explicit_orientation_does_not_relax_unreachable_position(self):
        planner = CartesianPosePlanner()
        planner.high[0] = .2
        with self.assertRaisesRegex(ValueError, "IK target unreachable"):
            planner.solve([.25, 0., 0.], np.zeros(6), np.eye(3))


if __name__ == "__main__":
    unittest.main()
