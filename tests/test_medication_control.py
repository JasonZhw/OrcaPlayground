import unittest

import numpy as np

from examples.euler.medication_sorting.execution import Grippers


class GripperTests(unittest.TestCase):
    def setUp(self):
        self.names = ["arm"] + [
            f"unitree_g1_1_idx{i}_gripper_{s}_{part}_joint{j}"
            for s in ["l", "r"]
            for i, (part, j) in enumerate([(p, j) for p in ["inner", "outer"] for j in [1, 3, 4, 2]])
        ]
        self.g = Grippers(self.names)

    def test_both_hold_survive_arm_updates_until_release(self):
        self.g.set("left", 0.7)
        self.g.set("right", 1)
        out = self.g.apply(np.zeros(17))
        np.testing.assert_allclose(out[1:9], [0, 0, 0.28, 0.9, 0, 0, -0.28, 0.9])
        np.testing.assert_allclose(out[9:], [0, 0, 0.28, 1.45, 0, 0, -0.28, 1.45])
        self.g.set("left", 0)
        out = self.g.apply(np.ones(17))
        np.testing.assert_allclose(out[9:], [0, 0, 0.28, 1.45, 0, 0, -0.28, 1.45])
        self.assertEqual(out[0], 1)

    def test_missing_joint_rejected(self):
        with self.assertRaises(ValueError):
            Grippers(self.names[:-1])


if __name__ == "__main__":
    unittest.main()
