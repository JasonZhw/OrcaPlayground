"""Asset-free tests for the blue-pick servo command transform."""
import unittest

import numpy as np

from examples.euler.tray_transfer.blue_env import compute_servo_control


class ServoControlTests(unittest.TestCase):
    def setUp(self):
        self.desired = np.array([.2, .5, .7])
        self.actual = np.array([.1, .6, .9])
        self.bias = np.array([2., -1., 100.])
        self.gains = np.array([10., 5., .01])
        self.limits = np.tile([-2., 2.], (3, 1))
        self.mask = np.array([True, True, False])

    def transform(self, **options):
        return compute_servo_control(
            self.desired, self.actual, self.bias, self.gains,
            self.limits, self.mask, **options,
        )

    def test_defaults_preserve_nominal_targets(self):
        control, clipped = self.transform()
        np.testing.assert_array_equal(control, self.desired)
        self.assertFalse(clipped.any())
        self.assertFalse(np.shares_memory(control, self.desired))

    def test_bias_and_feedback_formula_excludes_gripper(self):
        control, clipped = self.transform(
            bias_compensation=True, position_feedback_gain=3.,
        )
        np.testing.assert_allclose(control, [.6, .1, .7])
        self.assertFalse(clipped.any())

    def test_feedback_can_be_used_without_bias(self):
        control, _ = self.transform(position_feedback_gain=2.)
        np.testing.assert_allclose(control, [.3, .4, .7])

    def test_control_limits_clip_correction(self):
        self.bias[:2] = [100., -100.]
        control, clipped = self.transform(bias_compensation=True)
        np.testing.assert_array_equal(control, [2., -2., .7])
        np.testing.assert_array_equal(clipped, [True, True, False])

    def test_inputs_are_not_modified(self):
        inputs = [self.desired, self.actual, self.bias, self.gains, self.limits, self.mask]
        originals = [value.copy() for value in inputs]
        self.transform(bias_compensation=True, position_feedback_gain=4.)
        for actual, expected in zip(inputs, originals):
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_state_gain_and_shape_rejected(self):
        for gain in [0., -1., np.nan, np.inf]:
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                self.transform(position_feedback_gain=gain)
        self.actual[0] = np.nan
        with self.assertRaises(ValueError):
            self.transform()
        self.actual = np.zeros(4)
        with self.assertRaises(ValueError):
            self.transform()

    def test_invalid_kp_and_limits_rejected(self):
        self.gains[0] = 0.
        with self.assertRaises(ValueError):
            self.transform(bias_compensation=True)
        self.gains[0] = 10.
        self.limits[0] = [1., -1.]
        with self.assertRaises(ValueError):
            self.transform()


if __name__ == "__main__":
    unittest.main(verbosity=2)
