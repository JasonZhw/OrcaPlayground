"""Reject lifted but held, wall-overlapping and unstable placement results."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from placement import stable_frame


class PlacementTests(unittest.TestCase):
    def test_success_requires_every_physical_condition(self):
        self.assertTrue(stable_frame(0.09, 0.01, True, True, 0.01))
        for sample in (
            (0.07, 0.01, True, True, 0.01),
            (0.09, -0.002, True, True, 0.01),
            (0.09, 0.01, False, True, 0.01),
            (0.09, 0.01, True, False, 0.01),
            (0.09, 0.01, True, True, 0.03),
        ):
            self.assertFalse(stable_frame(*sample))


if __name__ == "__main__":
    unittest.main()
