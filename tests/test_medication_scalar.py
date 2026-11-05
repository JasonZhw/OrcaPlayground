"""The scalar interface must preserve expert actions and both independent latches."""

import numpy as np
import pytest

from examples.euler.medication_sorting import execution as control


def test_scalar_reproduces_open_shape_and_hold_targets():
    assert hasattr(control, "scalar_to_ad"), "Single-value gripper interface is missing"
    np.testing.assert_allclose(
        control.scalar_to_ad(np.array([0, 0.2, 0.4, 0.55, 0.7, 0.85, 1])),
        [[0, -1], [0, -0.015], [0, 0.97], [0.14, 0.935], [0.28, 0.9], [0.28, 1.175], [0.28, 1.45]],
        atol=1e-12,
    )


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_closure_rejected(bad):
    with pytest.raises(ValueError):
        control.scalar_to_ad(bad)


def test_scalar_latches_keep_other_hand_closed_while_one_releases():
    names = ["arm"] + [
        f"g1_gripper_{side}_{part}_joint{joint}"
        for side in ("l", "r")
        for part in ("inner", "outer")
        for joint in (1, 3, 4, 2)
    ]
    grippers = control.Grippers(names)
    grippers.set("left", 1)
    grippers.set("right", 1)
    grippers.set("left", 0)
    out = grippers.apply(np.ones(17))
    np.testing.assert_array_equal(out[1:9], [0, 0, 0, -1, 0, 0, 0, -1])
    np.testing.assert_array_equal(out[9:], [0, 0, 0.28, 1.45, 0, 0, -0.28, 1.45])
    assert out[0] == 1
    assert grippers.closure == {"left": 0.0, "right": 1.0}
