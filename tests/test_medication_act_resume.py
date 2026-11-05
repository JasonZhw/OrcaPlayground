"""Protect optimizer continuity and disk reserve before expensive ACT updates."""
import importlib
import pytest


def continuation():
    return importlib.import_module("examples.euler.medication_sorting.act.continue_train")


def test_resume_rejects_reset_optimizer_and_wrong_counter():
    m = continuation()
    m.validate_resume(20000, [20000, 20000], 20000)
    with pytest.raises(ValueError):
        m.validate_resume(0, [20000], 20000)
    with pytest.raises(ValueError):
        m.validate_resume(20000, [0, 20000], 20000)
    with pytest.raises(ValueError):
        m.validate_resume(20000, [], 20000)


def test_disk_budget_includes_all_future_saves_and_reserve():
    m = continuation()
    m.validate_space(3_000, 600, 2, 1_000)
    with pytest.raises(RuntimeError):
        m.validate_space(2_100, 600, 2, 1_000)


def test_action_metrics_keep_gripper_error_separate():
    import numpy as np
    m = continuation()
    target = np.zeros((50, 16))
    pred = target.copy()
    pred[:, 7] = 1
    pred[:, 15] = .5
    result = m.action_metrics(pred, target)
    assert result["arm_mae_1"] == result["arm_mae_50"] == 0
    assert result["closure_mae_1"] == result["closure_mae_50"] == .75
    with pytest.raises(ValueError):
        m.action_metrics(pred[:1], target)
