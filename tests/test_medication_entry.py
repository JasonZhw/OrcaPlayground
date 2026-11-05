"""One public entry dispatches collection, inspection and motion-only trials."""

import importlib
import sys

import pytest


def test_default_entry_collects_current_layout(tmp_path, monkeypatch):
    entry = importlib.import_module("examples.euler.medication_sorting.main")
    calls = []
    monkeypatch.setattr(entry, "collect_one", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(sys, "argv", ["main", "--dataset-root", str(tmp_path), "--execute"])
    entry.main()
    assert calls[0][0][0] == tmp_path
    assert calls[0][1]["max_bytes"] == 25_000_000_000


@pytest.mark.parametrize("mode,execute", [("inspect", False), ("run", True)])
def test_trial_modes_dispatch_without_collection(tmp_path, monkeypatch, mode, execute):
    entry = importlib.import_module("examples.euler.medication_sorting.main")
    workflow = importlib.import_module("examples.euler.medication_sorting.workflow")
    calls = []
    monkeypatch.setattr(workflow, "run_trial", calls.append)
    args = ["main", "--mode", mode, "--output", str(tmp_path / "trial")]
    if execute:
        args.append("--execute")
    monkeypatch.setattr(sys, "argv", args)
    entry.main()
    assert calls[0].execute is execute
    assert calls[0].layout_mode == "current"
    assert calls[0].task_index == 0
    assert calls[0].record is False


def test_motion_mode_requires_execute(tmp_path, monkeypatch):
    entry = importlib.import_module("examples.euler.medication_sorting.main")
    monkeypatch.setattr(sys, "argv", ["main", "--mode", "run", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 2
