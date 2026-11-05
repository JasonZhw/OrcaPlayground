"""A physically successful but corrupt recording must stop a collection batch."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples.euler.medication_sorting.batch import main


@pytest.mark.parametrize("raw_valid", [False, True])
def test_batch_requires_valid_recording_not_only_physical_success(tmp_path, raw_valid):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index("--output") + 1])
        out.mkdir()
        (out / "result.json").write_text(json.dumps({"success": True, "recording": {"raw_valid": raw_valid}}))
        return SimpleNamespace(returncode=0)

    args = ["batch", "--execute", "--record", "--episodes", "2", "--output", str(tmp_path / "batch")]
    with patch("sys.argv", args), patch("subprocess.run", run):
        if raw_valid:
            main()
        else:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
    rows = json.loads((tmp_path / "batch/batch.json").read_text())
    assert len(calls) == len(rows) == (2 if raw_valid else 1)
    assert rows[0]["result"]["success"] is raw_valid
