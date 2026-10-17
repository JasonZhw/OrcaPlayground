import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.medication_sorting.check_training_readiness import run_checks
from tools.medication_sorting import check_training_readiness as readiness


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_lerobot_info(root: Path, *, state_dim: int = 8, action_dim: int = 8, image_keys: tuple[str, ...] = ()) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [action_dim]},
    }
    for key in image_keys:
        features[key] = {"dtype": "video", "shape": [3, 224, 224]}
    _write_json(root / "meta/info.json", {"fps": 20, "features": features})


def test_missing_openpi_repository_is_a_blocker(tmp_path: Path):
    result = run_checks(
        project_root=tmp_path,
        repo_dir=tmp_path / "software/openpi",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoint",
        venv_dir=tmp_path / ".venv",
    )

    assert result["ready"] is False
    assert "repository_missing" in {item["code"] for item in result["blockers"]}
    assert result["execution"]["training_started"] is False


def test_repository_without_resolvable_commit_is_a_blocker(tmp_path: Path):
    repo = tmp_path / "software/openpi"
    (repo / ".git").mkdir(parents=True)
    (repo / "README.md").write_text("openpi", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nrequires-python = '>=3.11'\n", encoding="utf-8")

    result = run_checks(
        project_root=tmp_path,
        repo_dir=repo,
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoint",
        venv_dir=tmp_path / ".venv",
    )

    assert "repository_commit_missing" in {item["code"] for item in result["blockers"]}


def test_dataset_without_lerobot_metadata_is_a_blocker(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = run_checks(
        project_root=tmp_path,
        repo_dir=tmp_path / "software/openpi",
        data_dir=data_dir,
        checkpoint_dir=tmp_path / "checkpoint",
        venv_dir=tmp_path / ".venv",
    )

    assert "dataset_metadata_missing" in {item["code"] for item in result["blockers"]}


def test_dataset_requires_two_rgb_views_and_matching_state_action_dims(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_lerobot_info(data_dir, state_dim=8, action_dim=7, image_keys=("observation.images.head",))

    result = run_checks(
        project_root=tmp_path,
        repo_dir=tmp_path / "software/openpi",
        data_dir=data_dir,
        checkpoint_dir=tmp_path / "checkpoint",
        venv_dir=tmp_path / ".venv",
    )

    blocker_codes = {item["code"] for item in result["blockers"]}
    assert "dataset_rgb_views_insufficient" in blocker_codes
    assert "dataset_state_action_dim_mismatch" in blocker_codes


def test_checkpoint_must_be_local_and_include_norm_stats(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)

    result = run_checks(
        project_root=tmp_path,
        repo_dir=tmp_path / "software/openpi",
        data_dir=tmp_path / "data",
        checkpoint_dir=checkpoint,
        venv_dir=tmp_path / ".venv",
    )

    assert "checkpoint_norm_stats_missing" in {item["code"] for item in result["blockers"]}
    assert result["execution"]["weights_downloaded"] is False


def _check(tmp_path: Path, **overrides):
    args = {
        "project_root": tmp_path,
        "repo_dir": tmp_path / "software/openpi",
        "data_dir": tmp_path / "data",
        "checkpoint_dir": tmp_path / "checkpoint",
        "venv_dir": tmp_path / ".venv",
    }
    args.update(overrides)
    return run_checks(**args)


def test_remote_checkpoint_is_reported_without_downloading(tmp_path: Path):
    result = _check(tmp_path, checkpoint_dir="gs://openpi-assets/checkpoints/pi05_base")
    assert "checkpoint_remote_unchecked" in {item["code"] for item in result["blockers"]}
    assert result["execution"]["weights_downloaded"] is False


def test_empty_params_directory_is_not_a_checkpoint(tmp_path: Path):
    (tmp_path / "checkpoint/params").mkdir(parents=True)
    result = _check(tmp_path)
    assert result["checkpoint"]["weights_present"] is False
    assert "checkpoint_weights_missing" in {item["code"] for item in result["blockers"]}


def test_malformed_metadata_returns_json_blocker(tmp_path: Path):
    path = tmp_path / "data/meta/info.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    result = _check(tmp_path)
    assert "dataset_metadata_invalid" in {item["code"] for item in result["blockers"]}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("shape", [[0], [-1], [True], [33]])
def test_invalid_or_oversized_dimensions_are_blocked(tmp_path: Path, shape):
    _write_json(tmp_path / "data/meta/info.json", {
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": shape},
            "action": {"dtype": "float32", "shape": shape},
        },
    })
    result = _check(tmp_path)
    assert any(item["code"] in {"dataset_state_feature_missing", "dataset_action_dim_exceeds_pi05"}
               for item in result["blockers"])


def test_two_non_rgb_images_do_not_satisfy_rgb_requirement(tmp_path: Path):
    _write_json(tmp_path / "data/meta/info.json", {
        "fps": 20,
        "features": {
            "observation.images.head": {"dtype": "video", "shape": [224, 224, 1]},
            "observation.images.wrist": {"dtype": "video", "shape": [224, 224, 1]},
        },
    })
    result = _check(tmp_path)
    assert "dataset_rgb_views_insufficient" in {item["code"] for item in result["blockers"]}


def test_names_do_not_claim_runtime_mapping_verified(tmp_path: Path):
    names = [f"joint_{i}" for i in range(7)] + ["gripper"]
    _write_json(tmp_path / "data/meta/info.json", {
        "fps": 20,
        "features": {key: {"dtype": "float32", "shape": [8], "names": names}
                     for key in ["observation.state", "action"]},
    })
    result = _check(tmp_path)
    assert result["dataset"]["custom_g1_mapping"]["status"] != "verified"


def test_empty_data_directory_is_not_records(tmp_path: Path):
    _write_lerobot_info(tmp_path / "data")
    (tmp_path / "data/data").mkdir()
    result = _check(tmp_path)
    assert "dataset_records_missing" in {item["code"] for item in result["blockers"]}


def test_installed_dependencies_with_conflicting_versions_are_blocked(tmp_path: Path, monkeypatch):
    repo = tmp_path / "software/openpi"
    repo.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        "[project]\nrequires-python = '>=3.11'\ndependencies = ['torch==2.7.1', 'numpy<2.0.0']\n",
        encoding="utf-8",
    )
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    def probe(args, **kwargs):
        if args[0] == str(python):
            packages = {name: "0.1.0" for name in readiness.REQUIRED_PACKAGES}
            packages.update(torch="2.14.1", numpy="2.2.6")
            return subprocess.CompletedProcess(args, 0, json.dumps({"python": "3.12.14", "packages": packages}), "")
        return subprocess.CompletedProcess(args, 1, "", "not a Git repository")
    monkeypatch.setattr(readiness.subprocess, "run", probe)
    result = _check(tmp_path)
    conflicts = [item for item in result["blockers"] if item["code"] == "dependency_conflict"]
    assert len(conflicts) == 2
    assert result["environment"]["status"] == "blocked"


def test_checkpoint_invalid_norm_stats_are_a_blocker(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test-only-invalid-weight-content")
    _write_json(checkpoint / "assets/g1/norm_stats.json", {"norm_stats": {"state": {}, "actions": {}}})
    result = _check(tmp_path)
    assert "checkpoint_norm_stats_invalid" in {item["code"] for item in result["blockers"]}


def test_cli_outputs_json_and_nonzero_exit_for_blockers(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(Path(readiness.__file__)),
         '--project-root', str(tmp_path), '--checkpoint', 'gs://openpi-assets/checkpoints/pi05_base'],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report['ready'] is False
    assert report['execution']['gpu_touched'] is False
    assert result.stderr == ''


def test_json_with_nonfinite_metadata_is_rejected(tmp_path: Path):
    _write_json(tmp_path / "data/meta/info.json", {"fps": 20, "features": {"bad": float("nan")}})
    result = _check(tmp_path)
    json.dumps(result, allow_nan=False)
    assert "dataset_metadata_invalid" in {item["code"] for item in result["blockers"]}


@pytest.mark.parametrize("dirty", [False, True])
def test_changed_repository_cannot_claim_pinned_source(tmp_path: Path, monkeypatch, dirty):
    repo = tmp_path / "software/openpi"
    (repo / ".git").mkdir(parents=True)
    def git_probe(args, **kwargs):
        if 'rev-parse' in args:
            commit = readiness.PINNED_COMMIT if dirty else 'a' * 40
            return subprocess.CompletedProcess(args, 0, commit + '\n', '')
        if 'status' in args and '--porcelain' in args:
            return subprocess.CompletedProcess(args, 0, ' M pyproject.toml\n' if dirty else '', '')
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(readiness.subprocess, 'run', git_probe)
    result = _check(tmp_path)
    expected = 'repository_dirty' if dirty else 'repository_commit_mismatch'
    assert expected in {item['code'] for item in result['blockers']}
