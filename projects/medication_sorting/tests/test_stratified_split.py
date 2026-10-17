"""按物品分层的离线契约测试；synthetic 样例不代表实测或采集成功。"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.medication_sorting import convert_episodes as converter
from tools.medication_sorting import split_episodes as splitter
from tools.medication_sorting.task_specs import TARGETS
from tools.medication_sorting.tests.test_collection_quality import episode as synthetic_episode


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    """仅写带 synthetic 标记的元数据；质量结论由 verifier stub 提供。"""
    source = tmp_path / "synthetic_raw"
    source.mkdir()
    accepted, rejected, rechecked, scans = [], [], [], []

    def add(object_name, index, *, seed=None):
        path = source / f"episode_{len(accepted):06d}"
        path.mkdir()
        metadata = {"fps": 5, "object_name": object_name,
                    "seed": len(accepted) if seed is None else seed,
                    "randomization_profile": "tabletop_v1", "randomized_reset": True,
                    "test_only_synthetic_fixture": True}
        write_json(path / "metadata.json", metadata)
        row = {"path": str(path), "accepted": True, "status": "accepted",
               "content_sha256": hashlib.sha256(f"synthetic:{object_name}:{index}".encode()).hexdigest(),
               "stats": {"frames": 30, "fps": 5}, "errors": [], "warnings": ["synthetic verifier stub"]}
        accepted.append(row)
        return row

    for object_name in TARGETS:
        for index in range(40):
            add(object_name, index)

    def verify_dataset(path, *, validation_fraction):
        assert Path(path) == source
        assert 0 < validation_fraction < 1
        return {"accepted": deepcopy(accepted), "rejected": deepcopy(rejected), "ok": not rejected}

    def verify_episode(path):
        rechecked.append(str(path))
        return deepcopy(next(row for row in accepted if row["path"] == str(path)))

    def scan_episodes(path, *, fps, terminal_phase):
        assert fps == 5 and terminal_phase == "settle_final"
        links = sorted(Path(path).iterdir())
        assert all(link.is_symlink() and link.resolve().is_dir() for link in links)
        scans.append(Path(path).name)
        return [{"path": str(link)} for link in links], []

    monkeypatch.setattr(splitter, "verify_dataset", verify_dataset)
    monkeypatch.setattr(splitter, "verify_episode", verify_episode)
    monkeypatch.setattr(splitter, "scan_episodes", scan_episodes)
    return SimpleNamespace(source=source, output=tmp_path / "split", accepted=accepted,
                           rejected=rejected, rechecked=rechecked, scans=scans, add=add)


def stratify(dataset, **kwargs):
    return splitter.materialize_split(dataset.source, dataset.output,
                                      per_object_train=32, per_object_validation=8, **kwargs)


def change_metadata(row, **changes):
    path = Path(row["path"]) / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(changes)
    write_json(path, metadata)


def test_stratification_rejects_mixed_experiment_contract(synthetic_dataset):
    for row in synthetic_dataset.accepted:
        change_metadata(row, expert_profile='randomized_tabletop_70',
                        experiment_layout_sha256='a'*64,
                        randomization_config={'xy_half_range_m': [.02, .02]})
    change_metadata(synthetic_dataset.accepted[0], expert_profile='scaled_bottles_70')
    with pytest.raises(ValueError, match='实验'):
        stratify(synthetic_dataset)


def test_stratified_split_materializes_exact_counts_and_manifest(synthetic_dataset):
    dataset = synthetic_dataset
    before = {row["path"]: (Path(row["path"]) / "metadata.json").read_bytes()
              for row in dataset.accepted}
    report = stratify(dataset)
    assert report["status"] == "materialized"
    assert report["train_unique_groups"] == len(report["train"]) == 128
    assert report["validation_unique_groups"] == len(report["validation"]) == 32
    assert set(report["per_object"]) == set(TARGETS)
    for name in TARGETS:
        assert report["per_object"][name] == {"accepted": 40, "unique_groups": 40,
                                              "train": 32, "validation": 8}
        rows = [row for row in dataset.accepted
                if json.loads((Path(row["path"]) / "metadata.json").read_text())["object_name"] == name]
        hashes = sorted(row["content_sha256"] for row in rows)
        assert [row["content_sha256"] for row in report["train"] if row["object_name"] == name] == hashes[:32]
        assert [row["content_sha256"] for row in report["validation"] if row["object_name"] == name] == hashes[32:]
    for partition in ("train", "validation"):
        for row in report[partition]:
            link = Path(row["link"])
            assert link.parent == dataset.output / partition
            assert link.is_symlink() and link.resolve() == Path(row["source"])
            metadata = json.loads((link / "metadata.json").read_text())
            for key in ("object_name", "seed", "randomization_profile"):
                assert row[key] == metadata[key]
            assert row["frames"] == 30 and row["warnings"] == ["synthetic verifier stub"]
            assert (link / "metadata.json").read_bytes() == before[row["source"]]
    assert {row["seed"] for row in report["train"]}.isdisjoint(row["seed"] for row in report["validation"])
    assert {row["content_sha256"] for row in report["train"]}.isdisjoint(
        row["content_sha256"] for row in report["validation"])
    assert sorted(dataset.rechecked) == sorted(row["path"] for row in dataset.accepted)
    assert dataset.scans == ["train", "validation"]
    assert report["training_loader_split_applied"] is False
    assert report["conversion"]["norm_stats_repo_id"] == report["conversion"]["train_repo_id"]
    assert report["conversion"]["train_repo_id"] != report["conversion"]["validation_repo_id"]
    assert json.loads((dataset.output / "split_manifest.json").read_text()) == report


def test_stratified_selection_is_independent_of_scan_order(synthetic_dataset):
    dataset = synthetic_dataset
    first = stratify(dataset)
    dataset.accepted.reverse()
    dataset.output = dataset.output.with_name("split_reversed")
    second = stratify(dataset, validation_fraction=0.49)
    for partition in ("train", "validation"):
        assert [row["source"] for row in first[partition]] == [row["source"] for row in second[partition]]


@pytest.mark.parametrize("kind", ["missing", "under", "over", "unbalanced"])
def test_stratified_requires_exactly_forty_per_target(synthetic_dataset, kind):
    dataset = synthetic_dataset
    first = next(iter(TARGETS))
    if kind == "missing":
        del dataset.accepted[:40]
    elif kind == "under":
        dataset.accepted.pop()
    elif kind == "over":
        dataset.add(first, 40)
    else:
        change_metadata(dataset.accepted[40], object_name=first)
    with pytest.raises(ValueError, match="40"):
        stratify(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize("cross_object", [False, True])
def test_stratified_rejects_duplicate_content_not_silently_grouping(synthetic_dataset, cross_object):
    dataset = synthetic_dataset
    other = 40 if cross_object else 1
    dataset.accepted[other]["content_sha256"] = dataset.accepted[0]["content_sha256"]
    with pytest.raises(ValueError, match="重复.*(内容|content_sha256)"):
        stratify(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize("location", ["same_partition", "within_object", "across_objects"])
def test_stratified_rejects_seed_conflicts_globally(synthetic_dataset, location):
    dataset = synthetic_dataset
    first = sorted(dataset.accepted[:40], key=lambda row: row["content_sha256"])
    other = first[1] if location == "same_partition" else first[-1]
    if location == "across_objects":
        other = sorted(dataset.accepted[40:80], key=lambda row: row["content_sha256"])[-1]
    seed = json.loads((Path(first[0]["path"]) / "metadata.json").read_text())["seed"]
    change_metadata(other, seed=seed)
    with pytest.raises(ValueError, match="seed"):
        stratify(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize(("key", "value"), [
    ("object_name", None), ("object_name", "unknown"), ("object_name", []),
    ("seed", None), ("seed", True), ("seed", -1), ("seed", 1.0), ("seed", "1"),
    ("randomization_profile", None), ("randomization_profile", ""),
    ("randomization_profile", "   "), ("randomization_profile", []),
    ("randomization_profile", "legacy"),
])
def test_stratified_rejects_invalid_or_inconsistent_metadata(synthetic_dataset, key, value):
    dataset = synthetic_dataset
    change_metadata(dataset.accepted[-1], **{key: value})
    with pytest.raises(ValueError, match=key):
        stratify(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize("key", ["object_name", "seed", "randomization_profile"])
def test_stratified_requires_explicit_metadata_fields(synthetic_dataset, key):
    dataset = synthetic_dataset
    path = Path(dataset.accepted[0]["path"]) / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata[key]
    write_json(path, metadata)
    with pytest.raises(ValueError, match=key):
        stratify(dataset)
    assert not dataset.output.exists()


def test_stratified_rejects_non_object_metadata(synthetic_dataset):
    dataset = synthetic_dataset
    write_json(Path(dataset.accepted[0]["path"]) / "metadata.json", [])
    with pytest.raises(ValueError, match="metadata"):
        stratify(dataset)
    assert not dataset.output.exists()


def test_stratified_ignores_rejected_rows_and_preserves_exclusions(synthetic_dataset):
    dataset = synthetic_dataset
    failed = dataset.source / "episode_rejected_synthetic"
    failed.mkdir()
    dataset.rejected.append({"path": str(failed), "accepted": False,
                             "errors": ["synthetic rejection: no metadata or frames"]})
    report = stratify(dataset)
    assert report["excluded"] == dataset.rejected
    assert str(failed) not in dataset.rechecked
    assert len(report["train"]) + len(report["validation"]) == 160


def test_rejected_episode_cannot_fill_stratum_quota(synthetic_dataset):
    dataset = synthetic_dataset
    row = dataset.accepted.pop()
    row.update(accepted=False, errors=["synthetic rejection"])
    dataset.rejected.append(row)
    with pytest.raises(ValueError, match="40"):
        stratify(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize("kind", ["rejected", "content_sha256", "object_name", "seed", "randomization_profile"])
def test_stratified_rechecks_quality_content_and_metadata_before_output(synthetic_dataset, monkeypatch, kind):
    dataset = synthetic_dataset
    original = splitter.verify_episode
    row = dataset.accepted[0]

    def changed(path):
        current = original(path)
        if str(path) == row["path"]:
            if kind == "rejected":
                current["accepted"] = False
            elif kind == "content_sha256":
                current["content_sha256"] = "f" * 64
            else:
                value = {"object_name": list(TARGETS)[1], "seed": 99999,
                         "randomization_profile": "changed"}[kind]
                change_metadata(row, **{kind: value})
        return current

    monkeypatch.setattr(splitter, "verify_episode", changed)
    with pytest.raises(ValueError, match="变化|不再合格|重新验收"):
        stratify(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize(("train", "validation"), [
    (32, None), (None, 8), (31, 8), (32, 7), (40, 0), (0, 40),
    (True, 8), (32.0, 8), (32, False), (32, 8.0),
])
def test_stratified_api_requires_paired_fixed_counts(synthetic_dataset, train, validation):
    dataset = synthetic_dataset
    with pytest.raises(ValueError, match="同时|32|8"):
        splitter.materialize_split(dataset.source, dataset.output,
                                   per_object_train=train, per_object_validation=validation)
    assert not dataset.output.exists()


@pytest.mark.parametrize("arguments", [[], ["--per-object-train", "32", "--per-object-validation", "8"]])
def test_cli_preserves_default_and_enables_stratification(synthetic_dataset, capsys, arguments):
    dataset = synthetic_dataset
    result = splitter.main(["--input", str(dataset.source), "--output", str(dataset.output), *arguments])
    report = json.loads(capsys.readouterr().out)
    assert result == 0 and report["status"] == "materialized"
    assert ("per_object" in report) is bool(arguments)


@pytest.mark.parametrize("arguments", [
    ["--per-object-train", "32"], ["--per-object-validation", "8"],
    ["--per-object-train", "31", "--per-object-validation", "8"],
])
def test_cli_rejects_incomplete_or_wrong_counts(synthetic_dataset, capsys, arguments):
    dataset = synthetic_dataset
    result = splitter.main(["--input", str(dataset.source), "--output", str(dataset.output), *arguments])
    report = json.loads(capsys.readouterr().out)
    assert result == 1 and report["status"] == "blocked"
    assert report["output_created"] is False
    assert not dataset.output.exists()


def test_default_api_still_groups_duplicates_without_stratification_metadata(synthetic_dataset):
    dataset = synthetic_dataset
    dataset.accepted[:] = dataset.accepted[:22]
    dataset.accepted[-1]["content_sha256"] = dataset.accepted[0]["content_sha256"]
    for row in dataset.accepted:
        write_json(Path(row["path"]) / "metadata.json", {"fps": 5, "test_only_synthetic_fixture": True})
    report = splitter.materialize_split(dataset.source, dataset.output)
    assert report["train_unique_groups"] == 20 and report["validation_unique_groups"] == 1
    assert "per_object" not in report
    assert len(report["train"]) + len(report["validation"]) == 22
    for partition in ("train", "validation"):
        assert all("object_name" not in row for row in report[partition])
    hashes = {partition: {row["content_sha256"] for row in report[partition]}
              for partition in ("train", "validation")}
    assert hashes["train"].isdisjoint(hashes["validation"])


def test_stratified_keeps_output_safety_and_rolls_back_failed_preflight(synthetic_dataset, monkeypatch):
    dataset = synthetic_dataset
    dataset.output.mkdir()
    marker = dataset.output / "keep.txt"
    marker.write_text("synthetic existing output")
    with pytest.raises(FileExistsError):
        stratify(dataset)
    assert marker.read_text() == "synthetic existing output"
    dataset.output = dataset.source / "nested_split"
    with pytest.raises(ValueError, match="输入目录"):
        stratify(dataset)
    assert not dataset.output.exists()
    dataset.output = dataset.source.parent / "bad_preflight"
    monkeypatch.setattr(splitter, "scan_episodes", lambda *a, **k: ([], [{"reason": "synthetic rejection"}]))
    with pytest.raises(ValueError, match="预检"):
        stratify(dataset)
    assert not dataset.output.exists() and dataset.source.is_dir()


def test_stratified_links_work_with_real_scan_and_converter(synthetic_dataset, synthetic_episode, monkeypatch):
    """复用一份既有 synthetic fixture；不制造 160 份 PNG/NPZ 或伪造实测证据。"""
    dataset = synthetic_dataset
    template = converter.load_episode(synthetic_episode, fps=5, terminal_phase="settle_final")

    def synthetic_load(path, **kwargs):
        assert kwargs["fps"] == 5 and kwargs["terminal_phase"] == "settle_final"
        metadata = {**template.metadata, **json.loads((Path(path) / "metadata.json").read_text())}
        return replace(template, path=Path(path), metadata=metadata)

    class SyntheticWriter:
        @classmethod
        def create(cls, **kwargs):
            Path(kwargs["root"]).mkdir()
            return cls()

        def add_frame(self, frame):
            assert frame["observation.state"].shape == (8,)

        def save_episode(self):
            pass

        def stop_image_writer(self):
            pass

    monkeypatch.setattr(converter, "load_episode", synthetic_load)
    monkeypatch.setattr(splitter, "scan_episodes", converter.scan_episodes)
    report = stratify(dataset)
    for partition, count in (("train", 128), ("validation", 32)):
        result = converter.convert_episodes(dataset.output / partition,
                                            dataset.output.parent / f"converted_{partition}",
                                            repo_id=report["conversion"][f"{partition}_repo_id"],
                                            fps=report["fps"], terminal_phase="settle_final",
                                            dataset_class=SyntheticWriter)
        assert result["converted_episodes"] == count
        assert result["converted_frames"] == count * template.length
        assert result["rejected"] == []
