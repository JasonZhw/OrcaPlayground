"""Bind preparation artifacts to their source code and recorded dataset."""

import hashlib
import json
import uuid

REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(root):
    manifest = json.loads((root.parent / "source_files.json").read_text())
    if manifest.get("revision") != REVISION or not manifest.get("files"):
        raise ValueError("Invalid OpenPI source manifest")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"OpenPI source changed: {name}")


def verify_norm(assets, data_root, info):
    report = json.loads((assets / "norm_provenance.json").read_text())
    expected = dict(
        frames=info["total_frames"], episodes=info["total_episodes"], state_dim=30, action_dim=16, action_horizon=50
    )
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Normalization contract or dataset counts changed")
    if digest(assets / "norm_stats.json") != report.get("stats_sha256"):
        raise ValueError("Normalization statistics changed")
    sources = report.get("sources", [])
    if len(sources) != info["total_episodes"] or len({s["path"] for s in sources}) != len(sources):
        raise ValueError("Normalization source list is incomplete")
    for source in sources:
        from pathlib import Path

        path = Path(source["path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file() or digest(path) != source["sha256"]:
            raise ValueError(f"Normalization source changed: {path}")


def write_launch_audit(root, name, resume, audit):
    suffix = f"resume_{uuid.uuid4().hex}" if resume else "launch"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{suffix}.json"
    with path.open("x") as stream:
        json.dump(audit, stream, indent=2)
    return path
