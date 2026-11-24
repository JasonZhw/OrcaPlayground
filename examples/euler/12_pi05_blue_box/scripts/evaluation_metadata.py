"""Record public provenance without connection details or deployment paths."""

import hashlib
import json
from pathlib import Path
import subprocess

from collect_versions import collect


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(checkpoint, tokenizer):
    sources = json.loads((ROOT / "configs/sources.json").read_text(encoding="utf-8"))
    files = {name: sha256(checkpoint / name) for name in sources["checkpoint"]["files"]}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=15,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", "."],
            text=True, timeout=15, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {
        "git_commit": commit, "project_dirty": dirty,
        "source_files_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for directory in ("scripts", "simulation", "configs", "assets", "third_party")
            for path in sorted((ROOT / directory).rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json", ".xml", ".yml", ".urdf", ".stl"}
        },
        "checkpoint_sha256": files,
        "tokenizer_sha256": {
            path.name: sha256(path) for path in sorted(tokenizer.iterdir())
            if path.is_file() and (path.suffix in {".json", ".model"}
                                   or path.name.startswith("tokenizer"))
        },
        "matches_published_model": (
            files["model.safetensors"] == sources["checkpoint"]["model_sha256"]
        ),
        "layout_sha256": sha256(ROOT / "assets/scene_v9.json"),
        "source_physics_sha256": sha256(ROOT / "assets/scene_contact_v9.xml"),
        "visual_resources": json.loads(
            (ROOT / "configs/visual_resources.json").read_text(encoding="utf-8")
        ),
        "environment": collect(),
        "scope": "Provenance only; visual package contents and policy success are not certified.",
    }
