"""Download one pinned source; never download all checkpoint copies."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = Path(__file__).resolve().parents[1] / "configs/sources.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("raw", "base", "tokenizer", "checkpoint"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    source = json.loads(SOURCES.read_text(encoding="utf-8"))[args.asset]
    patterns = None
    if args.asset == "checkpoint":
        patterns = list(source["files"])
    elif args.asset == "raw":
        patterns = ["raw/*/cameras.mp4", "raw/*/trajectory.npz", "raw/*/result.json"]
    elif args.asset == "tokenizer":
        patterns = ["*.json", "*.model", "tokenizer*",
                    "special_tokens_map.json", "added_tokens.json"]
    destination = Path(snapshot_download(
        repo_id=source["repo_id"], repo_type=source["repo_type"],
        revision=source["revision"], local_dir=args.destination.resolve(),
        allow_patterns=patterns,
    ))
    if args.asset == "checkpoint":
        for name, size in source["files"].items():
            if (destination / name).stat().st_size != size:
                raise ValueError(f"Unexpected checkpoint size: {name}")
        with (destination / "model.safetensors").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != source["model_sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
    print(f"Downloaded {args.asset} at revision {source['revision']}")


if __name__ == "__main__":
    main()
