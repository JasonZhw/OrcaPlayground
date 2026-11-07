"""Train π0.5 from base weights with the corrected action transform."""

import json
from pathlib import Path
import sys

sys.path.insert(0, "/root/gpufree-data/SY/openpi/scripts")

import jax
import train

from fixed_v3_base_delta_fix1_config import get_config


ROOT = Path("/root/gpufree-data/SY")
DATASET = ROOT / "lerobot/blue_intel_fixed_v3"
RAW = ROOT / "datasets/blue_intel_fixed_v3/raw"


def verify_dataset():
    summary = json.loads((DATASET / "conversion_summary.json").read_text())
    info = json.loads((DATASET / "meta/info.json").read_text())
    if summary.get("selection") != "fixed_v3_only":
        raise RuntimeError("Dataset selection is not fixed_v3_only")
    if summary.get("episodes") != 30 or info.get("total_episodes") != 30:
        raise RuntimeError("Training dataset must contain exactly 30 episodes")
    if summary.get("frames") != info.get("total_frames"):
        raise RuntimeError("Converted frame count does not match metadata")
    sources = summary.get("sources", [])
    if len(sources) != 30 or len({s["source"] for s in sources}) != 30:
        raise RuntimeError("Dataset manifest does not contain 30 unique sources")
    if any(Path(s["source"]).resolve().parent != RAW.resolve() for s in sources):
        raise RuntimeError("Dataset manifest contains an unexpected source path")
    print(
        f"VERIFIED_FIXED_V3_ONLY episodes={summary['episodes']} "
        f"frames={summary['frames']} transform=single_arm7_delta_finger2_absolute",
        flush=True,
    )


if __name__ == "__main__":
    verify_dataset()
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU training is required")
    train.main(get_config())
