"""Check the installed runtime, package versions and portable scene references."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    sources = json.loads((ROOT / "configs/input_files.sha256.json").read_text(encoding="utf-8"))
    for name, expected in sources.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Original training input identity differs: {name}")
    for xml in (ROOT / "assets").glob("*.xml"):
        tree = ET.parse(xml)
        for asset in tree.findall("./asset/*"):
            if asset.get("file"):
                path = Path(asset.get("file"))
                if path.is_absolute() or not (xml.parent / path).is_file():
                    raise ValueError(f"Nonportable or missing scene asset: {xml.name}")
    report = {"source_hashes_verified": len(sources), "portable_scene_assets": True, "packages": {}}
    names = ["lerobot", "torch", "torchvision", "transformers", "numpy", "huggingface_hub"]
    if args.simulation:
        names.extend(["orca-gym", "orca-lab", "mujoco", "nvidia-curobo"])
        for command in ("ffmpeg", "ffprobe"):
            if not shutil.which(command):
                raise RuntimeError(f"Missing executable: {command}")
    for name in names:
        report["packages"][name] = importlib.metadata.version(name)
    if args.cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        report["gpu"] = torch.cuda.get_device_name()
        report["cuda"] = torch.version.cuda
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
