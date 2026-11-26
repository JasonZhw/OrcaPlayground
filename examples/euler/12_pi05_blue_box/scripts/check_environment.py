"""Check the installed runtime, package versions and portable scene references."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def requirements(path):
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            result.extend(requirements(path.parent / line[3:].strip()))
        else:
            item = Requirement(line)
            if item.marker is None or item.marker.evaluate():
                result.append(item)
    return result


def check_versions(profile, version=importlib.metadata.version):
    files = (["requirements-training.txt"] if profile == "training" else
             ["requirements.txt", "requirements-simulation.txt"])
    installed, issues = {}, []
    for item in (item for name in files for item in requirements(ROOT / name)):
        try:
            actual = version(item.name)
        except importlib.metadata.PackageNotFoundError:
            issues.append({"package": item.name, "required": str(item.specifier),
                           "installed": None})
            continue
        installed[item.name] = actual
        if item.specifier and not item.specifier.contains(actual, prereleases=True):
            issues.append({"package": item.name, "required": str(item.specifier),
                           "installed": actual})
    return installed, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--profile", choices=("training", "simulation"))
    parser.add_argument("--strict-versions", action="store_true",
                        help="Fail on missing packages or version mismatches in the selected profile")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    profile = args.profile or ("simulation" if args.simulation else "training")
    if args.simulation and profile != "simulation":
        parser.error("--simulation conflicts with --profile training")
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
    installed, issues = check_versions(profile)
    runtime_issues = []
    dependencies = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120,
    )
    if dependencies.returncode:
        runtime_issues.append("Installed package dependencies are incompatible")
    report = {"source_hashes_verified": len(sources), "portable_scene_assets": True,
              "profile": profile, "python": platform.python_version(),
              "packages": installed, "version_issues": issues,
              "runtime_issues": runtime_issues,
              "pip_check_passed": dependencies.returncode == 0,
              "pip_check_output": (dependencies.stdout + dependencies.stderr).strip()}
    if platform.python_version_tuple()[:2] != ("3", "12"):
        issues.append({"package": "Python", "required": "3.12.x",
                       "installed": platform.python_version()})
    if profile == "simulation":
        for command in ("ffmpeg", "ffprobe"):
            if not shutil.which(command):
                runtime_issues.append(f"Missing executable: {command}")
        try:
            report["packages"]["nvidia-curobo"] = importlib.metadata.version("nvidia-curobo")
        except importlib.metadata.PackageNotFoundError:
            issues.append({"package": "nvidia-curobo", "required": "pinned source installation",
                           "installed": None})
        report["curobo_revision_check"] = "not_verified; retain the installer log"
    if args.cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                runtime_issues.append("CUDA is unavailable")
            else:
                report["gpu"] = torch.cuda.get_device_name()
                report["cuda"] = torch.version.cuda
                report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        except (ImportError, OSError, RuntimeError) as error:
            runtime_issues.append(f"CUDA runtime check failed: {type(error).__name__}")
    report["strict_versions_passed"] = not issues
    print(json.dumps(report, indent=2))
    if runtime_issues or (issues and (
            args.strict_versions or any(row["installed"] is None for row in issues))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
