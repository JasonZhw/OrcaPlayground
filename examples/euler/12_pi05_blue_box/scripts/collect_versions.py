"""Export dependency versions without addresses, credentials or filesystem paths."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def collect():
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    report = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items(), key=lambda entry: entry[0].lower())),
        "scope": (
            "Versions of this reporting environment only. Original training "
            "launch settings and source modifications need separate evidence."
        ),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15, check=True,
        )
        report["gpu_query"] = {
            "fields": ["name", "memory_total_mib", "driver_version"],
            "rows": result.stdout.strip().splitlines(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        report["gpu_query"] = {"unavailable": type(error).__name__}
    try:
        import torch

        report["torch_runtime"] = {
            "version": str(torch.__version__),
            "compiled_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as error:
        report["torch_runtime"] = {"unavailable": type(error).__name__}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = collect()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print("Environment version report saved.")


if __name__ == "__main__":
    main()
