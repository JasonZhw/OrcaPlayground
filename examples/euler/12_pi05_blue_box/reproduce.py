"""Run the V9 data, collection, training and evaluation stages explicitly."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "check", "download", "prepare-scene", "collect", "convert",
        "prepare-train", "train", "infer", "evaluate", "studio",
    ))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entries = {
        "check": "scripts/check_environment.py",
        "download": "scripts/download_assets.py",
        "prepare-scene": "scripts/prepare_scene.py",
        "collect": "simulation/collect_200.py",
        "convert": "scripts/convert_raw.py",
        "prepare-train": "scripts/prepare_train.py",
        "train": "scripts/train.py",
        "infer": "scripts/infer.py",
        "evaluate": "simulation/evaluate.py",
        "studio": "scripts/start_studio.py",
    }
    run(entries[args.stage], *args.arguments)


if __name__ == "__main__":
    main()
