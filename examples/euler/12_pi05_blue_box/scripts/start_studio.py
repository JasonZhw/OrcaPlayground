"""Start OrcaLab with the V9 workspace or load the layout into a running instance."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def call(command, payload):
    executable = shutil.which("orcalab-cli")
    if not executable:
        raise RuntimeError("Install the pinned orca-lab package")
    result = subprocess.run(
        [executable, "--workspace", str(ROOT), command, "--json", "-"], input=json.dumps(payload),
        capture_output=True, text=True, timeout=120, check=True,
    )
    response = json.loads(result.stdout)
    if response.get("isError"):
        raise RuntimeError(result.stdout)
    for entry in response.get("content", []):
        if entry.get("type") == "text":
            try:
                message = json.loads(entry["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("success") is False:
                raise RuntimeError(entry["text"])
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true",
                        help="Load V9 in an already running OrcaLab and start external simulation")
    args = parser.parse_args()
    if args.load:
        call("load_layout", {"layout_path": str(ROOT / "assets/scene_v9.json")})
        call("start_simulation", {"program_name": "external"})
    else:
        executable = shutil.which("orcalab")
        if not executable:
            raise RuntimeError("Install the pinned orca-lab package")
        subprocess.run([executable, str(ROOT), "--layout", str(ROOT / "assets/scene_v9.json")], check=True)


if __name__ == "__main__":
    main()
