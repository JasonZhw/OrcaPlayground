"""Resolve portable scene and planning assets into the ignored runtime directory."""

import json
from pathlib import Path
import sys

import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "simulation"))
from build_model import build
from task_config import ASSETS, MODELS


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ASSETS / "g1_scene_right_arm.yml").read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]
    kinematics["urdf_path"] = str(ASSETS / "g1_scene_right_arm.urdf")
    kinematics["asset_root_path"] = str(ASSETS)
    (MODELS / "g1_scene_right_arm.yml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    build()
    report = {
        "prepared": True, "scene": "V9 single blue target",
        "action_order": "right arm 7 absolute joints, inner and outer gripper masters",
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
