"""Paths and action ordering for the V9 fixed-torso task."""

import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "assets"
ROOT = Path(os.environ.get("PI05_V9_WORK", PROJECT / "runtime")).resolve()
MODELS = ROOT / "models"
ROBOT = "unitree_g1_1"
TARGET = "intel_core_i5_processor_package_1_3c_intelcore"
ADDRESS = os.environ.get("ORCAGYM_ADDR", "localhost:50051")
ARM_SUFFIXES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ACTION_JOINTS = [f"{ROBOT}_{name}" for name in ARM_SUFFIXES] + [
    f"{ROBOT}_idx71_gripper_r_inner_joint1",
    f"{ROBOT}_idx81_gripper_r_outer_joint1",
]
