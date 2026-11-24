"""cuRobo planner for the scene-calibrated seven-joint G1 arm."""
import json

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, Pose, GoalToolPose

from task_config import ASSETS, MODELS


class ArmPlanner:
    def __init__(self):
        self.calibration = json.loads((ASSETS / "calibration.json").read_text())
        self.base_pos = np.array(self.calibration["base_position"])
        self.base_rot = np.array(self.calibration["base_rotation"])
        config = MotionPlannerCfg.create(
            robot=str(MODELS / "g1_scene_right_arm.yml"),
            scene_model=str(ASSETS / "planning_world_collection.yml"),
            num_ik_seeds=32, num_trajopt_seeds=4,
            interpolation_dt=0.02, use_cuda_graph=True,
            position_tolerance=0.003, orientation_tolerance=0.04,
        )
        self.planner = MotionPlanner(config)
        initial = JointState.from_position(
            torch.zeros((1, 7), device="cuda"), joint_names=self.planner.joint_names,
        )
        fk = self.planner.compute_kinematics(initial)
        pose = fk.tool_poses.to_dict()["grasp_center"]
        p = pose.position.detach().cpu().numpy().reshape(3)
        expected = np.array(self.calibration["ee_position_base"])
        error = float(np.linalg.norm(p - expected))
        print("CUROBO_FK_ERROR_METERS", error, flush=True)
        if error > 0.0005:
            raise RuntimeError(f"Scene/cuRobo calibration mismatch: {error}")

    def plan(self, joints, position_world, rotation_world):
        position = self.base_rot.T @ (np.array(position_world) - self.base_pos)
        rotation = self.base_rot.T @ np.array(rotation_world)
        xyzw = Rotation.from_matrix(rotation).as_quat()
        pose = Pose.from_list([*position.tolist(), *xyzw[[3, 0, 1, 2]].tolist()])
        goal = GoalToolPose.from_poses(
            {"grasp_center": pose}, ordered_tool_frames=self.planner.tool_frames,
        )
        state = JointState.from_position(
            torch.tensor(np.asarray(joints), device="cuda", dtype=torch.float32).reshape(1, 7),
            joint_names=self.planner.joint_names,
        )
        result = self.planner.plan_pose(
            current_state=state, goal_tool_poses=goal, max_attempts=4,
        )
        if result is None or not bool(result.success.flatten()[0]):
            raise RuntimeError("cuRobo trajectory failure: " + str(getattr(result,"status",None)) + " goal="+str(position_world))
        trajectory = result.interpolated_trajectory.position.detach().cpu().numpy().reshape(-1, 7)
        count = int(result.interpolated_last_tstep.flatten()[0].item())
        return trajectory[:count].copy()


if __name__ == "__main__":
    planner = ArmPlanner()
    rot = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    result = planner.plan(np.zeros(7), [30.1, -7.18246, 0.85], rot)
    print("TRAJECTORY", result.shape, result[-1].tolist())
