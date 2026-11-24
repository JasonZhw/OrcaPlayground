"""Reduced physics with name-based controls and full-model Studio rendering."""
import re

import mujoco
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

from task_config import ADDRESS, ARM_SUFFIXES, ROBOT, TARGET
from build_model import OUTPUT, QposBridge, reference


class DualArmEnv(OrcaGymEulerEnv):
    def initialize_simulation(self):
        full, data = reference()
        reduced = mujoco.MjModel.from_xml_path(str(OUTPUT))
        self.bridge = QposBridge(full, data.qpos, reduced)
        if not self.offline:
            studio = self.studio_bridge()
            original_load = studio.load_model_xml
            async def load_model():
                actual_path = await original_load()
                actual = mujoco.MjModel.from_xml_path(actual_path)
                expected = [full.joint(i).name for i in range(full.njnt)]
                found = [actual.joint(i).name for i in range(actual.njnt)]
                def body_key(model, joint_id):
                    name = model.body(int(model.jnt_bodyid[joint_id])).name
                    return re.sub(r"^\[\d+\]_", "", name)
                aliases = {}
                template = actual.qpos0.copy()
                for jid, name in enumerate(expected):
                    if name in found:
                        destination = actual.joint(name).id
                    else:
                        matches = [i for i in range(actual.njnt)
                                   if body_key(actual, i) == body_key(full, jid)]
                        if name.startswith(ROBOT + "_") or len(matches) != 1:
                            raise ValueError(f"Unmatched scene joint: {name}")
                        destination = matches[0]
                    other = actual.joint(destination)
                    if int(other.type[0]) != int(full.jnt_type[jid]):
                        raise ValueError("Scene joint types changed")
                    aliases[name] = other.name
                    width = {0: 7, 1: 4, 2: 1, 3: 1}[int(other.type[0])]
                    src, dst = int(full.jnt_qposadr[jid]), int(other.qposadr[0])
                    template[dst:dst + width] = data.qpos[src:src + width]
                if len(set(aliases.values())) != actual.njnt:
                    raise ValueError("Scene joint mapping is not one-to-one")
                actual_data = mujoco.MjData(actual)
                actual_data.qpos[:] = template
                mujoco.mj_forward(actual, actual_data)
                for name in (f"{ROBOT}_pelvis", TARGET):
                    a, b = full.body(name).id, actual.body(name).id
                    if np.linalg.norm(data.xpos[a] - actual_data.xpos[b]) > 1e-5:
                        raise ValueError(f"Scene pose changed: {name}")
                for name in (f"{ROBOT}_ee_center_site_r", f"{ROBOT}_ee_center_site_l"):
                    a, b = full.site(name).id, actual.site(name).id
                    if (np.linalg.norm(data.site_xpos[a] - actual_data.site_xpos[b]) > 1e-5
                            or np.max(np.abs(data.site_xmat[a] - actual_data.site_xmat[b])) > 1e-5):
                        raise ValueError(f"Scene end-effector calibration changed: {name}")
                self.bridge = QposBridge(actual, template, reduced, aliases)
                return str(OUTPUT)
            studio.load_model_xml = load_model
            original_render = studio.render
            async def render(qpos, sim_time, **kwargs):
                return await original_render(self.bridge.pad(qpos), sim_time, **kwargs)
            studio.render = render
        return super().initialize_simulation()

    def __init__(self, offline=False):
        self.offline = offline
        super().__init__(
            frame_skip=20, orcagym_addr=ADDRESS, agent_names=[ROBOT],
            time_step=0.001, model_xml_path=str(OUTPUT), skip_grpc_load=offline,
            render_mode="none" if offline else "human",
        )
        self.joints = self.model.get_joint_dict()
        self.actuators = self.model.get_actuator_dict()
        by_joint = {a["JointName"]: a["ActuatorId"] for a in self.actuators.values()}
        self.arm_names = [f"{ROBOT}_{s}" for s in ARM_SUFFIXES]
        self.arm_qids = [self.joints[n]["QposIdxStart"] for n in self.arm_names]
        self.arm_aids = [by_joint[n] for n in self.arm_names]
        def finger(index, side, link):
            name = f"{ROBOT}_idx{index}_gripper_r_{side}_joint{link}"
            return by_joint[name]
        self.finger_aids = [finger(71, "inner", 1), finger(81, "outer", 1)]
        self.finger_chains = [
            (finger(71, "inner", 1), finger(72, "inner", 3), finger(73, "inner", 4), 1),
            (finger(81, "outer", 1), finger(82, "outer", 3), finger(83, "outer", 4), -1),
        ]
        self.action_ids = self.arm_aids + self.finger_aids
        self.qids = np.zeros(self.model.nu, dtype=int)
        self.vids = np.zeros(self.model.nu, dtype=int)
        self.gain = np.zeros(self.model.nu)
        self.ctrl_limits = np.zeros((self.model.nu, 2))
        for item in self.actuators.values():
            aid = item["ActuatorId"]
            joint = self.joints[item["JointName"]]
            self.qids[aid] = joint["QposIdxStart"]
            self.vids[aid] = joint["QvelIdxStart"]
            self.gain[aid] = item["GainPrm"][0]
            self.ctrl_limits[aid] = item["CtrlRange"]
        pose = self.data.qpos.copy()
        for suffix, value in {'left_shoulder_pitch_joint': -0.3, 'left_elbow_joint': 1.5, 'left_shoulder_roll_joint': 0.3}.items():
            pose[self.joints[f'{ROBOT}_{suffix}']['QposIdxStart']] = value
        self.set_joint_qpos(pose)
        self.mj_forward()
        self.start_qpos = self.data.qpos.copy()
        self.setpoint = self.start_qpos[self.qids].copy()
        self.fixed_pelvis = self.data.body_xpos(f"{ROBOT}_pelvis").copy()
        self.left_qids = [self.joints[f"{ROBOT}_{s.replace('right_', 'left_')}"]["QposIdxStart"]
                          for s in ARM_SUFFIXES]
        self.frames = 0

    def reset_model(self):
        self.set_joint_qpos(self.start_qpos.copy())
        self.set_joint_qvel(np.zeros(self.model.nv))
        self.mj_forward()
        self.setpoint = self.start_qpos[self.qids].copy()
        return self.observe(), {}

    def observe(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                "target": self.data.body_xpos(TARGET).copy(),
                "ee": self.data.site_xpos(f"{ROBOT}_ee_center_site_r").copy(),
                "pelvis": self.data.body_xpos(f"{ROBOT}_pelvis").copy()}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError("Invalid actuator target")
        # Only the right arm and its gripper masters are commanded by the policy.
        command = self.setpoint.copy()
        command[self.action_ids] = action[self.action_ids]
        for _ in range(self.frame_skip):
            gravity = np.divide(self.data.qfrc_bias[self.vids], self.gain,
                                out=np.zeros_like(self.gain), where=self.gain > 0)
            ctrl = command + gravity
            for master, middle, distal, sign in self.finger_chains:
                ctrl[distal] = (sign * self.data.qpos[self.qids[master]]
                               - self.data.qpos[self.qids[middle]] + gravity[distal])
            self.do_simulation(np.clip(ctrl, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1]), 1)
        self.frames += 1
        obs = self.observe()
        if not np.isfinite(obs["qpos"]).all():
            raise RuntimeError("Non-finite physics state")
        if np.linalg.norm(obs["pelvis"] - self.fixed_pelvis) > 1e-8:
            raise RuntimeError("Fixed pelvis moved")
        return obs, 0.0, False, False, {"time": float(self.data.time)}

    def restore(self):
        self.reset_simulation()
        self.reset_model()
        self.render()
