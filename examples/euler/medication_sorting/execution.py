"""Execute planned arm motions and latch gripper commands while recording each step."""

import asyncio
import json
import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from scipy.spatial.transform import Rotation

from .planning import plan_motion, smooth_blend, time_samples
from .recording import AD_KNOTS, CLOSURE_KNOTS


def scalar_to_ad(closure):
    """Decode a normalized closure command; reject invalid values instead of clipping."""
    value = np.asarray(closure, dtype=float)
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("Gripper closure must be finite and in [0, 1]")
    return np.stack([np.interp(value, CLOSURE_KNOTS, AD_KNOTS[:, i]) for i in range(2)], axis=-1)


class Grippers:
    def __init__(self, joint_names):
        self.indices = {}
        self.commands = {}
        self.size = len(joint_names)
        self.closure = {}
        for arm, side in [("left", "l"), ("right", "r")]:
            ids = []
            for part in ["inner", "outer"]:
                for joint in [1, 3, 4, 2]:
                    suffix = f"_gripper_{side}_{part}_joint{joint}"
                    matches = [i for i, n in enumerate(joint_names) if n.endswith(suffix)]
                    if len(matches) != 1:
                        raise ValueError(f"Expected one servo for {suffix}")
                    ids.append(matches[0])
            self.indices[arm] = np.array(ids)
            self.set(arm, 0)

    def set(self, arm, closure):
        """Latch one normalized command; linkage targets are an internal detail."""
        if arm not in self.indices or np.ndim(closure) != 0:
            raise ValueError("Expected a known arm and one closure value")
        a, drive = scalar_to_ad(closure)
        self.closure[arm] = float(closure)
        self.commands[arm] = np.array([0, 0, a, drive, 0, 0, -a, drive], dtype=float)

    def apply(self, requested):
        out = np.asarray(requested, dtype=float).copy()
        if out.shape != (self.size,) or not np.isfinite(out).all():
            raise ValueError("Invalid servo vector")
        for arm, ids in self.indices.items():
            out[ids] = self.commands[arm]
        return out


class SortingEnv(OrcaGymEulerEnv):
    def __init__(self, xml_path, frame_skip=10, timestep=0.002, endpoint="127.0.0.1:50051"):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=endpoint,
            agent_names=["unitree_g1_1"],
            time_step=timestep,
            model_xml_path=str(xml_path),
            skip_grpc_load=True,
            render_mode="none",
            device="cpu",
        )
        self.mj_forward()
        self._sync_view()
        items = [self.model.get_actuator_byid(i) for i in range(self.model.nu)]
        self.joint_names = [item["JointName"] for item in items]
        self.qids = np.array([self.jnt_qposadr(n) for n in self.joint_names])
        self.vids = np.array([self.jnt_dofadr(n) for n in self.joint_names])
        self.gains = np.array([item["GainPrm"][0] for item in items])
        self.limits = np.array([item["CtrlRange"] if item["CtrlLimited"] else [-np.inf, np.inf] for item in items])
        for item in items:
            if not (
                item["TrnType"] == 0
                and item["DynType"] == 0
                and item["GainType"] == 0
                and item["BiasType"] == 1
                and item["GainPrm"][0] > 0
                and np.isclose(item["BiasPrm"][1], -item["GainPrm"][0])
            ):
                raise ValueError("Expected original position servos")
        self.arm_mask = np.array(["_gripper_" not in n for n in self.joint_names])
        self.grippers = Grippers(self.joint_names)
        for arm in ["left", "right"]:
            self.grippers.set(arm, 0.4 / 1.97)  # Same neutral linkage target as a=0,d=0.
        self.command = self.data.qpos.ravel()[self.qids].copy()
        self.last_control = self.command.copy()
        self.fix_base()

    def fix_base(self):
        anchors = [
            n
            for n in self.model.get_body_names()
            if n == "ActorManipulator_Anchor" or (n.startswith("ORCA_MANIPULATOR_") and n.endswith("_Anchor"))
        ]
        if len(anchors) != 1:
            raise ValueError("Expected one manipulator anchor")
        anchor = anchors[0]
        slot = self.equality_find_slot_by_body(anchor)
        if slot < 0:
            raise ValueError("No anchor weld available")
        eq = self.equality_constraint(slot)
        other = eq["obj2_id"] if eq["obj1_id"] == self.model.body_name2id(anchor) else eq["obj1_id"]
        if eq["type"] != 1 or not self.model.body_id2name(other).lower().endswith("_dummy"):
            raise ValueError("Anchor already in use")
        pose = self.get_body_xpos_xmat_xquat(["unitree_g1_1_pelvis"])["unitree_g1_1_pelvis"]
        self.set_mocap_pos_and_quat({anchor: {"pos": pose["xpos"].copy(), "quat": pose["xquat"].copy()}})
        weld = np.zeros(11)
        weld[6] = 1
        weld[10] = 1
        self.equality_update(slot, eq_type=1, obj1_name=anchor, obj2_name="unitree_g1_1_pelvis", data=weld, active=True)
        self.mj_forward()
        self._sync_view()

    def step(self, command):
        self.command = self.grippers.apply(command)
        if np.any(self.command < self.limits[:, 0]) or np.any(self.command > self.limits[:, 1]):
            raise ValueError("Servo target outside control range")
        for _ in range(self.frame_skip):
            # Feed-forward bias compensates the robot, not the gripper mechanisms.
            control = self.command.copy()
            control[self.arm_mask] += 11.0 * (
                self.command[self.arm_mask] - self.data.qpos.ravel()[self.qids][self.arm_mask]
            )
            control[self.arm_mask] += self.data.qfrc_bias.ravel()[self.vids][self.arm_mask] / self.gains[self.arm_mask]
            self.last_control = np.clip(control, self.limits[:, 0], self.limits[:, 1])
            self.do_simulation(self.last_control, 1)
        obs = self.data.qpos.ravel().copy()
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite simulation state")
        return obs, 0.0, False, False, {}

    def reset_model(self):
        raise RuntimeError("Create a new run from the current exported layout to reset")


class Executor:
    def __init__(self, env, scene, bridge, loop, out, record=False, task_index=0, object_key="all"):
        self.env, self.scene, self.bridge, self.loop, self.out = env, scene, bridge, loop, out
        self.index = 0
        self.trace = []
        self.plans = []
        self.rec = None
        self.home = env.command.copy()
        self.started = time.monotonic()
        self.record = record
        self.task_index = task_index
        self.object_key = object_key
        self.carried_guard = None

    def sync(self):
        self.scene.set_qpos(self.env.data.qpos.ravel().copy())

    def tick(self, command, stage):
        tick_started = time.monotonic()
        e, s = self.env, self.scene
        command = e.grippers.apply(command)
        q = e.data.qpos.ravel().copy()
        state = []
        action = []
        for arm in ["left", "right"]:
            a = s.arms[arm]
            g = e.grippers.indices[arm]
            state.extend(q[a.qpos_ids])
            state.extend(q[e.qids[g]])
            ids = [e.joint_names.index(n) for n in a.joint_names]
            action.extend(command[ids])
            action.append(e.grippers.closure[arm])
        if self.rec or self.index % 2 == 0:
            self.loop.run_until_complete(
                asyncio.wait_for(
                    self.bridge.render(
                        q, float(e.data.time), simulate_index=self.index, request_idr=self.rec is not None
                    ),
                    15,
                )
            )
        if self.rec:
            self.rec.append(state, action, float(e.data.time), self.index, stage)
        e.step(command)
        self.sync()
        self.trace.append(
            {
                "index": self.index,
                "time": float(e.data.time),
                "stage": stage,
                "state": state,
                "action": action,
                "qpos_after": e.data.qpos.ravel().tolist(),
                "objects": {k: s.body_pose(v)[0].tolist() for k, v in s.bodies.items()},
            }
        )
        self.index += 1
        if self.carried_guard is not None:
            arm, key, local = self.carried_guard
            hp, hr = s.end_pose(arm)
            op, _ = s.body_pose(key)
            drift = float(np.linalg.norm(hr.T @ (op - hp) - local))
            if drift > 0.04:
                raise RuntimeError(f"{key}: lost grasp during {stage} (relative drift {drift:.3f} m)")
        if self.rec is None:
            # Cap live trials at simulation speed. Never catch up in a burst
            # after a slow render or planning pause; recording keeps exact indices.
            time.sleep(max(0.0, 0.02 - (time.monotonic() - tick_started)))

    def hold(self, n, stage):
        for _ in range(n):
            self.tick(self.env.command, stage)

    def align_initial_command(self):
        """Ease out the initial servo preload before planning or recording starts."""
        for _ in range(3):
            start = self.env.command.copy()
            goal = start.copy()
            for binding in self.scene.arms.values():
                ids = [self.env.joint_names.index(name) for name in binding.joint_names]
                goal[ids] = self.env.data.qpos.ravel()[binding.qpos_ids]
            for command in time_samples([start, goal], max_velocity=0.2, max_acceleration=0.4):
                self.tick(command, "initial_command_alignment")
            self.hold(25, "initial_command_settle")
            q = self.env.data.qpos.ravel().copy()
            for binding in self.scene.arms.values():
                ids = [self.env.joint_names.index(name) for name in binding.joint_names]
                q[binding.qpos_ids] = self.env.command[ids]
            if all(self.scene.collision_free(q, arm, penetration_tolerance=0.001) for arm in self.scene.arms):
                return
        raise RuntimeError("Initial commanded arm pose remains in collision; reset the robot away from the bins")

    def ramp_gripper(self, arm, closure, seconds, stage):
        """Keep the arm still while easing this hand's command to its new value."""
        if not np.isfinite([closure, seconds]).all() or not 0 <= closure <= 1 or seconds <= 0:
            raise ValueError("Gripper target must be in [0, 1] and duration positive")
        start = self.env.grippers.closure[arm]
        for blend in smooth_blend(np.linspace(0, 1, int(np.ceil(seconds / 0.02)) + 1)):
            self.env.grippers.set(arm, start + (closure - start) * float(blend))
            self.tick(self.env.command, stage)

    def move(
        self,
        arm,
        position,
        rotation,
        stage,
        target=None,
        carried=None,
        allow=False,
        max_velocity=0.4,
        max_acceleration=0.8,
    ):
        self.sync()
        s, e = self.scene, self.env
        a = s.arms[arm]
        q = e.data.qpos.ravel().copy()
        ids = [e.joint_names.index(n) for n in a.joint_names]
        # Preserve the last commanded position at phase boundaries. Starting at
        # measured qpos would turn tracking error into a sudden command jump.
        q[a.qpos_ids] = e.command[ids]

        path = plan_motion(s, arm, position, rotation, q, target=target, carried=carried, allow=allow, seed=self.index)
        self.plans.append(
            {
                "stage": stage,
                "arm": arm,
                "position": list(position),
                "rotation": np.asarray(rotation).tolist(),
                "path": path.tolist(),
                "max_velocity": max_velocity,
                "max_acceleration": max_acceleration,
            }
        )
        (self.out / "plans.json").write_text(json.dumps(self.plans, indent=2))
        print("MOVE", stage, arm, "waypoints", len(path), "goal", np.round(position, 4).tolist(), flush=True)
        for part in time_samples(path, dt=0.02, max_velocity=max_velocity, max_acceleration=max_acceleration):
            command = e.command.copy()
            command[ids] = part
            self.tick(command, stage)
        self.hold(30, stage + "_settle")
        self.sync()
        p, r = s.end_pose(arm)
        error = float(np.linalg.norm(p - position))
        angle = float(np.linalg.norm(Rotation.from_matrix(np.asarray(rotation) @ r.T).as_rotvec()))
        print("REACHED", stage, "error", round(error, 4), "angle", round(angle, 3), flush=True)
        if error > 0.012 or angle > 0.12:
            raise RuntimeError(f"{stage}: actual end pose missed target ({error:.4f} m, {angle:.3f} rad)")
