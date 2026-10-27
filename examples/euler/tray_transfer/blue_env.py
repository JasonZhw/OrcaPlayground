"""CPU Euler adapter for the blue-part pick test.

Offline construction never connects to OrcaLab. Online construction uses the
current Studio model and the upstream Euler lifecycle (pause, load, reset).
Only :meth:`step` advances physics; configuration edits are offline-only.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

from .gripper_control import RightGripperHold


def compute_servo_control(
    desired: np.ndarray,
    actual: np.ndarray,
    bias: np.ndarray,
    gains: np.ndarray,
    control_limits: np.ndarray,
    compensation_mask: np.ndarray,
    *,
    bias_compensation: bool = False,
    position_feedback_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded servo targets and a per-actuator clipping mask.

    Inputs use actuator order. ``bias`` contains the corresponding generalized
    force components, not joint positions. The mask excludes the gripper's
    closed-chain joints; its existing servos are left unchanged.
    """
    desired, actual, bias, gains = (
        np.asarray(value, dtype=float) for value in (desired, actual, bias, gains)
    )
    mask = np.asarray(compensation_mask, dtype=bool)
    limits = np.asarray(control_limits, dtype=float)
    if desired.ndim != 1 or any(
        value.shape != desired.shape for value in (actual, bias, gains, mask)
    ) or limits.shape != (desired.size, 2):
        raise ValueError("Servo arrays must share actuator order and dimension")
    if not all(np.isfinite(value).all() for value in (desired, actual, bias, gains)):
        raise ValueError("Servo state and gains must be finite")
    if np.any(gains[mask] <= 0) or not np.isfinite(position_feedback_gain) or position_feedback_gain <= 0:
        raise ValueError("Position gains must be finite and positive")
    if np.isnan(limits).any() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Invalid actuator control limits")
    control = desired.copy()
    if bias_compensation:
        control[mask] += bias[mask] / gains[mask]
    control[mask] += (position_feedback_gain - 1.0) * (desired[mask] - actual[mask])
    bounded = np.clip(control, limits[:, 0], limits[:, 1])
    return bounded, bounded != control


@contextmanager
def _offline_xml(path: str):
    """Resolve assets in an exported standalone XML without changing physics."""
    source = Path(path).resolve(strict=True)
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    for kind, directory in (("mesh", "meshdir"), ("texture", "texturedir")):
        asset_dir = source.parent
        if compiler is not None and compiler.get(directory):
            asset_dir /= compiler.get(directory)
        for node in root.findall(f"./asset/{kind}[@file]"):
            asset = Path(node.get("file"))
            candidates = (asset_dir / asset, source.parent / "assets" / asset)
            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"Missing exported {kind} asset: {asset}")
            node.set("file", str(resolved))
    # This adapter consumes standalone exports, not a repository XML include tree.
    if root.findall(".//include"):
        raise ValueError("Use a standalone exported XML without include elements")
    with tempfile.TemporaryDirectory(prefix="tray-blue-model-") as temporary:
        target = Path(temporary) / "scene.xml"
        tree.write(target, encoding="utf-8")
        yield str(target)


class BluePickEnv(OrcaGymEulerEnv):
    """Position-servo control for the scene's G1 and four-bar grippers.

    ``online=True`` is an explicit simulator takeover: the base class pauses
    Studio and initializes its local simulation from Studio's exported model.
    It does not preserve an arbitrary running simulation snapshot. Exported
    ``model_xml_path`` is used only offline; online always uses the live model.
    """

    agent_name = "unitree_g1_1"
    pelvis_name = "unitree_g1_1_pelvis"

    def __init__(
        self,
        model_xml_path: str | None = None,
        endpoint: str = "127.0.0.1:50051",
        online: bool = False,
        timestep: float = 0.002,
        frame_skip: int = 10,
        bias_compensation: bool = False,
        position_feedback_gain: float = 1.0,
    ) -> None:
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("timestep must be finite and positive")
        if isinstance(frame_skip, bool) or not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        if not online and model_xml_path is None:
            raise ValueError("Offline mode requires model_xml_path")
        if not isinstance(bias_compensation, bool):
            raise ValueError("bias_compensation must be a boolean")
        if not np.isfinite(position_feedback_gain) or position_feedback_gain <= 0:
            raise ValueError("position_feedback_gain must be finite and positive")
        self.online = online
        self.bias_compensation = bias_compensation
        self.position_feedback_gain = float(position_feedback_gain)
        self.base_anchor_slot: int | None = None
        self.base_anchor_original: dict | None = None
        self.held_base_pose: dict[str, np.ndarray] | None = None
        options = dict(
            frame_skip=frame_skip,
            orcagym_addr=endpoint,
            agent_names=[self.agent_name],
            time_step=timestep,
            skip_grpc_load=not online,
            render_mode="human" if online else "none",
            device="cpu",
        )
        if online:
            super().__init__(**options)
        else:
            with _offline_xml(model_xml_path) as resolved_xml:
                super().__init__(model_xml_path=resolved_xml, **options)
        # The upstream constructor resets data without a forward pass.
        self.mj_forward()
        self._sync_view()
        self.joint_qpos_indices = {
            name: self.jnt_qposadr(name) for name in self.model.get_joint_dict()
        }
        self.robot_joint_names = tuple(
            name for name in self.joint_qpos_indices if name.startswith(self.agent_name + "_")
        )
        self.actuator_indices = {
            name: self.model.actuator_name2id(name)
            for name in self.model.get_actuator_dict()
        }
        ordered = [self.model.get_actuator_byid(i) for i in range(self.model.nu)]
        self.actuator_joint_names = tuple(item["JointName"] for item in ordered)
        self.right_gripper_hold = RightGripperHold(self.actuator_joint_names)
        self.actuator_qpos_indices = np.array([
            self.joint_qpos_indices[name] for name in self.actuator_joint_names
        ])
        self.actuator_dof_indices = np.array([
            self.jnt_dofadr(name) for name in self.actuator_joint_names
        ])
        if self.model.nu != 45:
            raise ValueError(f"Expected the calibrated 45-servo G1, got nu={self.model.nu}")
        for name, item in self.model.get_actuator_dict().items():
            if not (
                int(item["TrnType"]) == 0
                and int(item["DynType"]) == 0
                and int(item["GainType"]) == 0
                and int(item["BiasType"]) == 1
                and float(item["GainPrm"][0]) > 0
                and np.isclose(item["BiasPrm"][0], 0)
                and np.isclose(item["BiasPrm"][1], -item["GainPrm"][0])
                and np.allclose(item["Gear"], [1, 0, 0, 0, 0, 0])
            ):
                raise ValueError(f"Unsupported position-servo configuration: {name}")
        self.control_limits = np.array([
            item["CtrlRange"] if item["CtrlLimited"] else [-np.inf, np.inf]
            for item in ordered
        ], dtype=float)
        self.servo_gains = np.array([item["GainPrm"][0] for item in ordered], dtype=float)
        self.compensation_mask = np.array([
            name.startswith(self.agent_name + "_") and "_gripper_" not in name
            for name in self.actuator_joint_names
        ])
        if np.count_nonzero(self.compensation_mask) != 29:
            raise ValueError("Expected 29 robot servos outside the gripper mechanisms")
        self.force_limits = np.array([item["ForceRange"] for item in ordered], dtype=float)
        self.force_limited = np.array([item["ForceLimited"] for item in ordered], dtype=bool)
        self.last_command: np.ndarray | None = None
        self.last_servo_control: np.ndarray | None = None
        # Flags accumulate over one control step, including every physics substep.
        self.last_control_saturated = np.zeros(self.model.nu, dtype=bool)
        self.last_force_saturated = np.zeros(self.model.nu, dtype=bool)
        self.action_space = spaces.Box(
            low=self.control_limits[:, 0], high=self.control_limits[:, 1], dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float64
        )

    def read_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """Return world position and quaternion in MuJoCo's wxyz order."""
        pose = self.get_body_xpos_xmat_xquat([body])[body]
        return np.array(pose["xpos"], copy=True), np.array(pose["xquat"], copy=True)

    def get_state(self) -> dict:
        """Copy state and current contacts; this does not step the simulation."""
        contacts = [
            {**contact, "pos": contact["pos"].copy(), "frame": contact["frame"].copy()}
            for contact in self.query_contact_simple()
        ]
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.ravel().copy(),
            "qvel": self.data.qvel.ravel().copy(),
            "contacts": contacts,
            "command": None if self.last_command is None else self.last_command.copy(),
            "servo_control": None if self.last_servo_control is None else self.last_servo_control.copy(),
            "control_saturated": self.last_control_saturated.copy(),
            "force_saturated": self.last_force_saturated.copy(),
        }

    def control_from_qpos(self, qpos: np.ndarray | None = None) -> np.ndarray:
        """Map a full configuration to absolute joint targets in actuator order."""
        q = self.data.qpos.ravel() if qpos is None else np.asarray(qpos, dtype=float)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError(f"Expected finite qpos of shape {(self.model.nq,)}")
        return q[self.actuator_qpos_indices].copy()

    def set_configuration(self, qpos_vector: np.ndarray) -> None:
        """Offline kinematic query only; online motion must use position servos."""
        if self.online:
            raise RuntimeError("set_configuration is offline-only; use step for motion")
        q = np.asarray(qpos_vector, dtype=float)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError(f"Expected finite qpos of shape {(self.model.nq,)}")
        self.set_joint_qpos(q)
        self.set_joint_qvel(np.zeros(self.model.nv))
        self.mj_forward()
        self._sync_view()

    def fix_base(self) -> dict[str, np.ndarray]:
        """Weld only the pelvis to a stationary mocap at its current world pose."""
        if self.base_anchor_slot is not None:
            return {key: value.copy() for key, value in self.held_base_pose.items()}
        anchors = [
            name for name in self.model.get_body_names()
            if name == "ActorManipulator_Anchor"
            or (name.startswith("ORCA_MANIPULATOR_") and name.endswith("_Anchor"))
        ]
        if len(anchors) != 1:
            raise ValueError(f"Expected one unused manipulator mocap, found {anchors}")
        anchor = anchors[0]
        slot = self.equality_find_slot_by_body(anchor)
        if slot < 0:
            raise ValueError(f"No equality slot for {anchor}")
        original = self.equality_constraint(slot)
        anchor_id = self.model.body_name2id(anchor)
        other_id = original["obj2_id"] if original["obj1_id"] == anchor_id else original["obj1_id"]
        other = self.model.body_id2name(other_id)
        if original["type"] != int(mujoco.mjtEq.mjEQ_WELD) or not other.lower().endswith("_dummy"):
            raise ValueError(f"Manipulator slot is not an unused dummy weld: {other}")
        pos, quat = self.read_pose(self.pelvis_name)
        if not np.isfinite(pos).all() or not np.isclose(np.linalg.norm(quat), 1):
            raise ValueError("Invalid pelvis pose; forward kinematics must be initialized")
        self.set_mocap_pos_and_quat({anchor: {"pos": pos, "quat": quat}})
        # Both body frames now coincide: zero relative position, identity rotation.
        weld = np.zeros(11)
        weld[6] = 1.0
        weld[10] = 1.0
        self.equality_update(
            slot, eq_type=int(mujoco.mjtEq.mjEQ_WELD),
            obj1_name=anchor, obj2_name=self.pelvis_name, data=weld, active=True,
        )
        self.base_anchor_slot = slot
        self.base_anchor_original = original
        self.held_base_pose = {"pos": pos.copy(), "quat": quat.copy()}
        return {key: value.copy() for key, value in self.held_base_pose.items()}

    def step(self, action: np.ndarray):
        """Track nominal targets, optionally compensating the 29 robot servos.

        An active ``right_gripper_hold`` overrides the eight requested gripper
        inputs until explicitly released; other joints continue to track action.
        ``last_command`` is the action after this override. ``last_servo_control`` is
        the final physics substep's bounded, compensated servo input. Bias
        compensation covers robot gravity and velocity terms, not a free
        object's contact-transmitted payload. Actuator force limits are never
        modified and remain enforced by the original model.
        """
        desired = self.right_gripper_hold.apply(action)
        if desired.shape != (self.model.nu,) or not np.isfinite(desired).all():
            raise ValueError(f"Expected finite action of shape {(self.model.nu,)}")
        if np.any(desired < self.control_limits[:, 0] - 1e-8) or np.any(desired > self.control_limits[:, 1] + 1e-8):
            raise ValueError("Position target exceeds actuator control limits")
        self.last_command = desired.copy()
        self.last_control_saturated.fill(False)
        self.last_force_saturated.fill(False)
        for _ in range(self.frame_skip):
            control, saturated = compute_servo_control(
                self.last_command,
                self.data.qpos.ravel()[self.actuator_qpos_indices],
                self.data.qfrc_bias.ravel()[self.actuator_dof_indices],
                self.servo_gains,
                self.control_limits,
                self.compensation_mask,
                bias_compensation=self.bias_compensation,
                position_feedback_gain=self.position_feedback_gain,
            )
            self.last_servo_control = control.copy()
            self.last_control_saturated |= saturated
            self.do_simulation(control, 1)
            forces = self.data.actuator_force.ravel()
            tolerance = 1e-6 + 1e-5 * np.max(np.abs(self.force_limits), axis=1)
            self.last_force_saturated |= self.force_limited & (
                (forces <= self.force_limits[:, 0] + tolerance)
                | (forces >= self.force_limits[:, 1] - tolerance)
            )
        return self._get_obs(), 0.0, False, False, {
            "time": float(self.data.time),
            "control_saturated": self.last_control_saturated.copy(),
            "force_saturated": self.last_force_saturated.copy(),
        }

    def reset_model(self):
        self.set_joint_qpos(self.init_qpos.copy())
        self.set_joint_qvel(self.init_qvel.copy())
        self.mj_forward()
        self._sync_view()
        # Reset puts mocap back at its XML pose; restore the explicitly held base.
        if self.base_anchor_slot is not None:
            anchor_id = self.equality_constraint(self.base_anchor_slot)["obj1_id"]
            self.set_mocap_pos_and_quat({
                self.model.body_id2name(anchor_id): self.held_base_pose,
            })
            self.mj_forward()
            self._sync_view()
        self.right_gripper_hold.release()
        self.last_command = None
        self.last_servo_control = None
        self.last_control_saturated.fill(False)
        self.last_force_saturated.fill(False)
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.data.qpos.ravel(), self.data.qvel.ravel())).copy()
