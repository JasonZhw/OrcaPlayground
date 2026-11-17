"""Read the scene, compute grasp targets, and orchestrate a sorting trial.

Read this file in sections: scene bindings -> task targets -> task stages -> run_trial.
Motion solving lives in planning.py, actuation in execution.py, capture in recording.py.
"""

import asyncio
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import grpc
import mujoco
import numpy as np
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
from scipy.spatial.transform import Rotation

from .execution import Executor, SortingEnv
from .planning import IKError, PlanningError, plan_motion
from .recording import SCALAR_GRIPPER_VERSION, TASK_TEXTS, RawEpisodeRecorder

# 抓持物体碰撞放松开关（临时调试用）。
# 物体已被夹住、跟着手走，它和环境（其它物体/筐壁/桌面）的接触其实是「手臂把它带到那」的
# 结果，应该由手臂自身的碰撞来兜底，不该再对抓持物体单独判碰撞——这会大量误拒 IK 解。
# 置 False 恢复严格检查。
RELAX_CARRIED_COLLISION = False

# 夹爪碰撞放松开关（临时调试用）。
# 抓取时手指包住目标后，常会碰到「紧挨着目标后面的另一个物体」（例如两个瓶子只隔 7.5cm，
# 手指长 12cm，包住前面的瓶子自然会戳到后面的）。这是正常的抓取附带接触，不是碰撞。
# 置 False 恢复严格检查。
RELAX_GRIPPER_COLLISION = False

# --- Scene bindings and geometry ---

BODY_SUFFIXES = {
    "calcium": "Medical_32",
    "melatonin": "Medical_31",
    "herbbox": "Medical_23",
    "pharyngitis": "Medical_12",
    "left_bin": "Box_Purple",
    "right_bin": "Box_Blue",
}
ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


class SceneBindingError(ValueError):
    """The scene cannot be bound uniquely and safely."""


def unique_suffix(names, suffix):
    matches = [name for name in names if name == suffix or name.endswith("_" + suffix)]
    if len(matches) != 1:
        raise SceneBindingError(f"Expected exactly one {suffix!r}; found {matches}")
    return matches[0]


def bind_scene(body_names):
    """Return semantic names without assuming generated numeric prefixes."""
    names = list(body_names)
    return {key: unique_suffix(names, suffix) for key, suffix in BODY_SUFFIXES.items()}


def prepare_model(source, dest):
    """Copy XML, resolve mesh paths, and enable six-dimensional tip friction.

    Preserve all joints, actuators and equality constraints. The Euler executor
    fixes the base through its public API, so model dimensions remain identical.
    """
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if source == dest:
        raise ValueError("Source XML must remain unchanged; choose a different destination")
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", compiler.get("assetdir", "")) if compiler is not None else ""
    for mesh in root.findall("./asset/mesh"):
        if "file" in mesh.attrib:
            file = Path(mesh.get("file"))
            candidates = [source.parent / meshdir / file, source.parent / "assets" / file]
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"Missing mesh {file}; searched {candidates}")
            mesh.set("file", str(resolved))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("impratio", "2")
    bodies = {body.get("name"): body for body in root.iter("body")}
    for side in ("l", "r"):
        for finger in ("inner", "outer"):
            name = unique_suffix(bodies, f"gripper_{side}_{finger}_link4")
            geoms = bodies[name].findall("geom")
            if not geoms:
                raise SceneBindingError(f"No fingertip collision geometry on {name}")
            for geom in geoms:
                if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
                    geom.set("condim", "6")
    # The exported four-bar gripper has convex collision meshes across its
    # closed-chain hinges. Parent filtering cannot suppress these sibling-body
    # contacts. Only exclude the three pairs within each connected linkage;
    # opposing fingers, objects and the rest of the robot still collide.
    contact = root.find("contact")
    existing = {frozenset((x.get("body1"), x.get("body2"))) for x in root.findall("./contact/exclude")}
    connected = {frozenset((x.get("body1"), x.get("body2"))) for x in root.findall("./equality/connect")}
    for side in ("l", "r"):
        for finger in ("inner", "outer"):
            tip = unique_suffix(bodies, f"gripper_{side}_{finger}_link4")
            prefix = tip.removesuffix("4")
            link2 = prefix + "2"
            if frozenset((tip, link2)) not in connected:
                continue
            for i in (1, 3, 4):
                other = prefix + str(i)
                if other not in bodies:
                    raise SceneBindingError(f"Incomplete four-bar linkage: {other}")
                pair = frozenset((other, link2))
                if pair not in existing:
                    if contact is None:
                        contact = ET.SubElement(root, "contact")
                    ET.SubElement(contact, "exclude", body1=other, body2=link2)
                    existing.add(pair)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    return dest


@dataclass(frozen=True)
class ArmBinding:
    joint_names: tuple
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    end_body_id: int
    end_site_id: int
    finger_geom_ids: frozenset
    gripper_geom_ids: frozenset
    geom_ids: frozenset
    finger_chain_geom_ids: frozenset


@dataclass(frozen=True)
class CarriedObject:
    arm: str
    body_id: int
    local_position: np.ndarray
    local_rotation: np.ndarray


class Scene:
    """Standalone public ``m``/``d`` planning state, separate from OrcaGym."""

    def __init__(self, path):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        m = self.m
        body_names = [m.body(i).name for i in range(1, m.nbody)]
        self.names = bind_scene(body_names)
        self.bodies = {key: m.body(name).id for key, name in self.names.items()}
        for key in ("calcium", "melatonin", "herbbox", "pharyngitis"):
            self.free_joint_address(self.bodies[key])
        joint_names = [m.joint(i).name for i in range(m.njnt)]
        site_names = [m.site(i).name for i in range(m.nsite)]
        self.arms = {}
        for side, short in (("left", "l"), ("right", "r")):
            names = tuple(unique_suffix(joint_names, f"{side}_{part}_joint") for part in ARM_PARTS)
            joints = [m.joint(name).id for name in names]
            if any(m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j] for j in joints):
                raise SceneBindingError(f"{side} arm requires seven bounded hinge joints")
            end = m.body(unique_suffix(body_names, f"arm_{short}_end_link")).id
            root = int(m.jnt_bodyid[joints[0]])
            tip_bodies = {
                m.body(unique_suffix(body_names, f"gripper_{short}_{finger}_link4")).id for finger in ("inner", "outer")
            }
            # 手指所有环节（link1-4），侧握时允许它们接触目标物体
            finger_chain_bodies = set()
            for finger in ("inner", "outer"):
                for i in (1, 2, 3, 4):
                    try:
                        finger_chain_bodies.add(m.body(unique_suffix(body_names, f"gripper_{short}_{finger}_link{i}")).id)
                    except Exception:
                        pass
            finger_chain = frozenset(g for g in range(m.ngeom) if m.geom_bodyid[g] in finger_chain_bodies)
            self.arms[side] = ArmBinding(
                names,
                m.jnt_qposadr[joints].copy(),
                m.jnt_dofadr[joints].copy(),
                m.jnt_range[joints, 0].copy(),
                m.jnt_range[joints, 1].copy(),
                end,
                m.site(unique_suffix(site_names, f"ee_center_site_{short}")).id,
                frozenset(g for g in range(m.ngeom) if m.geom_bodyid[g] in tip_bodies),
                self.body_geoms(end, descendants=True),
                self.body_geoms(root, descendants=True),
                finger_chain,
            )
        self.last_collisions = []
        mujoco.mj_forward(m, self.d)

    def body_id(self, body):
        if isinstance(body, str):
            return self.bodies[body] if body in self.bodies else self.m.body(body).id
        return int(body)

    def free_joint_address(self, body):
        body = self.body_id(body)
        joints = range(int(self.m.body_jntadr[body]), int(self.m.body_jntadr[body] + self.m.body_jntnum[body]))
        free = [j for j in joints if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise SceneBindingError(f"Body {self.m.body(body).name} must have one free joint")
        return int(self.m.jnt_qposadr[free[0]])

    def body_geoms(self, body, *, descendants=False):
        body = self.body_id(body)
        bodies = {body}
        if descendants:
            for i in range(body + 1, self.m.nbody):
                if int(self.m.body_parentid[i]) in bodies:
                    bodies.add(i)
        return frozenset(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in bodies)

    def set_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.m.nq,) or not np.isfinite(qpos).all():
            raise ValueError("qpos must be a finite full model configuration")
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)

    def body_pose(self, body):
        body = self.body_id(body)
        return self.d.xpos[body].copy(), self.d.xmat[body].reshape(3, 3).copy()

    def end_pose(self, arm, *, site=False):
        binding = self.arms[arm]
        if site:
            i = binding.end_site_id
            return self.d.site_xpos[i].copy(), self.d.site_xmat[i].reshape(3, 3).copy()
        return self.body_pose(binding.end_body_id)

    def attach_object(self, arm, target, qpos):
        """Measure the grasp transform for collision queries; add no constraint."""
        self.set_qpos(qpos)
        hand_p, hand_r = self.end_pose(arm)
        object_p, object_r = self.body_pose(target)
        return CarriedObject(arm, self.body_id(target), hand_r.T @ (object_p - hand_p), hand_r.T @ object_r)

    def collision_free(
        self, qpos, arm, *, target=None, allow_target_contact=False, carried=None, penetration_tolerance=0.0
    ):
        """Check actual contacts involving the moving arm or carried object.

        Only same-gripper internal contacts are ignored. Contact with an unheld target requires explicit allowance. A carried
        object retains the same finger-chain contact permission used to grasp it.
        Existing floor/scene contacts unrelated to the moving assembly are inert.
        """
        if not np.isfinite(penetration_tolerance) or penetration_tolerance < 0:
            raise ValueError("penetration_tolerance must be finite and nonnegative")
        binding = self.arms[arm]
        qpos = np.asarray(qpos, dtype=float)
        self.last_collisions = []
        self.set_qpos(qpos)
        values = qpos[binding.qpos_ids]
        if np.any(values < binding.lower) or np.any(values > binding.upper):
            return False
        carried_geoms = frozenset()
        if carried is not None:
            if carried.arm != arm:
                raise ValueError("Carried object belongs to a different arm")
            p, r = self.end_pose(arm)
            address = self.free_joint_address(carried.body_id)
            self.d.qpos[address : address + 3] = p + r @ carried.local_position
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (r @ carried.local_rotation).reshape(9))
            self.d.qpos[address + 3 : address + 7] = quat
            mujoco.mj_forward(self.m, self.d)
            carried_geoms = self.body_geoms(carried.body_id, descendants=True)
        target_geoms = self.body_geoms(target, descendants=True) if target is not None else frozenset()
        moving = binding.geom_ids | carried_geoms
        for contact in self.d.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if not ({a, b} & moving) or contact.dist > -penetration_tolerance:
                continue
            # 临时放松：抓持物体的一切接触都忽略（跟着手走，由手臂碰撞兜底）
            if RELAX_CARRIED_COLLISION and (a in carried_geoms or b in carried_geoms):
                continue
            # 临时放松：夹爪（手掌+手指）与任何非手臂几何的接触都忽略
            # （抓取时手指会碰到紧挨目标后面的物体、筐沿等）
            if RELAX_GRIPPER_COLLISION and (
                (a in binding.gripper_geom_ids and b not in binding.geom_ids)
                or (b in binding.gripper_geom_ids and a not in binding.geom_ids)
            ):
                continue
            if a in binding.gripper_geom_ids and b in binding.gripper_geom_ids:
                continue
            if a in carried_geoms and b in carried_geoms:
                continue
            allowed_object = carried_geoms | (target_geoms if allow_target_contact else frozenset())
            # 放松：整只手（手掌、手指、手腕、前臂）都允许接触【目标物体】。
            # 手往前伸时手内部碰到目标物体是正常的（顶握手掌贴盒顶、侧握手指中段贴瓶身、
            # 前臂扫过瓶身都是抓取过程的一部分），不算碰撞；其它物体仍然严格禁止接触。
            if (a in binding.geom_ids and b in allowed_object) or (
                b in binding.geom_ids and a in allowed_object
            ):
                continue
            self.last_collisions.append((self.m.geom(a).name, self.m.geom(b).name, float(contact.dist)))
        return not self.last_collisions


def vertices(scene, body):
    """Conservative world corners of collision boxes/cylinders, actual mesh vertices."""
    points = []
    for g in scene.body_geoms(body):
        m, d = scene.m, scene.d
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        size = m.geom_size[g].copy()
        typ = int(m.geom_type[g])
        if typ == 7:
            mesh = m.geom_dataid[g]
            start = m.mesh_vertadr[mesh]
            count = m.mesh_vertnum[mesh]
            local = m.mesh_vert[start : start + count]
        else:
            if typ == 5:
                size = np.array([size[0], size[0], size[1]])
            elif typ == 2:
                size[:] = size[0]
            elif typ == 3:
                size = np.array([size[0], size[0], size[0] + size[1]])
            elif typ != 6:
                raise ValueError(f"Unsupported geometry type {typ}")
            local = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * size
        points.append(local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    if not points:
        raise ValueError("No collision geometry")
    return np.concatenate(points)


def placement_satisfied(scene, object_key, bin_key):
    """Check COM inside the bin floor and object bottom within the release band.

    Orientation after placement is unrestricted. Both the per-object check and
    final four-object check use this same geometry; callers own error messages.
    """
    bin_position, bin_rotation = scene.body_pose(bin_key)
    local_vertices = (vertices(scene, object_key) - bin_position) @ bin_rotation
    center = (scene.d.xipos[scene.body_id(object_key)] - bin_position) @ bin_rotation
    floors = [
        geom
        for geom in scene.body_geoms(bin_key)
        if int(scene.m.geom_type[geom]) == 6 and scene.m.geom_size[geom, 2] < 0.015
    ]
    floor = min(floors, key=lambda geom: scene.m.geom_pos[geom, 2])
    return placement_inside(center, scene.m.geom_size[floor, :2], float(local_vertices[:, 2].min()))


def release_over_bin(scene, key, bin_key):
    """Check actual geometry before allowing a drop from the transfer hover.

    The entire footprint must fit above the floor; the object bottom must be
    above the rim and no more than 12 cm above it. Landing is checked separately.
    """
    floors = [g for g in scene.body_geoms(bin_key)
              if int(scene.m.geom_type[g]) == 6 and scene.m.geom_size[g, 2] < 0.015]
    floor = min(floors, key=lambda g: scene.m.geom_pos[g, 2])
    rotation = scene.d.geom_xmat[floor].reshape(3, 3)
    origin = scene.d.geom_xpos[floor]
    item = (vertices(scene, key) - origin) @ rotation
    rim = ((vertices(scene, bin_key) - origin) @ rotation)[:, 2].max()
    bottom = item[:, 2].min()
    return bool(np.isfinite(item).all()
                and np.all(np.abs(item[:, :2]) <= scene.m.geom_size[floor, :2])
                and rim <= bottom <= rim + 0.12)


# --- Task configuration and target positions ---

# Offsets use the current scene's world axes from the fixed pelvis position.
# G1 faces -Y in nursinghome. These are initial placement regions, not IK solutions.
BASELINE_XY = {
    "calcium": (0.09, -0.34),
    "melatonin": (0.09, -0.415),
    "pharyngitis": (-0.13, -0.30),
    "herbbox": (-0.025, -0.30),
}
DROP_SLOTS = {
    "calcium": [0, -0.045],
    "melatonin": [0, 0.045],
    "pharyngitis": [0.027, 0],
    "herbbox": [-0.021, 0],
}
# Side-grasped bottles need more wrist/rim clearance than top-grasped boxes.
RELEASE_BOTTOM_CLEARANCE = {"calcium": 0.02, "melatonin": 0.02, "pharyngitis": 0.02, "herbbox": 0.02}

# ---------------------------------------------------------------------------
# 运动规则：从示范数据反推（episode_0000-0002，50Hz，成功采集）
#
#   阶段时长中位数：clearance 2.08s | approach 5.14s | descend 1.84s
#                   lift 3.30s | transfer 3.86s | lower 2.76s
#                   withdraw 2.20s | return 3.70s | 每步后 settle 0.60s
#   关节角速度：中位 0.35 rad/s，90 分位 0.35，最大 0.54 rad/s
#   抓放高度：approach 4.880 | descend(grasp) 4.815 | lift 4.935
#             transfer 4.993 | lower 4.945
#
# 仓库原来的默认 0.4 m/s 比示范快 5–10 倍，是"动作看着猛"的直接原因。
# 每项为 (max_velocity m/s, max_acceleration m/s^2, settle 帧)。
# ---------------------------------------------------------------------------
MOTION = {
    "clearance": (0.4, 0.8, 20),   # 原默认速度
    "approach":  (0.4, 0.8, 30),
    "descend":   (0.4, 0.8, 30),
    "lift":      (0.4, 0.8, 30),
    "transfer":  (0.4, 0.8, 30),
    "lower":     (0.2, 0.4, 20),   # 原 lower 显式速度
    "withdraw":  (0.2, 0.4, 20),
    "return":    (0.4, 0.8, 30),
}

HOVER_RIM_CLEARANCE = 0.06
BOTTLE_CARRY_EXTRA_HEIGHT = 0.03
DESTINATIONS = {"calcium": "left_bin", "melatonin": "left_bin", "pharyngitis": "right_bin", "herbbox": "right_bin"}


def placement_targets(scene, key, carried, hand_rotation):
    """Keep the held item's footprint inside the bin and lower its bottom below the rim.

    Height is measured from collision geometry, not an assumed object origin.
    These are target poses; normal IK/collision and actual-pose checks still apply.
    """
    bin_key = DESTINATIONS[key]
    bin_p, bin_r = scene.body_pose(bin_key)
    object_p, object_r = scene.body_pose(key)
    local = (vertices(scene, key) - object_p) @ object_r
    projected = local @ (np.asarray(hand_rotation) @ carried.local_rotation).T @ bin_r
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    floors = [
        g for g in scene.body_geoms(bin_key) if int(scene.m.geom_type[g]) == 6 and scene.m.geom_size[g, 2] < 0.015
    ]
    floor = min(floors, key=lambda g: scene.m.geom_pos[g, 2])
    corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    floor_world = (corners * scene.m.geom_size[floor]) @ scene.d.geom_xmat[floor].reshape(3, 3).T
    floor_local = (floor_world + scene.d.geom_xpos[floor] - bin_p) @ bin_r
    floor_lo, floor_hi = floor_local.min(axis=0), floor_local.max(axis=0)
    safe_lo = floor_lo[:2] + 0.008 - lo[:2]
    safe_hi = floor_hi[:2] - 0.008 - hi[:2]
    if np.any(safe_lo > safe_hi):
        # 物体比筐略大（如 herbbox 16.2cm vs 筐地板 16cm）时不再直接报错，
        # 放在筐中心、允许边缘轻微探出——几毫米级的溢出不影响实际入框。
        xy_offset = (safe_lo + safe_hi) / 2.0
    else:
        xy_offset = np.clip(DROP_SLOTS[key], safe_lo, safe_hi)
    offset = np.r_[xy_offset, floor_hi[2] + RELEASE_BOTTOM_CLEARANCE[key] - lo[2]]
    release_object = bin_p + bin_r @ offset
    release_hand = release_object - np.asarray(hand_rotation) @ carried.local_position
    rim = ((vertices(scene, bin_key) - bin_p) @ bin_r)[:, 2].max()
    descent = max(0.06, rim + HOVER_RIM_CLEARANCE - (offset[2] + lo[2]))
    # 上限保护：筐较深时上式会算出几十厘米，导致传送点超出机械臂可达范围
    descent = min(descent, 0.18)
    if key in ("calcium", "melatonin"):
        descent += BOTTLE_CARRY_EXTRA_HEIGHT
    return {
        "release": release_hand,
        "above": release_hand + bin_r[:, 2] * descent,
        "object_position": release_object,
        "bottom_clearance": RELEASE_BOTTOM_CLEARANCE[key],
    }


def sample_layout(base, seed, xy_jitter=0.0, yaw_degrees=0.0):
    if not np.isfinite([xy_jitter, yaw_degrees]).all() or xy_jitter < 0 or yaw_degrees < 0:
        raise ValueError("Randomization ranges must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    out = {}
    for key, pose in sorted(base.items()):
        p = np.array(pose["position"], dtype=float)
        q = np.array(pose["quaternion"], dtype=float)
        if (
            p.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(np.r_[p, q]).all()
            or not np.isclose(np.linalg.norm(q), 1)
        ):
            raise ValueError("Invalid baseline pose")
        p[:2] += rng.uniform(-xy_jitter, xy_jitter, 2)
        r = Rotation.from_euler("z", rng.uniform(-yaw_degrees, yaw_degrees), degrees=True) * Rotation.from_quat(
            q[[1, 2, 3, 0]]
        )
        out[key] = {"position": p.tolist(), "quaternion": r.as_quat()[[3, 0, 1, 2]].tolist()}
    return out


def placement_inside(center_in_bin, half_floor_xy, lowest_z):
    center = np.asarray(center_in_bin)
    return bool(
        np.isfinite(np.r_[center, half_floor_xy, lowest_z]).all()
        and np.all(np.abs(center[:2]) <= half_floor_xy)
        and -0.01 <= lowest_z <= 0.08
    )


TASKS = (("left", "calcium"), ("right", "pharyngitis"), ("left", "melatonin"), ("right", "herbbox"))


def grasp_target(scene, arm, key, yaw_deg=0.0, center_xy=None, no_offset=False):
    """Return a grasp pose derived from the current object geometry (metres, world frame).

    center_xy：可选 (x, y)。视觉接地（L1）时用模型报的粗位置覆盖物体中心的
    真实 x/y，让抓取点围绕「模型看到的位置」计算，而非真值。
    no_offset：True 时不做手腕偏移（直接到物体抓取中心），agentic 流程靠
    腕摄像头微调来对齐，不预先给 0.133 偏差。
    """
    p, r = scene.body_pose(key)
    if center_xy is not None:
        p = np.array([float(center_xy[0]), float(center_xy[1]), p[2]])
    bounds = vertices(scene, key)
    height = np.ptp(bounds[:, 2])
    # Jaw closes along the object's local X; local Z is approach direction.
    jaw = r[:, 0].copy()
    jaw[2] = 0
    jaw /= np.linalg.norm(jaw)
    # Rotate about the object's own short axis so yaw randomization preserves the grasp.
    yaw_frame = np.column_stack((jaw, np.cross([0, 0, 1.0], jaw), [0, 0, 1.0]))
    rotation = (
        yaw_frame
        @ Rotation.from_euler("x", 0 if arm == "right" else -90, degrees=True).as_matrix()
        @ np.diag([-1.0, 1.0, -1.0])
    )
    # agent 指定的抓取偏航：绕竖直轴旋转接近方向，握点不变
    if yaw_deg:
        rotation = Rotation.from_euler("z", float(yaw_deg), degrees=True).as_matrix() @ rotation
    fraction = {"pharyngitis": 0.73, "herbbox": 0.70}[key] if arm == "right" else 0.51
    grip_center = p + r @ np.array([0, 0.01 if arm == "left" else 0, height * fraction])
    if no_offset:
        # agentic 流程：直接到物体抓取中心，不做手腕偏移，靠腕摄像头微调对齐
        grasp = grip_center
    else:
        # 偏移方向按物体/手臂决定（沿手的接近轴）：
        #   左臂瓶子=侧握，接近轴水平 → 手腕在物体侧后方
        #   右臂盒子=顶握，接近轴竖直 → 手腕在物体正上方
        # 偏移量=手腕到夹爪中心的距离。
        # 偏移量=手腕到夹爪中心的距离，四个物体统一用 0.16：
        #   左臂瓶子=侧握（水平），右臂盒子=顶握（竖直），值一样、方向由旋转决定。
        # 预备抓取点离物体约 17cm，给前伸 6cm + 微调留冗余；
        # X 对齐大模型判断的坐标，只有接近轴带偏移，横向靠前伸+左右微调抵消。
        grasp = grip_center - rotation @ np.array([0.0, 0.0, 0.18])
    # 预备抓取点抬高 10cm（实测 6.5cm 太低，前臂会扫到瓶子导致 IK 无解；
    # 8cm 以上就稳定可达），到手后再下放到抓取点。
    pre = grasp + np.array([0, 0, 0.08])
    return {"object_position": p, "position": grasp, "rotation": rotation, "approach": pre}


def choose_next_object(scene, arm, keys, qpos, *, seed=0):
    """Choose a nearby remaining object whose approach and descent both plan.

    This probes the private planning scene only; it does not move the simulator
    or certify the later lift and placement.
    """
    allowed = {key for assigned_arm, key in TASKS if assigned_arm == arm}
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= allowed:
        raise ValueError(f"Invalid remaining objects for {arm}: {keys}")
    qpos = np.asarray(qpos).copy()
    report = {"arm": arm, "rejected": {}}
    try:
        scene.set_qpos(qpos)
        hand, _ = scene.end_pose(arm)
        distances = {key: float(np.linalg.norm(scene.body_pose(key)[0][:2] - hand[:2])) for key in keys}
        ranked = sorted(keys, key=lambda key: (distances[key], key))
        report.update(candidates=ranked, distance_xy_m=distances)
        for key in ranked:
            scene.set_qpos(qpos)
            target = grasp_target(scene, arm, key)
            try:
                approach = plan_motion(
                    scene, arm, target["approach"], target["rotation"], qpos, target=key, seed=seed
                )
                pre_q = qpos.copy()
                pre_q[scene.arms[arm].qpos_ids] = approach[-1]
                plan_motion(
                    scene, arm, target["position"], target["rotation"], pre_q,
                    target=key, allow=True, seed=seed,
                )
            except (IKError, PlanningError) as exc:
                report["rejected"][key] = f"{type(exc).__name__}: {exc}"
                continue
            report["selected"] = key
            return key, report
        raise PlanningError(f"No feasible {arm} pickup: {report['rejected']}")
    finally:
        scene.set_qpos(qpos)


# --- Task stages: approach, grasp, carry, release, return ---


class Trial(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grasp_targets = {}
        self.place_targets = {}
        self.selections = []
        self._pending_release = None

    def lower_for_release(self, arm, key, position, rotation, carried):
        v, a, _ = MOTION["lower"]
        try:
            self.move(arm, position, rotation, key + "_lower", target=key,
                      carried=carried, allow=True, max_velocity=v, max_acceleration=a)
            return True
        except (IKError, PlanningError) as exc:
            # These exceptions occur before move sends any commands. Do not
            # catch execution failures such as lost grasp or tracking error.
            self.sync()
            if not release_over_bin(self.scene, key, DESTINATIONS[key]):
                raise
            path = self.out / "release_fallbacks.json"
            events = json.loads(path.read_text()) if path.exists() else []
            event = {"object": key, "mode": "release_from_hover", "reason": str(exc)}
            events.append(event)
            path.write_text(json.dumps(events, indent=2))
            print("RELEASE_FROM_HOVER", json.dumps(event), flush=True)
            return False

    def record_start(self):
        if self.record:
            task = TASK_TEXTS[self.task_index]
            if self.object_key != "all":
                label = {
                    "calcium": "calcium bottle",
                    "melatonin": "melatonin bottle",
                    "herbbox": "herbal drink box",
                    "pharyngitis": "pharyngitis medicine box",
                }[self.object_key]
                side = "left" if DESTINATIONS[self.object_key] == "left_bin" else "right"
                color = "purple" if side == "left" else "blue"
                task = f"Use the {side} arm to put the {label} in the {color} bin on the {side}."
            self.rec = RawEpisodeRecorder(
                self.out / "episode",
                task=task,
                fps=50.0,
                metadata={
                    "physics": {"fingertip_condim": 6, "impratio": 2, "four_bar_internal_contacts_excluded": True},
                    "control": {
                        "arm_position_feedback_gain": 12,
                        "period_seconds": 0.02,
                        "gripper_command": "one normalized closure per hand",
                        "arm_bias_compensation": True,
                        "motion_profile": "quintic_v1",
                        "max_joint_velocity": 0.4,
                        "max_joint_acceleration": 0.8,
                        "release_bottom_clearance_m": RELEASE_BOTTOM_CLEARANCE,
                        "release_ramp_seconds": 1.2,
                        "release_fallback": "checked_hover_if_lowering_unplannable",
                    },
                    "source_scene_sha256": hashlib.sha256((self.out / "scene.xml").read_bytes()).hexdigest(),
                    "actuator_joint_names": self.env.joint_names,
                    "selected_objects": [k for _, k in TASKS if self.object_key in ("all", k)],
                    "target_selection": "nearest_feasible_v1",
                    "expert_trajectory": {
                        "version": "bottle_carry_clearance_v2",
                        "planner": "static_servo_headroom_v1",
                        "bottle_lift_m": 0.12 + BOTTLE_CARRY_EXTRA_HEIGHT,
                        "bottle_hover_extra_m": BOTTLE_CARRY_EXTRA_HEIGHT,
                        "box_lift_m": 0.12,
                    },
                    "initial_layout": json.loads((self.out / "initial_layout.json").read_text()),
                    "arms": {k: list(v.joint_names) for k, v in self.scene.arms.items()},
                },
            )
            self.rec.start(self.bridge, self.loop)
            # Camera startup can yield black IDR frames. Warm up at frozen physics;
            # discard these indices so they can never align to recorded actions.
            for _ in range(12):
                self.loop.run_until_complete(
                    asyncio.wait_for(
                        self.bridge.render(
                            self.env.data.qpos.ravel().copy(),
                            float(self.env.data.time),
                            simulate_index=self.index,
                            request_idr=True,
                        ),
                        15,
                    )
                )
                self.index += 1
                time.sleep(0.08)

    def prepare_pick(self, arm, prefix):
        """抓前准备：小幅抬臂 + 张爪。

        +8cm 的 clearance 有两个作用，缺一不可：
          1) 把手臂从「初始位姿可能与筐/物体轻微接触」中解救出来，
             否则规划会报 "Start in collision"；
          2) 给 IK 一个更靠近目标的起始种子。
        张爪同理：规划时夹爪若闭合会被判起点碰撞。
        """
        self.sync()
        s, e = self.scene, self.env
        hand_p, hand_r = s.end_pose(arm)
        v, a, settle = MOTION["clearance"]
        self.move(arm, hand_p + [0, 0, 0.08], hand_r, prefix + "_clearance",
                  max_velocity=v, max_acceleration=a)
        self.hold(settle, prefix + "_clearance_settle")
        e.grippers.set(arm, 0)
        self.hold(20, prefix + "_open_clearance")

    def pick_available(self, arm, keys):
        self.prepare_pick(arm, arm)
        self.sync()
        s, e = self.scene, self.env
        binding = s.arms[arm]
        q = e.data.qpos.ravel().copy()
        ids = [e.joint_names.index(name) for name in binding.joint_names]
        q[binding.qpos_ids] = e.command[ids]
        key, report = choose_next_object(s, arm, keys, q, seed=self.index)
        self.selections.append(report)
        (self.out / "selection.json").write_text(json.dumps(self.selections, indent=2))
        print("SELECT_TARGET", json.dumps(report), flush=True)
        self.pick_place(arm, key, prepared=True)
        return key

    def pick_place(self, arm, key, *, prepared=False, yaw_deg=0.0, center_xy=None, release=True):
        if not prepared:
            self.prepare_pick(arm, key)
        s, e = self.scene, self.env
        self.sync()
        target = grasp_target(s, arm, key, yaw_deg=yaw_deg, center_xy=center_xy)
        initial = target["object_position"].copy()
        grasp, rotation, pre = target["position"], target["rotation"], target["approach"]
        self.grasp_targets[key] = {name: value.tolist() for name, value in target.items()}
        (self.out / "grasp_targets.json").write_text(json.dumps(self.grasp_targets, indent=2))
        print("GRASP_TARGET", key, "arm", arm, json.dumps(self.grasp_targets[key]), flush=True)
        v, a, settle = MOTION["approach"]
        self.move(arm, pre, rotation, key + "_approach", target=key,
                  max_velocity=v, max_acceleration=a)
        self.hold(settle, key + "_approach_settle")
        e.grippers.set(arm, 0)
        self.hold(40, key + "_open")
        v, a, settle = MOTION["descend"]
        self.move(arm, grasp, rotation, key + "_descend", target=key, allow=True,
                  max_velocity=v, max_acceleration=a)
        self.hold(settle, key + "_descend_settle")
        for closure, seconds in [(0.4, 1.2), (0.7, 1.5), (1.0, 2.0)]:
            self.ramp_gripper(arm, closure, seconds, key + "_close")
        self.hold(30, key + "_grip_hold")
        self.sync()
        carried = s.attach_object(arm, key, e.data.qpos.ravel().copy())
        lift_height = 0.12 + (BOTTLE_CARRY_EXTRA_HEIGHT if arm == "left" else 0)
        v, a, settle = MOTION["lift"]
        self.move(arm, grasp + [0, 0, lift_height], rotation, key + "_lift",
                  target=key, carried=carried, allow=True,
                  max_velocity=v, max_acceleration=a)
        self.hold(settle, key + "_lift_settle")
        self.sync()
        lifted = s.body_pose(key)[0]
        if lifted[2] - initial[2] < 0.07:
            raise RuntimeError(f"{key}: object did not follow the lift")
        bin_key = DESTINATIONS[key]
        carried = s.attach_object(arm, key, e.data.qpos.ravel().copy())
        self.carried_guard = (arm, key, carried.local_position.copy())
        bin_p, bin_r = s.body_pose(bin_key)
        # Align boxes to the bin's long axis even when the pickup yaw varied.
        release_rotation = bin_r @ carried.local_rotation.T if arm == "right" else rotation.copy()
        placement = placement_targets(s, key, carried, release_rotation)
        above = placement["above"]
        v, a, settle = MOTION["transfer"]
        self.move(arm, above, release_rotation, key + "_transfer", target=key,
                  carried=carried, allow=True, max_velocity=v, max_acceleration=a)
        self.hold(settle, key + "_transfer_settle")
        # Re-read the real grasp and bin after carrying: neither is assumed rigid.
        self.sync()
        carried = s.attach_object(arm, key, e.data.qpos.ravel().copy())
        placement = placement_targets(s, key, carried, release_rotation)
        release_hand, above = placement["release"], placement["above"]
        self.place_targets[key] = {name: np.asarray(value).tolist() for name, value in placement.items()}
        (self.out / "placement_targets.json").write_text(json.dumps(self.place_targets, indent=2))
        print("PLACE_TARGET", key, json.dumps(self.place_targets[key]), flush=True)
        lowered = self.lower_for_release(arm, key, release_hand, release_rotation, carried)
        # Arrive and hold while still fully closed, then release over 1.2 seconds.
        self.hold(15, key + "_pre_release_hold")
        if not release:
            # 夹持状态保留：等 agent 视觉确认「到位了」之后再调用 release_and_return。
            # （RobotUse：place ≠ release，释放是不可逆动作，必须由上层确认后再做。）
            self._pending_release = (arm, key, lowered, above, release_rotation)
            return
        self.carried_guard = None
        self.ramp_gripper(arm, 0, 1.2, key + "_release")
        self.hold(25, key + "_release_settle")
        if lowered:
            v, a, settle = MOTION["withdraw"]
            self.move(arm, above, release_rotation, key + "_withdraw",
                      max_velocity=v, max_acceleration=a)
            self.hold(settle, key + "_withdraw_settle")
        self.hold(50, key + "_placement_check")
        self.sync()
        if not placement_satisfied(s, key, bin_key):
            raise RuntimeError(f"{key}: object did not enter destination bin")
        self._return_home(arm, key)

    def release_and_return(self, arm, key):
        """完成一次「先 place（不释放）后 release」的抓放：张开夹爪 + 抽离 + 回位。

        前提是 pick_place(..., release=False) 已把物体降到筐内并保留夹持状态。
        """
        s, e = self.scene, self.env
        if self._pending_release is None:
            raise RuntimeError("没有待释放的夹持状态（先调用 pick_place(release=False)）")
        a2, k2, lowered, above, release_rotation = self._pending_release
        if a2 != arm or k2 != key:
            raise RuntimeError(f"待释放的是 {a2}/{k2}，不是 {arm}/{key}")
        self._pending_release = None
        self.carried_guard = None
        self.ramp_gripper(arm, 0, 1.2, key + "_release")
        self.hold(25, key + "_release_settle")
        if lowered:
            v, a, settle = MOTION["withdraw"]
            self.move(arm, above, release_rotation, key + "_withdraw",
                      max_velocity=v, max_acceleration=a)
            self.hold(settle, key + "_withdraw_settle")
        self.hold(50, key + "_placement_check")
        self.sync()
        bin_key = DESTINATIONS[key]
        if not placement_satisfied(s, key, bin_key):
            raise RuntimeError(f"{key}: object did not enter destination bin")
        self._return_home(arm, key)

    def _return_home(self, arm, key):
        s, e = self.scene, self.env
        home_q = e.data.qpos.ravel().copy()
        ids = [e.joint_names.index(n) for n in s.arms[arm].joint_names]
        home_q[s.arms[arm].qpos_ids] = self.home[ids]
        s.set_qpos(home_q)
        hp, hr = s.end_pose(arm)
        v, a, settle = MOTION["return"]
        self.move(arm, hp + [0, 0, 0.08], hr, key + "_return",
                  max_velocity=v, max_acceleration=a)
        self.hold(settle, key + "_return_settle")
        print("PLACED", key, flush=True)


# --- Run one inspected or executed trial ---


def run_trial(args):
    args.output.mkdir(parents=True, exist_ok=False)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    channel = grpc.aio.insecure_channel(args.endpoint, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)])
    bridge = OrcaStudioBridge(GrpcServiceStub(channel))
    e = None
    trial = None
    result = {"success": False, "training_ready": False, "started_at": datetime.now().isoformat()}
    result["policy_action_interface"] = {"dimensions": 16, "gripper_mapping": SCALAR_GRIPPER_VERSION}
    try:
        source = loop.run_until_complete(asyncio.wait_for(bridge.load_model_xml(), 30))
        p = prepare_model(source, args.output / "scene.xml")
        s = Scene(p)
        report = {
            "objects": {
                k: {
                    "body": s.names[k],
                    "position": s.body_pose(v)[0].tolist(),
                    "dimensions": np.ptp(vertices(s, v), axis=0).tolist(),
                    "mass_kg": float(s.m.body_mass[v]),
                }
                for k, v in s.bodies.items()
            },
            "physics": {
                "impratio": float(s.m.opt.impratio),
                "fingertip_condim": [int(s.m.geom_condim[g]) for a in s.arms.values() for g in a.finger_geom_ids],
            },
            "source_xml": source,
        }
        (args.output / "scene_report.json").write_text(json.dumps(report, indent=2))
        if not args.execute:
            result["inspection_only"] = True
            print(json.dumps(report, indent=2))
            return result
        e = SortingEnv(p, endpoint=args.endpoint)
        loop.run_until_complete(bridge.pause_simulation())
        # Reset object poses ONCE before any execution/recording, never while carrying.
        base = {}
        s.set_qpos(e.data.qpos.ravel().copy())
        pelvis = s.body_pose("unitree_g1_1_pelvis")[0]
        for key, offset in BASELINE_XY.items():
            pos, rot = s.body_pose(key)
            if args.layout_mode == "baseline":
                pos[:2] = pelvis[:2] + np.array(offset)
            quat = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
            base[key] = {"position": pos.tolist(), "quaternion": quat.tolist()}
        reference_layout = getattr(args, "reference_layout", None)
        if reference_layout is not None:
            base = json.loads(reference_layout.read_text())["poses"]
            if set(base) != set(BASELINE_XY):
                raise ValueError("Reference layout must contain all four medication poses")
        sampled = sample_layout(base, args.seed, args.xy_jitter, args.yaw_jitter_deg)
        q = e.data.qpos.ravel().copy()
        for key, pose in sampled.items():
            address = s.free_joint_address(key)
            q[address : address + 7] = pose["position"] + pose["quaternion"]
        e.set_joint_qpos(q)
        e.set_joint_qvel(np.zeros(e.model.nv))
        e.mj_forward()
        # step() refreshes the public view before the first planned motion.
        e.step(e.command)
        (args.output / "initial_layout.json").write_text(
            json.dumps({"mode": args.layout_mode, "seed": args.seed, "poses": sampled}, indent=2)
        )
        trial = Trial(e, s, bridge, loop, args.output, args.record, args.task_index, args.object)
        trial.hold(75, "initial_settle")
        trial.align_initial_command()
        trial.home = e.command.copy()
        trial.record_start()
        pending = [(arm, key) for arm, key in TASKS if args.object in ("all", key)]
        while pending:
            for arm in ("left", "right"):
                candidates = [key for assigned, key in pending if assigned == arm]
                if candidates:
                    key = trial.pick_available(arm, candidates)
                    pending.remove((arm, key))
        trial.hold(50, "final_placement_check")
        trial.sync()
        for _, key in TASKS:
            if args.object not in ["all", key]:
                continue
            if not placement_satisfied(s, key, DESTINATIONS[key]):
                raise RuntimeError(f"{key}: final placement was disturbed")
        result["success"] = True
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        print("TRIAL_FAILED", result["reason"], flush=True)
        raise
    finally:
        result["finished_at"] = datetime.now().isoformat()
        if trial:
            result["wall_seconds"] = time.monotonic() - trial.started
            if not result["success"]:
                result["failure_qpos"] = e.data.qpos.ravel().tolist()
            (args.output / "trace.json").write_text(json.dumps(trial.trace))
            np.save(args.output / "final_qpos.npy", e.data.qpos.ravel())
            if trial.rec:
                result["motion_success"] = result["success"]
                try:
                    result["recording"] = trial.rec.close(result["success"], result.get("reason", "completed"))
                    if not result["recording"]["raw_valid"]:
                        result["success"] = False
                        result.setdefault("reason", "Recorded episode failed integrity validation")
                except Exception as exc:
                    result["recording_error"] = f"{type(exc).__name__}: {exc}"
                    result["success"] = False
            result["last_stage"] = trial.trace[-1]["stage"] if trial.trace else None
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        if e:
            e.close()
        loop.run_until_complete(channel.close())
        loop.close()
    if args.execute and not result["success"]:
        raise SystemExit(1)
    return result
