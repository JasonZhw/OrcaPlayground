"""场景绑定、抓取/放置几何、模型预处理。

本文件只保留 agentic 流程需要的部分：
  Scene            MuJoCo 侧的行状态（m/d）与双臂/物体绑定
  prepare_model    导出模型预处理（mesh 路径解析、指尖 condim、四连杆接触排除）
  vertices         物体碰撞几何的世界顶点（bbox 来源）
  grasp_target     抓取点/预备点的几何推导（bbox 驱动，不认物体名）
  placement_targets / placement_satisfied  放置点与入筐判据
运动求解在 planning.py，执行在 execution.py。
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# 抓持物体/夹爪碰撞放松开关（默认关闭；打开会让「手碰其他物体」也不判碰撞，仅用于排查）
RELAX_CARRIED_COLLISION = False
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


# --- 任务几何常量 ---
DROP_SLOTS = {
    "calcium": [0, -0.045],
    "melatonin": [0, 0.045],
    "pharyngitis": [0.027, 0],
    "herbbox": [-0.021, 0],
}
# Side-grasped bottles need more wrist/rim clearance than top-grasped boxes.
RELEASE_BOTTOM_CLEARANCE = {"calcium": 0.02, "melatonin": 0.02, "pharyngitis": 0.02, "herbbox": 0.02}
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


def placement_inside(center_in_bin, half_floor_xy, lowest_z):
    center = np.asarray(center_in_bin)
    return bool(
        np.isfinite(np.r_[center, half_floor_xy, lowest_z]).all()
        and np.all(np.abs(center[:2]) <= half_floor_xy)
        and -0.01 <= lowest_z <= 0.08
    )


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
