"""场景绑定与模型预处理（MuJoCo 侧）。

  Scene            行状态（m/d）、双臂/物体绑定、位姿查询、抓持关系
  prepare_model    导出模型预处理（mesh 路径解析、指尖 condim、四连杆接触排除）
  bind_scene       把语义名映射到实际 body 名（BODY_SUFFIXES）
  vertices         物体碰撞几何的世界顶点（bbox 来源，几何计算的基础）

抓取/放置的几何推导在 geometry.py，运动求解在 planning.py，执行在 execution.py。
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

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
    """把 BODY_SUFFIXES 里的语义名映射到实际 body 名。

    宽容处理：某个后缀在当前场景里没有时跳过（比如新场景没有 Medical_32），
    这样 Scene 不会因为缺某个物体就崩；真正找不到的物体由上层走场景理解。
    """
    names = list(body_names)
    bound = {}
    for key, suffix in BODY_SUFFIXES.items():
        try:
            bound[key] = unique_suffix(names, suffix)
        except SceneBindingError:
            continue
    return bound


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
            if key in self.bodies:
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

    def movable_bodies(self) -> list[str]:
        """返回场景里所有带 free joint 的可移动 body 名（物体/筐/道具）。

        供场景理解做候选：不依赖 BODY_SUFFIXES，直接从模型结构发现。
        """
        names = []
        for i in range(1, self.m.nbody):
            adr = int(self.m.body_jntadr[i])
            num = int(self.m.body_jntnum[i])
            for j in range(adr, adr + num):
                if int(self.m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                    names.append(self.m.body(i).name)
                    break
        return names

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
