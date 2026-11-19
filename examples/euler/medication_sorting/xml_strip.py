"""G1 任务模型配置与一致性校验工具。

该模块准备任务所需的模型配置，并在运行模型与展示模型之间转换 qpos。
处理范围由 agent_name 和保留列表限定；场景物体、相机与任务资源会经过
一致性校验后才投入运行。
"""

from __future__ import annotations

import re

import numpy as np

# 默认任务控制链的关节名称片段
KEEP_RIGHT_ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
KEEP_RIGHT_GRIP = ("gripper_r_inner_joint", "gripper_r_outer_joint")
KEEP_DEFAULT = KEEP_RIGHT_ARM + KEEP_RIGHT_GRIP

_JNT_NQ = {0: 7, 1: 4, 2: 1, 3: 1}
_JNT_NV = {0: 6, 1: 3, 2: 1, 3: 1}
_JNT_KIND = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

_TAG_RE = re.compile(r"<(/?)([A-Za-z_][\w:.-]*)((?:\s+[\w:.-]+\s*=\s*\"[^\"]*\")*)\s*(/?)>")
_ATTR_RE = re.compile(r"([\w:.-]+)\s*=\s*\"([^\"]*)\"")
_JOINT_DEF_TAGS = {"joint", "freejoint"}
_ACTUATOR_TAGS = {
    "general", "motor", "position", "velocity", "intvelocity",
    "damper", "cylinder", "muscle", "adhesion",
}


def _attrs(s: str) -> dict:
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(s or "")}


def _body_rel_pose(md, bid: int, pid: int):
    """子 body 相对父 body 的 pos/quat（MuJoCo wxyz）。"""
    import mujoco
    dp = np.asarray(md.xpos[bid] - md.xpos[pid], dtype=np.float64)
    qn = np.zeros(4)
    mujoco.mju_negQuat(qn, md.xquat[pid])
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, dp, qn)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, qn, md.xquat[bid])
    nrm = np.linalg.norm(quat)
    if nrm > 0:
        quat = quat / nrm
    return pos, quat


_BODY_OPEN_RE = re.compile(
    r"(<body\b)((?:\s+[\w:.-]+\s*=\s*\"[^\"]*\")*)(\s*/?>)",
)


def bake_dropped_bodies(xml: str, mj, md, drop: set) -> tuple[str, list[str]]:
    """将参考状态转换为任务模型中的 body 局部位姿。"""
    import mujoco
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in drop:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        bid = int(mj.jnt_bodyid[jid])
        bname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname:
            continue
        pid = int(mj.body_parentid[bid])
        poses[bname] = _body_rel_pose(md, bid, pid)

    if not poses:
        return xml, []

    def _repl(m):
        a = _attrs(m.group(2))
        bname = a.get("name")
        if bname not in poses:
            return m.group(0)
        pos, quat = poses[bname]
        a["pos"] = f"{pos[0]:.8g} {pos[1]:.8g} {pos[2]:.8g}"
        a["quat"] = f"{quat[0]:.8g} {quat[1]:.8g} {quat[2]:.8g} {quat[3]:.8g}"
        for k in ("euler", "axisangle", "xyaxes", "zaxis"):
            a.pop(k, None)
        parts = [m.group(1)]
        order = ["name", "pos", "quat"] + [k for k in a if k not in ("name", "pos", "quat")]
        for k in order:
            parts.append(f' {k}="{a[k]}"')
        parts.append(m.group(3))
        return "".join(parts)

    return _BODY_OPEN_RE.sub(_repl, xml), sorted(poses)


# ---------------------------------------------------------------------------
# 电柜按钮的行程约束
# ---------------------------------------------------------------------------
_BUTTON_JOINT_TOKENS = ("ElectricalCabinet_Button", "ElectricalCabinet_button")


def plan_strip(joint_names, agent_name: str, keep=KEEP_DEFAULT, keep_base: bool = False):
    """按 agent 前缀返回任务配置、控制链和场景关节分组。"""
    prefix = f"{agent_name}_"
    drop, kept, foreign = set(), [], []
    for n in joint_names:
        if not n.startswith(prefix):
            foreign.append(n)
            continue
        tail = n[len(prefix):]
        if any(k in tail for k in keep):
            kept.append(n)
        elif keep_base and "floating_base_joint" in tail:
            kept.append(n)
        else:
            drop.add(n)
    return drop, kept, foreign


# ---------------------------------------------------------------------------
# 生成任务模型 XML，并同步关联元素
# ---------------------------------------------------------------------------
def strip_xml(xml: str, drop: set):
    """按关节配置更新 XML 及其关联元素，保留场景与传感资源。"""
    report = {"joint": [], "actuator": [], "equality": [], "sensor": [],
              "tendon": [], "other_ref": []}
    if not drop:
        return xml, report

    out, pos = [], 0
    for m in _TAG_RE.finditer(xml):
        tag = m.group(2)
        a = _attrs(m.group(3))
        closing, selfclose = m.group(1) == "/", m.group(4) == "/"
        if closing:
            continue

        refs = [a[k] for k in ("joint", "joint1", "joint2") if k in a]
        hit_ref = [r for r in refs if r in drop]
        is_def = tag in _JOINT_DEF_TAGS and not refs
        hit_def = is_def and a.get("name") in drop

        if not (hit_def or hit_ref):
            continue

        if hit_def:
            report["joint"].append(a.get("name", "?"))
        elif tag in _ACTUATOR_TAGS:
            report["actuator"].append(a.get("name") or hit_ref[0])
        elif tag in ("weld", "connect", "flex", "distance") or "joint1" in a:
            report["equality"].append(a.get("name") or "|".join(refs))
        elif tag.startswith("joint") or tag in ("actuatorpos", "actuatorvel",
                                                "actuatorfrc", "jointpos",
                                                "jointvel", "jointactuatorfrc",
                                                "jointlimitpos", "jointlimitvel",
                                                "jointlimitfrc"):
            report["sensor"].append(tag)
        elif tag == "joint":
            report["tendon"].append(hit_ref[0])
        else:
            report["other_ref"].append(f"{tag}:{a.get('name') or hit_ref[0]}")

        # 同步处理自闭合元素和带内容的成对元素
        if selfclose:
            end = m.end()
        else:
            close = re.search(rf"</\s*{re.escape(tag)}\s*>", xml[m.end():])
            end = m.end() + close.end() if close else m.end()
        out.append(xml[pos:m.start()])
        pos = end

    out.append(xml[pos:])
    return "".join(out), report


# ---------------------------------------------------------------------------
# 运行模型与展示模型之间的 qpos 转换
# ---------------------------------------------------------------------------
class QposBridge:
    """按关节名将运行时 qpos 转换为 OrcaStudio 展示格式。"""

    def __init__(self, mj_full, md_full, mj_str):
        import mujoco
        self.nq_full = int(mj_full.nq)
        self.nq_str = int(mj_str.nq)
        self.template = np.asarray(md_full.qpos, dtype=np.float64).copy()
        self.pairs: list[tuple[int, int, int]] = []
        self.missing: list[str] = []
        for j in range(int(mj_str.njnt)):
            name = mujoco.mj_id2name(mj_str, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            jf = mujoco.mj_name2id(mj_full, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jf < 0:
                self.missing.append(name)
                continue
            w = _JNT_NQ[int(mj_str.jnt_type[j])]
            self.pairs.append((int(mj_str.jnt_qposadr[j]), int(mj_full.jnt_qposadr[jf]), w))
        self.covered = sum(w for _, _, w in self.pairs)

    def pad(self, qpos):
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size == self.nq_full:
            return q
        if q.size != self.nq_str:
            return q
        out = self.template.copy()
        for s, f, w in self.pairs:
            out[f:f + w] = q[s:s + w]
        return out


# ---------------------------------------------------------------------------
# 模型一致性检查
# ---------------------------------------------------------------------------
def disable_non_chain_collision(xml_path, agent_name, keep_tokens):
    """XML 层面：关闭机器人「非保留链」（躯干/腿/头/底座）的 geom 碰撞。

    只关碰撞（contype/conaffinity=0），保留视觉几何体，所以渲染里躯干还在，
    但机械臂不会再撞到它。场景物体（不以 agent_name 开头）不受影响。
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    changed = 0
    for body in root.iter("body"):
        name = body.get("name", "")
        if not (name == agent_name or name.startswith(agent_name + "_")):
            continue  # 不是机器人本体（场景物体等），跳过
        if any(tok.lower() in name.lower() for tok in keep_tokens):
            continue  # 保留链（手臂+夹爪），保留碰撞
        for geom in body.findall("geom"):
            if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                changed += 1
    if changed:
        tree.write(str(xml_path), encoding="utf-8", xml_declaration=False)
    return changed

