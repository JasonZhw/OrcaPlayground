"""Build and verify a fixed-torso G1 without removing bodies or cameras."""
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from task_config import ASSETS, MODELS, ROBOT

SOURCE = ASSETS / "scene_contact_v9.xml"
OUTPUT = MODELS / "scene_single_target_v9.xml"
REPORT = MODELS / "single_target_v9_manifest.json"
NQ = {0: 7, 1: 4, 2: 1, 3: 1}


def names(model, kind, count):
    return [mj.mj_id2name(model, kind, i) for i in range(count)]


def reference():
    model = mj.MjModel.from_xml_path(str(SOURCE))
    data = mj.MjData(model)
    calibration = json.loads((ASSETS / "calibration.json").read_text())
    joint = model.joint(f"{ROBOT}_waist_pitch_joint")
    data.qpos[joint.qposadr[0]] = calibration["waist_pitch"]
    mj.mj_forward(model, data)
    return model, data


class QposBridge:
    def __init__(self, full, template, reduced, joint_aliases=None):
        self.template = np.asarray(template).copy()
        self.nq = reduced.nq
        self.pairs = []
        for jid in range(reduced.njnt):
            name = mj.mj_id2name(reduced, mj.mjtObj.mjOBJ_JOINT, jid)
            old = full.joint((joint_aliases or {}).get(name, name))
            if int(old.type[0]) != int(reduced.jnt_type[jid]):
                raise ValueError(f"Joint type mismatch: {name}")
            self.pairs.append((int(reduced.jnt_qposadr[jid]),
                               int(old.qposadr[0]), NQ[int(old.type[0])]))
        assert sum(w for _, _, w in self.pairs) == reduced.nq

    def pad(self, qpos):
        qpos = np.asarray(qpos)
        if qpos.shape != (self.nq,):
            raise ValueError(f"Unexpected reduced qpos shape: {qpos.shape}")
        output = self.template.copy()
        for source, destination, width in self.pairs:
            output[destination:destination + width] = qpos[source:source + width]
        return output


def build():
    MODELS.mkdir(parents=True, exist_ok=True)
    full, reference_data = reference()
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    suffixes = ["floating_base_joint", "waist_yaw_joint", "waist_roll_joint",
                "waist_pitch_joint"]
    suffixes += [f"{side}_{part}_joint" for side in ("left", "right")
                 for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                              "ankle_pitch", "ankle_roll")]
    drop = {f"{ROBOT}_{s}" for s in suffixes}
    existing = set(names(full, mj.mjtObj.mjOBJ_JOINT, full.njnt))
    if not drop.issubset(existing):
        raise ValueError(f"Missing expected joints: {drop - existing}")
    body_nodes = {e.get("name"): e for e in root.findall(".//body")}
    for joint_name in sorted(drop):
        jid = full.joint(joint_name).id
        bid = int(full.jnt_bodyid[jid])
        parent = int(full.body_parentid[bid])
        node = body_nodes[full.body(bid).name]
        inverse = np.zeros(4)
        mj.mju_negQuat(inverse, reference_data.xquat[parent])
        pos, quat = np.zeros(3), np.zeros(4)
        mj.mju_rotVecQuat(pos, reference_data.xpos[bid] - reference_data.xpos[parent], inverse)
        mj.mju_mulQuat(quat, inverse, reference_data.xquat[bid])
        for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
            node.attrib.pop(attr, None)
        node.set("pos", " ".join(format(v, ".17g") for v in pos))
        node.set("quat", " ".join(format(v, ".17g") for v in quat))
    removed = []
    for parent in list(root.iter()):
        for element in list(parent):
            definition = element.tag in ("joint", "freejoint") and element.get("name") in drop
            reference_hit = any(element.get(k) in drop for k in ("joint", "joint1", "joint2"))
            if definition or reference_hit:
                removed.append({"tag": element.tag, "name": element.get("name")})
                parent.remove(element)
    # Check references indirectly addressing removed actuators before accepting.
    removed_names = {r["name"] for r in removed if r["name"]}
    for element in root.iter():
        if element.get("actuator") in removed_names:
            raise ValueError("A sensor still references a removed actuator")
    if root.find("keyframe") is not None and len(root.find("keyframe")):
        raise ValueError("Keyframe qpos conversion is required")
    for asset in root.findall("./asset/*"):
        if asset.get("file"):
            asset.set("file", str((SOURCE.parent / asset.get("file")).resolve()))
    ET.indent(tree)
    candidate = OUTPUT.with_suffix(".candidate.xml")
    tree.write(candidate, encoding="utf-8", xml_declaration=True)
    reduced = mj.MjModel.from_xml_path(str(candidate))
    data = mj.MjData(reduced)
    for jid in range(reduced.njnt):
        old = full.joint(reduced.joint(jid).name)
        width = NQ[int(old.type[0])]
        dest, src = int(reduced.jnt_qposadr[jid]), int(old.qposadr[0])
        data.qpos[dest:dest + width] = reference_data.qpos[src:src + width]
    mj.mj_forward(reduced, data)
    errors = {}
    for label, kind, count, old_arr, new_arr in (
        ("body", mj.mjtObj.mjOBJ_BODY, full.nbody, reference_data.xpos, data.xpos),
        ("geom", mj.mjtObj.mjOBJ_GEOM, full.ngeom, reference_data.geom_xpos, data.geom_xpos),
        ("site", mj.mjtObj.mjOBJ_SITE, full.nsite, reference_data.site_xpos, data.site_xpos),
        ("camera", mj.mjtObj.mjOBJ_CAMERA, full.ncam, reference_data.cam_xpos, data.cam_xpos),
    ):
        count_attr = "ncam" if label == "camera" else "n" + label
        assert getattr(full, count_attr) == getattr(reduced, count_attr)
        assert names(full, kind, count) == names(reduced, kind, count)
        errors[label] = float(np.max(np.abs(old_arr - new_arr))) if count else 0.0
        if errors[label] > 1e-7:
            raise ValueError(f"{label} geometry changed: {errors[label]}")
    for attr in ("geom_contype", "geom_conaffinity", "geom_friction", "geom_condim"):
        if not np.array_equal(getattr(full, attr), getattr(reduced, attr)):
            raise ValueError(f"Collision property changed: {attr}")
    for old, new in ((reference_data.xquat, data.xquat),
                     (reference_data.site_xmat, data.site_xmat)):
        if np.max(np.abs(old - new)) > 1e-7:
            raise ValueError("Body or site orientation changed")
    bridge = QposBridge(full, reference_data.qpos, reduced)
    assert np.allclose(bridge.pad(data.qpos), reference_data.qpos, atol=1e-12)
    retained = names(reduced, mj.mjtObj.mjOBJ_JOINT, reduced.njnt)
    required = {n for n in existing if n and n.startswith(ROBOT + "_") and n not in drop}
    assert required.issubset(retained)
    report = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output": str(OUTPUT), "dropped_joints": sorted(drop), "removed_elements": removed,
        "retained_robot_joints": sorted(required), "reference_waist_pitch_rad": 0.35,
        "full_nq": full.nq, "reduced_nq": reduced.nq, "full_nu": full.nu,
        "reduced_nu": reduced.nu, "pose_max_abs_error_m": errors,
        "collision_masks_preserved": True, "qpos_bridge_verified": True,
        "scene_joint_names": [n for n in retained if not n.startswith(ROBOT + "_")],
        "camera_note": "Studio cameras and their body mounts retained; no camera joints removed",
    }
    candidate.replace(OUTPUT)
    report["output_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
