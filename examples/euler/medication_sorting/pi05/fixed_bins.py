"""Static evaluation fixtures with explicit qpos mapping to the original renderer.

Only the Euler physics copy loses the bins' free joints. Studio stays paused and
receives its original joint layout, with the fixed bin poses filled in explicitly.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class FixedBinRenderMap:
    original_qpos: np.ndarray
    destination_ids: np.ndarray
    physics_nq: int
    bins: dict

    @property
    def source_nq(self):
        return len(self.original_qpos)

    def expand(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.physics_nq,) or not np.isfinite(qpos).all():
            raise ValueError("Expected finite qpos in the fixed-bin physics layout")
        expanded = self.original_qpos.copy()
        expanded[self.destination_ids] = qpos
        return expanded


class FixedBinRenderBridge:
    """Render-only adapter also used by the initial neutral-pose executor."""

    def __init__(self, bridge, mapping):
        self.bridge, self.mapping = bridge, mapping

    async def render(self, qpos, *args, **kwargs):
        return await self.bridge.render(self.mapping.expand(qpos), *args, **kwargs)


def prepare_fixed_bins(source, destination, bin_names):
    """Remove only two world-child free joints; preserve collision definitions."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("Fixed-bin model needs a new destination; never overwrite a scene")
    if len(bin_names) != 2 or len(set(bin_names)) != 2:
        raise ValueError("Exactly two distinct bins required")
    tree = ET.parse(source)
    root = tree.getroot()
    if root.findall(".//key"):
        raise ValueError("Keyframes require explicit remapping before removing joints")
    original = mujoco.MjModel.from_xml_path(str(source))
    bodies = {body.get("name"): body for body in root.findall("./worldbody/body")}
    removed, bins = set(), {}
    for name in bin_names:
        if name not in bodies:
            raise ValueError(f"Bin must be a named world-child body: {name}")
        body = bodies[name]
        joints = body.findall("joint") + body.findall("freejoint")
        if len(joints) != 1 or (joints[0].tag != "freejoint" and joints[0].get("type") != "free"):
            raise ValueError(f"Expected one bin free joint: {name}")
        joint_name = joints[0].get("name")
        if not joint_name:
            raise ValueError("Bin free joints must be named")
        address = int(original.joint(joint_name).qposadr[0])
        pose = original.qpos0[address : address + 7]
        bins[name] = {"position": pose[:3].tolist(), "quaternion_wxyz": pose[3:].tolist()}
        body.set("pos", " ".join(map(str, pose[:3])))
        for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
            body.attrib.pop(attr, None)
        body.set("quat", " ".join(map(str, pose[3:])))
        body.remove(joints[0])
        removed.add(joint_name)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("fusestatic", "false")
    # prepare_model already resolved the project's mesh paths. Compile and check
    # the new copy before returning it for use by the simulator.
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    fixed = mujoco.MjModel.from_xml_path(str(destination))
    names = [fixed.joint(i).name for i in range(fixed.njnt)]
    if set(names) != {original.joint(i).name for i in range(original.njnt)} - removed:
        raise ValueError("Unexpected joint changes while fixing bins")
    destination_ids = []
    for index, name in enumerate(names):
        joint = original.joint(name)
        width = (7, 4, 1, 1)[int(fixed.jnt_type[index])]
        if int(joint.type[0]) != int(fixed.jnt_type[index]):
            raise ValueError(f"Joint type changed: {name}")
        destination_ids.extend(range(int(joint.qposadr[0]), int(joint.qposadr[0]) + width))
    if original.nq - fixed.nq != 14 or original.nv - fixed.nv != 12:
        raise ValueError("Only two free joints may be removed")
    return FixedBinRenderMap(original.qpos0.copy(), np.asarray(destination_ids), fixed.nq, bins)
