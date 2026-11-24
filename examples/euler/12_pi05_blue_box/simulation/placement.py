"""Full collision-box containment and physical basket-floor support."""

import mujoco
import numpy as np

from build_model import OUTPUT
from task_config import TARGET


def stable_frame(lift, clearance, supported, released, speed):
    return bool(lift > 0.08 and clearance >= -0.001 and supported
                and released and speed < 0.02)


class Placement:
    def __init__(self, env):
        model = mujoco.MjModel.from_xml_path(str(OUTPUT))
        self.basket = next(n for n in env.model.get_body_names() if n.endswith("basket_01"))
        body = model.body(TARGET).id
        self.target_geom = int(model.body_geomadr[body])
        if model.body_geomnum[body] != 1 or model.geom_type[self.target_geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Expected one box collision geometry for the target")
        self.offset = model.geom_pos[self.target_geom].copy()
        self.halfsize = model.geom_size[self.target_geom].copy()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[self.target_geom])
        self.object_rotation = rotation.reshape(3, 3)
        self.velocity_address = int(model.joint(int(model.body_jntadr[body])).dofadr[0])
        self.inner_min, self.inner_max = np.full(2, -np.inf), np.full(2, np.inf)
        basket = model.body(self.basket).id
        begin = int(model.body_geomadr[basket])
        floor = []
        for gid in range(begin, begin + int(model.body_geomnum[basket])):
            pos, dims = model.geom_pos[gid], model.geom_size[gid]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, model.geom_quat[gid])
            span = np.abs(matrix.reshape(3, 3)) @ dims
            if dims[0] > 0.1 and dims[1] > 0.1:
                floor.append(gid)
            for axis in range(2):
                if dims[1 - axis] > 0.1 and dims[axis] < 0.02:
                    if pos[axis] < 0:
                        self.inner_min[axis] = max(self.inner_min[axis], pos[axis] + span[axis])
                    else:
                        self.inner_max[axis] = min(self.inner_max[axis], pos[axis] - span[axis])
        if len(floor) != 1 or not np.isfinite([self.inner_min, self.inner_max]).all():
            raise ValueError("Could not derive basket interior and floor")
        self.floor_geom = floor[0]
        self.geom_bodies = {g["GeomId"]: g["BodyName"] for g in env.model.get_geom_dict().values()}
        if (self.geom_bodies[self.floor_geom] != self.basket
                or self.geom_bodies[self.target_geom] != TARGET):
            raise ValueError("Public geometry mapping differs from the scene")

    def measure(self, env):
        poses = env.get_body_xpos_xmat_xquat([TARGET, self.basket])
        target, basket = poses[TARGET], poses[self.basket]
        rt = np.asarray(target["xmat"]).reshape(3, 3)
        rb = np.asarray(basket["xmat"]).reshape(3, 3)
        center = rb.T @ (np.asarray(target["xpos"]) + rt @ self.offset - basket["xpos"])
        extent = np.abs(rb.T @ rt @ self.object_rotation) @ self.halfsize
        clearance = float(min(np.min(center[:2] - extent[:2] - self.inner_min),
                              np.min(self.inner_max - center[:2] - extent[:2])))
        supported, touching = False, False
        for contact in env.query_contact_simple():
            pair = (contact["geom1"], contact["geom2"])
            if self.target_geom in pair:
                other = pair[1] if pair[0] == self.target_geom else pair[0]
                supported |= other == self.floor_geom
                touching |= "gripper_r_" in self.geom_bodies[other]
        speed = float(np.linalg.norm(env.data.qvel[self.velocity_address:self.velocity_address + 3]))
        return clearance, bool(supported), not touching, speed
