"""Read placement evidence through the Euler public API, without stepping."""

import numpy as np

from .geometry import body_bounds, primitive_bounds, quat_matrix


def placement_geometry(env, config: dict, object_body: str) -> dict:
    """Resolve the tray floor by dimensions rather than a volatile geom UUID."""
    geoms = env.model.get_geom_dict()
    floor_size = np.asarray(config["floor_size_xyz_m"])
    floors = [g for g in geoms.values()
              if g["BodyName"] == config["body_name"] and g["Type"] == 6
              and np.allclose(2 * np.asarray(g["Size"]), floor_size, atol=1e-5, rtol=0)
              and np.allclose(np.abs(quat_matrix(g["Quat"])), np.eye(3), atol=1e-5)]
    if len(floors) != 1:
        raise ValueError("Expected exactly one aligned tray-floor collider")

    def primitive(g):
        return {"geom_type": g["Type"], "geom_size": g["Size"], "geom_pos": g["Pos"],
                "geom_quat": g["Quat"], "geom_contype": g["Contype"],
                "geom_conaffinity": g["Conaffinity"]}

    floor = floors[0]
    return {**config, "object_body": object_body, "floor_id": floor["GeomId"],
            "floor_top": float(primitive_bounds(primitive(floor))[1][2]),
            "object_geoms": [primitive(g) for g in geoms.values() if g["BodyName"] == object_body],
            "geom_bodies": {g["GeomId"]: g["BodyName"] for g in geoms.values()}}


def sample_placement(env, geometry: dict) -> dict:
    """Measure whole-object containment, release, support, and velocity."""
    obj = geometry["object_body"]
    pos, quat = env.read_pose(obj)
    tray_pos, tray_quat = env.read_pose(geometry["body_name"])
    tray_rot = quat_matrix(tray_quat)
    if not np.allclose(tray_rot[:, 2], [0, 0, 1], atol=.01):
        raise ValueError("Tray tilted beyond the calibrated placement assumption")
    lower, upper = body_bounds(geometry["object_geoms"], tray_rot.T @ quat_matrix(quat),
                               tray_rot.T @ (pos - tray_pos))
    contacts = env.query_contact_simple()
    bodies = geometry["geom_bodies"]
    indices = [i for i, c in enumerate(contacts) if obj in (bodies[c["geom1"]], bodies[c["geom2"]])]
    forces = env.query_contact_force(indices)
    grip_force, floor_force, external = 0., 0., 0.
    for i in indices:
        contact = contacts[i]
        other = contact["geom2"] if bodies[contact["geom1"]] == obj else contact["geom1"]
        normal_force = float(forces[i][0])
        if not np.isfinite(normal_force):
            raise ValueError("Non-finite target contact force; placement evidence is invalid")
        force = max(0., normal_force)
        if "_gripper_" in bodies[other]:
            grip_force += force
        elif other == geometry["floor_id"]:
            floor_force += force
        else:
            # Rail/wrist support also fails: the floor must carry the released part.
            external += force
    jacp, jacr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
    env.mj_jacBody(jacp, jacr, obj)
    qvel = env.data.qvel.ravel()
    return {"object_bounds_in_tray_min": lower.tolist(), "object_bounds_in_tray_max": upper.tolist(),
            "tray_inner_xy_min": geometry["inner_xy_min"], "tray_inner_xy_max": geometry["inner_xy_max"],
            "tray_floor_top_z": geometry["floor_top"],
            "object_linear_speed_m_s": float(np.linalg.norm(jacp @ qvel)),
            "object_angular_speed_rad_s": float(np.linalg.norm(jacr @ qvel)),
            "gripper_target_normal_force_n": grip_force, "tray_floor_normal_force_n": floor_force,
            "external_support_force_n": external,
            "tray_position_world": tray_pos.tolist(), "tray_quaternion_wxyz": tray_quat.tolist()}
