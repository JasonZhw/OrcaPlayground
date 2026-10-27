"""Geometry preflight for the two-object task; this module never controls the robot."""

import numpy as np

from .geometry import body_bounds, candidate_placement, primitive_bounds, quat_matrix


def resolve_name(names, suffix):
    """Match stable asset suffixes while refusing duplicate scene instances."""
    matches = [name for name in names if name == suffix or name.endswith("_" + suffix)]
    if len(matches) != 1:
        reason = "missing" if not matches else "ambiguous"
        raise ValueError(f"{reason} name {suffix!r}: {matches}")
    return matches[0]


def boxes_overlap(lower_a, upper_a, lower_b, upper_b):
    return bool(np.all(np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b) > 1e-8))


def build_report(snapshot, config):
    """Compute candidate resting poses from current physics frames, not editor actors.

    Objects keep their current world orientation. Results are geometric candidates,
    not TCP commands, IK solutions, grasp certificates or success measurements.
    """
    responses = snapshot["responses"]
    bodies = {body["name"]: body for body in responses["QueryAllBodies"]["body_info"]}
    poses = {pose["body_name"]: pose for pose in
             responses["QueryBodyPosMatQuat"]["body_pos_mat_quat_list"]}
    all_geoms = responses["QueryAllGeoms"]["geom_data_list"]
    joints = responses["QueryAllJoints"]["joint_info"]
    sites = {site["name"]: site for site in responses["QueryAllSites"]["site_info"]}
    errors = []

    def body_pose(name):
        pose = poses[name]
        return np.asarray(pose["pos"], dtype=float), quat_matrix(pose["quat"])

    def collision_geoms(name):
        # Current task assets each have one rigid body. Refuse to silently omit a child collider.
        descendant_ids = {bodies[name]["id"]}
        for _ in range(len(bodies)):
            expanded = descendant_ids | {
                body["id"] for body in bodies.values() if body["parent_id"] in descendant_ids
            }
            if expanded == descendant_ids:
                break
            descendant_ids = expanded
        descendants = {b["name"] for b in bodies.values() if b["id"] in descendant_ids}
        geoms = [g for g in all_geoms if g["body_name"] in descendants
                 and (g["geom_contype"] or g["geom_conaffinity"])]
        if any(g["body_name"] != name for g in geoms):
            raise ValueError(f"{name}: child-body colliders require a rigid-frame adapter")
        if not geoms:
            raise ValueError(f"{name}: no collision geometry")
        return geoms

    tray_config = config["tray"]
    tray_name = resolve_name(bodies, tray_config["body_suffix"])
    base_name = resolve_name(bodies, config["robot"]["base_body"])
    tray_pos, tray_rot = body_pose(tray_name)
    base_pos, base_rot = body_pose(base_name)
    tray_geoms = collision_geoms(tray_name)
    # Runtime geom UUIDs change after reload. Match the measured floor's shape instead.
    expected_size = np.asarray(tray_config["floor_size_xyz_m"], dtype=float)
    if expected_size.shape != (3,) or not np.all(np.isfinite(expected_size)) or np.any(expected_size <= 0):
        raise ValueError("floor_size_xyz_m must contain three positive full dimensions")
    floors = [g for g in tray_geoms if g["geom_type"] == 6
              and np.allclose(2 * np.asarray(g["geom_size"]), expected_size, atol=1e-5, rtol=0)
              and np.allclose(np.abs(quat_matrix(g["geom_quat"])), np.eye(3), atol=1e-5)]
    if len(floors) != 1:
        raise ValueError(f"Expected one aligned floor with size {expected_size.tolist()}, found {len(floors)}")
    floor = floors[0]
    floor_min, floor_max = primitive_bounds(floor)
    inner_min = np.asarray(tray_config["inner_xy_min"], dtype=float)
    inner_max = np.asarray(tray_config["inner_xy_max"], dtype=float)
    margin = float(tray_config["edge_clearance_m"])
    release_clearance = float(tray_config["release_clearance_m"])
    if (inner_min.shape != (2,) or inner_max.shape != (2,)
            or not np.all(np.isfinite([inner_min, inner_max]))
            or not np.isfinite(margin) or margin < 0
            or not np.isfinite(release_clearance) or release_clearance < 0
            or np.any(inner_min + margin >= inner_max - margin)):
        raise ValueError("Invalid tray interior or clearance in task configuration")
    if np.any(inner_min < floor_min[:2]) or np.any(inner_max > floor_max[:2]):
        errors.append("Configured tray opening extends beyond the physical floor")
    if not np.allclose(tray_rot[:, 2], [0, 0, 1], atol=1e-3):
        errors.append("Tray is tilted; upright placement assumptions need recalibration")

    objects = []
    used_names = {tray_name, base_name}
    for item in config["objects"]:
        name = resolve_name(bodies, item["body_suffix"])
        if name in used_names:
            raise ValueError(f"Task roles must use distinct bodies: {name}")
        used_names.add(name)
        geoms = collision_geoms(name)
        pos, rot = body_pose(name)
        lower, upper = body_bounds(geoms)
        placement = candidate_placement(
            geoms, tray_rot.T @ rot, item["placement_center_xy"], float(floor_max[2]), release_clearance
        )
        p_min, p_max = np.array(placement["bounds_min"]), np.array(placement["bounds_max"])
        if np.any(p_min[:2] < inner_min + margin) or np.any(p_max[:2] > inner_max - margin):
            errors.append(f"{item['id']}: placement exceeds the tray interior with edge clearance")
        for geom in tray_geoms:
            g_min, g_max = primitive_bounds(geom)
            if boxes_overlap(p_min, p_max, g_min, g_max):
                errors.append(f"{item['id']}: candidate envelope overlaps tray collider {geom['geom_name']}")
        origin_world = tray_pos + tray_rot @ np.asarray(placement["body_origin_tray"])
        placement.update({
            "body_origin_world": origin_world.tolist(),
            "body_origin_base": (base_rot.T @ (origin_world - base_pos)).tolist(),
            "body_quat_world_wxyz": poses[name]["quat"],
            "orientation_rule": "preserve_current_world_orientation",
        })
        objects.append({
            "id": item["id"], "body_name": name, "mass_kg": bodies[name]["mass"],
            "body_origin_world": pos.tolist(), "body_origin_base": (base_rot.T @ (pos - base_pos)).tolist(),
            "collision_size_body_m": (upper - lower).tolist(),
            "collision_bounds_body": [lower.tolist(), upper.tolist()],
            "collision_geom_count": len(geoms), "placement": placement,
        })
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            a, b = first["placement"], second["placement"]
            if boxes_overlap(a["bounds_min"], a["bounds_max"], b["bounds_min"], b["bounds_max"]):
                errors.append(f"Placement envelopes overlap: {first['id']} and {second['id']}")
    robot_sites = {}
    for name in config["robot"]["tcp_sites"]:
        site = sites[resolve_name(sites, name)]
        robot_sites[name] = {
            "pos_world": site["pos"], "local_pos": site["local_pos"],
            "local_quat_wxyz": site["local_quat"], "body_id": site["site_bodyid"],
        }
    return {
        "task_id": config["task_id"], "scope": "geometry_preflight_only",
        "object_order": [obj["id"] for obj in objects],
        "geometry_passed": not errors, "geometry_errors": errors,
        "robot": {
            "base_body": base_name, "base_origin_world": base_pos.tolist(),
            "base_quat_world_wxyz": poses[base_name]["quat"],
            "base_has_free_joint": any(j["type"] == 0 and j["body_id"] == bodies[base_name]["id"] for j in joints),
            "manipulation_anchor": responses.get("GetBodyManipulationAnchored"),
            "tcp_sites": robot_sites,
        },
        "tray": {
            "body_name": tray_name, "origin_world": tray_pos.tolist(),
            "quat_world_wxyz": poses[tray_name]["quat"], "floor_top_local_z": float(floor_max[2]),
            "inner_xy_min": inner_min.tolist(), "inner_xy_max": inner_max.tolist(),
            "edge_clearance_m": margin, "release_clearance_m": release_clearance,
        },
        "objects": objects,
        "camera_names": responses["GetCameraNames"]["camera_names"],
        "unverified": ["base_stabilization", "gripper_opening_and_grasp", "ik_and_path_collisions",
                       "payload_and_slip", "stable_release_of_both_objects", "camera_frames_and_sync"],
    }
