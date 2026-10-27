"""Pure collision-primitive bounds for tray preflight, in meters.

Inputs use public QueryAllGeoms fields: poses are body-local and quaternions
are MuJoCo wxyz. These bounds do not establish reachability or grasp success.
"""

import numpy as np


def _vector(value, length, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return vector


def _rotation(value):
    if value is None:
        return np.eye(3)
    rotation = np.asarray(value, dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("rotation must be a finite proper 3x3 rotation matrix")
    return rotation


def quat_matrix(wxyz):
    """Return the rotation matrix of a finite nonzero wxyz quaternion."""
    quaternion = _vector(wxyz, 4, "quaternion")
    scale = np.max(np.abs(quaternion))
    if scale == 0:
        raise ValueError("quaternion must not be zero")
    quaternion = quaternion / scale
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def primitive_bounds(geom, frame_rotation=None, frame_translation=None):
    """Exact AABB of one primitive after transforming its body's frame.

    Cylinder size is radius, half-height; capsule size is radius, half-length
    of the central segment. Unused size entries are ignored. Meshes require
    vertices and are deliberately rejected rather than approximated by size.
    """
    try:
        kind = geom["geom_type"]
        size = _vector(geom["geom_size"], 3, "geom_size")
        position = _vector(geom["geom_pos"], 3, "geom_pos")
        local_rotation = quat_matrix(geom["geom_quat"])
    except KeyError as exc:
        raise ValueError(f"missing geometry field: {exc.args[0]}") from exc
    if kind not in (2, 3, 4, 5, 6):
        raise ValueError(f"unsupported geom_type {kind}; collision meshes need vertex data")
    if size[0] <= 0 or (kind in (4, 6) and np.any(size <= 0)) or (
        kind in (3, 5) and size[1] < 0
    ):
        raise ValueError("primitive radii/half-sizes must be positive; axial half-length may be zero")

    frame = _rotation(frame_rotation)
    translation = np.zeros(3) if frame_translation is None else _vector(
        frame_translation, 3, "frame_translation"
    )
    rotation = frame @ local_rotation
    center = frame @ position + translation
    if kind == 2:  # sphere
        extent = np.full(3, size[0])
    elif kind == 6:  # box
        extent = np.abs(rotation) @ size
    elif kind == 4:  # ellipsoid
        extent = np.linalg.norm(rotation * size, axis=1)
    else:
        axis = np.clip(np.abs(rotation[:, 2]), 0, 1)
        if kind == 3:  # capsule = line segment swept by a sphere
            extent = size[0] + size[1] * axis
        else:  # cylinder = line segment swept by a perpendicular disk
            extent = size[1] * axis + size[0] * np.sqrt(np.maximum(0, 1 - axis**2))
    lower, upper = center - extent, center + extent
    if not np.all(np.isfinite([lower, upper])):
        raise ValueError("transformed geometry bounds are not finite")
    return lower, upper


def body_bounds(geoms, rotation=None, translation=None):
    """Union collision primitives attached to one body; ignore visual geoms.

    Callers must supply geoms in the same body's local frame, not concatenate
    differently posed child bodies without transforming them first.
    """
    bounds = [
        primitive_bounds(geom, rotation, translation)
        for geom in geoms
        if geom.get("geom_contype", 0) or geom.get("geom_conaffinity", 0)
    ]
    if not bounds:
        raise ValueError("body has no collision geoms")
    lower, upper = zip(*bounds)
    return np.min(lower, axis=0), np.max(upper, axis=0)


def candidate_placement(body_geoms, object_rotation_in_tray, center_xy, tray_floor_z, clearance=0):
    """Place the rotated collision-AABB center at XY and its bottom above floor.

    Returns a body-origin pose and bounds in tray coordinates as JSON-ready
    lists. This is geometric fit only: support stability and contacts still
    require validation in simulation.
    """
    target_xy = _vector(center_xy, 2, "center_xy")
    floor, gap = _vector([tray_floor_z, clearance], 2, "floor and clearance")
    if gap < 0:
        raise ValueError("clearance must be nonnegative")
    lower, upper = body_bounds(body_geoms, rotation=object_rotation_in_tray)
    origin = np.r_[target_xy - (lower[:2] + upper[:2]) / 2, floor + gap - lower[2]]
    return {
        "body_origin_tray": origin.tolist(),
        "bounds_min": (lower + origin).tolist(),
        "bounds_max": (upper + origin).tolist(),
        "size": (upper - lower).tolist(),
    }
