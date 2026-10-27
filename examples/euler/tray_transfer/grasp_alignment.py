"""Measure loaded fingertip contacts in world and object coordinates.

The reported rigid translation is a calibration estimate, not a motion plan:
moving the hand changes contact locations and may collide with the object.
"""

import numpy as np

from .geometry import quat_matrix


def analyze_contacts(position, quaternion, com_local, contacts, tip_names):
    """Measure two fingers separately, excluding unloaded and support contacts."""
    origin = np.asarray(position, dtype=float)
    com_body = np.asarray(com_local, dtype=float)
    if origin.shape != (3,) or com_body.shape != (3,) or not np.isfinite([origin, com_body]).all():
        raise ValueError("Object origin and center of mass must be finite 3-vectors")
    if len(tip_names) != 2 or len(set(tip_names)) != 2:
        raise ValueError("Expected two distinct fingertip body names")
    rotation = quat_matrix(quaternion)
    com_world = origin + rotation @ com_body
    tips = []
    for name in tip_names:
        relevant = [c for c in contacts if c["other_body"] == name]
        forces = np.array([c["normal_force_n"] for c in relevant], dtype=float)
        if not np.isfinite(forces).all() or np.any(forces < 0):
            raise ValueError("Contact normal forces must be finite and nonnegative")
        loaded = [c for c in relevant if c["normal_force_n"] > 1e-8]
        if not loaded:
            continue
        points = np.array([c["position_world_m"] for c in loaded], dtype=float)
        weights = np.array([c["normal_force_n"] for c in loaded])
        if points.shape != (len(loaded), 3) or not np.isfinite(points).all():
            raise ValueError("Loaded contacts must have finite world positions")
        point = np.average(points, axis=0, weights=weights)
        tips.append(
            {
                "body": name,
                "normal_force_n": float(weights.sum()),
                "center_world_m": point.tolist(),
                "center_body_m": (rotation.T @ (point - origin)).tolist(),
            }
        )
    result = {
        "bilateral_loaded_contact": len(tips) == 2,
        "com_world_m": com_world.tolist(),
        "com_body_m": com_body.tolist(),
        "tips": tips,
    }
    if len(tips) != 2:
        return result
    points = np.array([tip["center_world_m"] for tip in tips])
    midpoint = points.mean(axis=0)
    error = midpoint - com_world
    result.update(
        midpoint_world_m=midpoint.tolist(),
        midpoint_minus_com_body_m=(rotation.T @ error).tolist(),
        horizontal_offset_world_m=float(np.linalg.norm(error[:2])),
        contact_height_difference_world_m=float(abs(points[0, 2] - points[1, 2])),
        translation_to_com_world_m=(-error).tolist(),
    )
    return result
