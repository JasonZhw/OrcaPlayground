"""Measured alignment for upright boxes in the simulator, not visual servoing."""

import numpy as np

from .skills import SkillResult


def box_grasp_alignment(object_vertices, finger_vertices, rotation, target_xy, *, fraction):
    """Align the actual open pads to a locked visual XY and a box-relative height.

    Geometry supplies identity/depth checks; it never silently replaces visual XY.
    A proposed delta still requires the normal IK and collision-checked motion.
    """
    obj = np.asarray(object_vertices, dtype=float)
    pads = [np.asarray(v, dtype=float) for v in finger_vertices]
    r, xy = np.asarray(rotation, dtype=float), np.asarray(target_xy, dtype=float)
    if (
        obj.ndim != 2
        or obj.shape[1:] != (3,)
        or len(obj) < 4
        or len(pads) != 2
        or any(v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4 for v in pads)
        or r.shape != (3, 3)
        or xy.shape != (2,)
        or not all(np.isfinite(v).all() for v in [obj, *pads, r, xy])
        or not np.isfinite(fraction)
        or not 0 < fraction < 1
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(r), 1)
    ):
        return SkillResult(False, "unknown", "Invalid box/pad alignment geometry")
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    if np.linalg.norm(xy - (lo[:2] + hi[:2]) / 2) > 0.025:
        return SkillResult(False, "target_mismatch", "Visual target differs from the identified box by over 25 mm")
    desired = np.r_[xy, lo[2] + fraction * (hi[2] - lo[2])]
    # Bounding centers in the hand frame avoid mesh tessellation weighting.
    centers = [(v @ r).min(axis=0) / 2 + (v @ r).max(axis=0) / 2 for v in pads]
    center = np.mean(centers, axis=0) @ r.T
    local_pads = [(v - center) @ r for v in pads]
    negative, positive = sorted(local_pads, key=lambda v: np.mean(v[:, 0]))
    gap = [float(negative[:, 0].max()), float(positive[:, 0].min())]
    local_object = (obj - desired) @ r
    if local_object[:, 0].min() < gap[0] + 0.001 or local_object[:, 0].max() > gap[1] - 0.001:
        return SkillResult(False, "grasp_geometry", "Box does not fit between open finger pads at the visual target")
    actual_object = (obj - center) @ r
    actual_fit = actual_object[:, 0].min() >= gap[0] + 0.001 and actual_object[:, 0].max() <= gap[1] - 0.001
    error = desired - center
    overlap = [float(min(v[:, 2].max(), hi[2]) - max(v[:, 2].min(), lo[2])) for v in pads]
    ready = actual_fit and np.linalg.norm(error[:2]) <= 0.008 and abs(error[2]) <= 0.006 and min(overlap) >= 0.02
    delta = np.zeros(3)
    if not ready:
        # Correct horizontal alignment at clearance height before going down.
        if np.linalg.norm(error[:2]) > 0.008 or not actual_fit:
            if np.linalg.norm(error[:2]) < 1e-8:
                return SkillResult(False, "grasp_geometry", "Current pad gap cannot be corrected horizontally")
            delta[:2] = error[:2] * min(1.0, 0.01 / np.linalg.norm(error[:2]))
        elif abs(error[2]) > 0.006:
            delta[2] = np.clip(error[2], -0.01, 0.01)
        else:
            return SkillResult(False, "grasp_geometry", "Insufficient pad/box overlap at the calibrated height")
    return SkillResult(
        True,
        "ok",
        detail={
            "ready": bool(ready),
            "actual_gap_fit": bool(actual_fit),
            "delta": delta.tolist(),
            "error_m": error.tolist(),
            "pad_center": center.tolist(),
            "desired_center": desired.tolist(),
            "overlap_m": overlap,
            "source": "visual_xy_and_simulator_pad_geometry",
        },
    )
