"""Offline acceptance of a released object resting inside a tray."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

SCOPE = (
    "Final sampled placement hold only; does not establish a successful grasp, "
    "absence of earlier penetration, or behavior between recorded samples."
)
TOLERANCE = 1e-9


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _vector(value, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (width,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {width} finite numbers")
    return array.astype(float)


def _failed(reason: str, *, metrics: dict | None = None) -> dict:
    return {"passed": False, "reason": reason, "checks": {}, "metrics": metrics or {}, "scope": SCOPE}


def evaluate_placement(
    trace: list[dict], minimum_hold_s: float = 1., edge_clearance_m: float = .02,
    *, max_sample_gap_s: float = .05, acceptance: str = "stable",
) -> dict:
    """Validate the final contiguous ``settle_placed`` hold, without simulator access.

    Bounds and the tray floor must be expressed in the tray frame at each sample.
    Speeds are nonnegative magnitudes; contact fields are nonnegative total normal
    forces for the target, separated into gripper, tray floor, and other supports.
    The sample immediately before the requested window is included when needed
    to bracket its start. No gap may exceed ``max_sample_gap_s`` (50 ms by default).
    Object height above the tray rim is unrestricted. Invalid evidence fails closed.
    ``released_inside_tray`` instead checks the final released containment with
    1 mm contact tolerance and the existing 20 mm release height allowance;
    velocity and floor-only support are not task requirements in that mode.
    """
    try:
        if acceptance not in ("stable", "released_inside_tray"):
            raise ValueError("Unknown placement acceptance")
        minimum_hold_s = _number(minimum_hold_s, "minimum_hold_s")
        edge_clearance_m = _number(edge_clearance_m, "edge_clearance_m")
        max_sample_gap_s = _number(max_sample_gap_s, "max_sample_gap_s")
        if minimum_hold_s <= 0 or edge_clearance_m < 0 or max_sample_gap_s <= 0:
            raise ValueError("Hold duration and sample gap must be positive; clearance must be nonnegative")
        if acceptance == "released_inside_tray":
            edge_clearance_m = 0.
        if not isinstance(trace, list) or len(trace) < 2:
            raise ValueError("At least two complete trace samples are required")
        rows = []
        for index, sample in enumerate(trace):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {index} must be a mapping")
            phase = sample["phase"]
            if not isinstance(phase, str) or not phase:
                raise ValueError(f"Sample {index} has no valid phase")
            row = {"phase": phase}
            for field in (
                "time_s", "tray_floor_top_z", "object_linear_speed_m_s", "object_angular_speed_rad_s",
                "gripper_target_normal_force_n", "tray_floor_normal_force_n", "external_support_force_n",
            ):
                row[field] = _number(sample[field], f"Sample {index}: {field}")
                if field != "tray_floor_top_z" and row[field] < 0:
                    raise ValueError(f"Sample {index}: {field} must be nonnegative")
            for field, width in (
                ("object_bounds_in_tray_min", 3), ("object_bounds_in_tray_max", 3),
                ("tray_inner_xy_min", 2), ("tray_inner_xy_max", 2),
            ):
                row[field] = _vector(sample[field], width, f"Sample {index}: {field}")
            if np.any(row["object_bounds_in_tray_min"] >= row["object_bounds_in_tray_max"]):
                raise ValueError(f"Sample {index} has empty or inverted object bounds")
            if np.any(row["tray_inner_xy_min"] + 2 * edge_clearance_m >= row["tray_inner_xy_max"]):
                raise ValueError(f"Sample {index} has no tray interior after edge clearance")
            if rows and row["time_s"] <= rows[-1]["time_s"]:
                raise ValueError("Trace times must be strictly increasing")
            rows.append(row)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _failed(f"Invalid placement evidence: {error}")

    if rows[-1]["phase"] != "settle_placed":
        return _failed("The final phase must be settle_placed")
    if acceptance == "released_inside_tray":
        # User task completion does not require a motionless, floor-only hold.
        # The original stable path below remains available for quality review.
        row = rows[-1]
        try:
            gripper = trace[-1]["gripper"]
            if type(gripper["hold_active"]) is not bool:
                raise ValueError("Gripper latch state must be a boolean")
            command = _vector(gripper["nominal_command"], 8, "gripper nominal command")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return _failed(f"Invalid release evidence: {error}")
        margin = float(np.minimum(row["object_bounds_in_tray_min"][:2] - row["tray_inner_xy_min"],
                                  row["tray_inner_xy_max"] - row["object_bounds_in_tray_max"][:2]).min())
        gap = float(row["object_bounds_in_tray_min"][2] - row["tray_floor_top_z"])
        checks = {
            "sampling_gap_within_limit": max(np.diff([r["time_s"] for r in rows])) <= max_sample_gap_s + TOLERANCE,
            "whole_bounds_inside_tray": margin >= -.001 - TOLERANCE,
            "bottom_near_tray_floor": -.002 - TOLERANCE <= gap <= .02 + TOLERANCE,
            "gripper_released": row["gripper_target_normal_force_n"] < .1 and not gripper["hold_active"],
            "gripper_open_command": bool(np.array_equal(command, [0., 0., 0., -1., 0., 0., 0., -1.])),
        }
        return {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()},
                "success_criterion": acceptance,
                "metrics": {"sample_time_s": row["time_s"], "minimum_edge_clearance_m": margin,
                            "bottom_gap_m": gap, "contact_tolerance_m": .001},
                "scope": "Final sampled released containment only; does not require stillness or floor-only support."}
    start = len(rows) - 1
    while start > 0 and rows[start - 1]["phase"] == "settle_placed":
        start -= 1
    hold = rows[start:]
    span = hold[-1]["time_s"] - hold[0]["time_s"]
    if span + TOLERANCE < minimum_hold_s:
        return _failed("The final contiguous hold is too short", metrics={"available_hold_span_s": span})
    cutoff = hold[-1]["time_s"] - minimum_hold_s
    first = max(i for i, row in enumerate(hold) if row["time_s"] <= cutoff + TOLERANCE)
    window = hold[first:]
    if len(window) < 2:
        return _failed("Insufficient samples in the requested hold window")

    gaps = np.diff([row["time_s"] for row in window])
    lower = np.stack([row["object_bounds_in_tray_min"] for row in window])
    upper = np.stack([row["object_bounds_in_tray_max"] for row in window])
    inner_min = np.stack([row["tray_inner_xy_min"] for row in window])
    inner_max = np.stack([row["tray_inner_xy_max"] for row in window])
    margins = np.minimum(lower[:, :2] - inner_min, inner_max - upper[:, :2])
    bottom_gap = lower[:, 2] - np.array([row["tray_floor_top_z"] for row in window])
    metrics = {
        "hold_samples": len(window),
        "hold_start_s": window[0]["time_s"], "hold_end_s": window[-1]["time_s"],
        "hold_span_s": window[-1]["time_s"] - window[0]["time_s"],
        "max_sample_gap_s": float(gaps.max()),
        "minimum_edge_clearance_m": float(margins.min()),
        "minimum_bottom_gap_m": float(bottom_gap.min()), "maximum_bottom_gap_m": float(bottom_gap.max()),
        "max_linear_speed_m_s": max(row["object_linear_speed_m_s"] for row in window),
        "max_angular_speed_rad_s": max(row["object_angular_speed_rad_s"] for row in window),
        "max_gripper_target_normal_force_n": max(row["gripper_target_normal_force_n"] for row in window),
        "min_tray_floor_normal_force_n": min(row["tray_floor_normal_force_n"] for row in window),
        "max_external_support_force_n": max(row["external_support_force_n"] for row in window),
    }
    checks = {
        "sampling_gap_within_limit": metrics["max_sample_gap_s"] <= max_sample_gap_s + TOLERANCE,
        "whole_bounds_inside_tray": metrics["minimum_edge_clearance_m"] >= edge_clearance_m - TOLERANCE,
        "bottom_at_tray_floor": (metrics["minimum_bottom_gap_m"] >= -.002 - TOLERANCE
                                 and metrics["maximum_bottom_gap_m"] <= .005 + TOLERANCE),
        "gripper_released": metrics["max_gripper_target_normal_force_n"] < .1,
        "supported_by_tray_floor": metrics["min_tray_floor_normal_force_n"] > .5,
        "no_external_support": metrics["max_external_support_force_n"] < .1,
        "linear_speed_below_limit": metrics["max_linear_speed_m_s"] < .01,
        "angular_speed_below_limit": metrics["max_angular_speed_rad_s"] < .1,
    }
    passed = all(checks.values())
    return {
        "passed": passed, "checks": checks, "metrics": metrics, "scope": SCOPE,
        "reason": "Placement hold passed" if passed else "Failed: " + ", ".join(
            name for name, okay in checks.items() if not okay
        ),
    }
