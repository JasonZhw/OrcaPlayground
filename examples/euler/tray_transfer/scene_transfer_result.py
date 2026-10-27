"""Sampled grasp, carry and release acceptance for the current scene plan."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .blue_pick import TIPS, evaluate_hold
from .placement_result import evaluate_placement

TRANSFER_PHASES = (
    "lift_hold", "transport_raise", "transport_over_tray", "transport_lower",
    "release", "retreat", "settle_placed",
)
SCOPE = "Recorded samples only; not proof of every physics substep or randomized task robustness."


def _number(value, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Expected a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Expected a finite number in the permitted range")
    return value


def _array(value, shape):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"Expected finite numeric array of shape {shape}")
    return array.astype(float)


def evaluate_scene_transfer(
    trace: list[dict], initial_position, *, edge_clearance_m: float = .02,
    contact_diagnostics: bool = True,
) -> dict:
    """Evaluate release separately from retention, so intentional opening is allowed.

    Each row contains the normal scene-pick fields and ``sample_placement``'s
    flat fields. The final phase sequence is exactly ``TRANSFER_PHASES``.
    Finger penetration and relative drift are diagnostics in retention mode.
    Tray-floor support is allowed during lowering; both fingertips must remain
    in contact until the first release sample. All other support still fails.
    """
    criterion = "transfer_retention" if contact_diagnostics else "transfer_strict_contact_diagnostic"
    base = {"success_criterion": criterion, "scope": SCOPE, "training_data": False}
    try:
        if type(contact_diagnostics) is not bool:
            raise ValueError("contact_diagnostics must be a boolean")
        initial = _array(initial_position, (3,))
        if not isinstance(trace, list) or len(trace) < 2:
            raise ValueError("Missing complete transfer trace")
        phases, times, groups = [], [], []
        for row in trace:
            phase = row["phase"]
            if not isinstance(phase, str) or not phase:
                raise ValueError("Every sample needs a phase")
            phases.append(phase)
            if not groups or phase != groups[-1]:
                groups.append(phase)
            times.append(_number(row["time_s"], nonnegative=True))
            if len(times) > 1 and times[-1] <= times[-2]:
                raise ValueError("Trace times must strictly increase")
            _number(row["lowest_collider_clearance_m"])
            _array(row["blue_position"], (3,))
            _array(row["relative_position"], (3,))
            rotation = _array(row["relative_rotation"], (3, 3))
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
                raise ValueError("Relative rotation must be a proper rotation matrix")
            if not isinstance(row["contacts"], list):
                raise ValueError("Contacts must be a list")
            for contact in row["contacts"]:
                if not isinstance(contact["other_body"], str) or not contact["other_body"]:
                    raise ValueError("Every contact needs an other_body")
                _number(contact["normal_force_n"], nonnegative=True)
                _number(contact["distance_m"])
            for name in ("tray_floor_normal_force_n", "external_support_force_n"):
                _number(row[name], nonnegative=True)
        start = groups.index("lift_hold")
        if groups[start:] != list(TRANSFER_PHASES) or any(p in TRANSFER_PHASES for p in groups[:start]):
            raise ValueError("Expected one ordered lift, transport, release, retreat and placement sequence")
        lift_end = phases.index("transport_raise")
        release = phases.index("release")
        first = next((i for i, row in enumerate(trace[:lift_end])
                      if row["lowest_collider_clearance_m"] >= .002), None)
        if first is None:
            raise ValueError("No separation from the initial support before transport")

        # Calling the legacy grasp evaluator only on the grasp prefix prevents
        # intentional release and later floor contact from contaminating it.
        grasp = evaluate_hold(trace[:lift_end], initial.tolist())
        if contact_diagnostics and "checks" in grasp:
            grasp["diagnostic_checks"] = {
                key: grasp["checks"].pop(key) for key in (
                    "relative_translation_under_5mm", "relative_rotation_under_5deg",
                    "penetration_under_2mm", "trajectory_target_penetration_under_2mm",
                )
            }
            grasp["passed"] = bool(all(grasp["checks"].values()))

        airborne = trace[first:release]
        tip_forces, external, floor_allowed, separated = [], [], [], []
        for row in airborne:
            tip_forces.append([math.fsum(c["normal_force_n"] for c in row["contacts"]
                                        if c["other_body"] == tip) for tip in TIPS])
            floor = row["tray_floor_normal_force_n"]
            lowering = row["phase"] == "transport_lower"
            other = math.fsum(c["normal_force_n"] for c in row["contacts"] if c["other_body"] not in TIPS)
            external.append(max(row["external_support_force_n"], other - (floor if lowering else 0.)))
            floor_allowed.append(lowering or floor < .1)
            separated.append(row["lowest_collider_clearance_m"] > 0 or (lowering and floor >= .1))
        # Relative drift is useful for tuning, but is not equivalent to dropping.
        carry = trace[lift_end - 1:release]
        positions = np.array([row["relative_position"] for row in carry])
        rotations = np.array([row["relative_rotation"] for row in carry])
        drift_angles = [float(np.arccos(np.clip((np.trace(r @ rotations[0].T) - 1) / 2, -1, 1)))
                        for r in rotations]
        peak = max((max(0., -c["distance_m"]) for row in airborne for c in row["contacts"]), default=0.)
        metrics = {
            "first_liftoff_time_s": times[first], "release_time_s": times[release],
            "liftoff_clearance_m": .002, "retention_samples": len(airborne),
            "min_tip_normal_force_n": dict(zip(TIPS, np.min(tip_forces, axis=0).tolist(), strict=True)),
            "max_external_support_force_n": max(external),
            "max_sample_gap_s": float(np.diff(times[first:release + 1]).max()),
            "max_relative_translation_m": float(np.linalg.norm(positions - positions[0], axis=1).max()),
            "max_relative_rotation_rad": max(drift_angles), "max_target_penetration_m": peak,
        }
        checks = {
            "sampling_gap_at_most_50ms": metrics["max_sample_gap_s"] <= .05 + 1e-9,
            "bilateral_contact_until_release": min(metrics["min_tip_normal_force_n"].values()) >= .1,
            "no_external_support": max(external) < .1,
            "tray_floor_support_only_during_lowering": all(floor_allowed),
            "separated_until_tray_touchdown": all(separated),
        }
        transport = {"passed": bool(all(checks.values())), "checks": checks, "metrics": metrics,
                     "diagnostic_checks": {
                         "relative_translation_under_5mm": metrics["max_relative_translation_m"] <= .005,
                         "relative_rotation_under_5deg": metrics["max_relative_rotation_rad"] <= math.radians(5),
                         "target_penetration_under_2mm": peak <= .002,
                     }}
        if not contact_diagnostics:
            transport["checks"].update(transport.pop("diagnostic_checks"))
            transport["passed"] = bool(all(transport["checks"].values()))
        placed = evaluate_placement(trace, minimum_hold_s=1., edge_clearance_m=edge_clearance_m)
        checks = {"grasped_clear_of_table": grasp["passed"], "retained_until_release": transport["passed"],
                  "placed_in_tray": placed["passed"]}
        return {**base, "passed": bool(all(checks.values())), "checks": checks,
                "grasp_quality": grasp, "transport": transport, "placement": placed}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {**base, "passed": False, "checks": {}, "reason": f"Invalid transfer evidence: {error}"}
