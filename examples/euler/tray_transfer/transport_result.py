"""Offline acceptance of this plan's suspended transport before release."""

import math
from numbers import Real

import numpy as np

SCOPE = (
    "Sampled suspended transport from the last lift_hold through the first release boundary; "
    "requires release above the floor. Not suitable for touching down before opening, "
    "and not proof of behavior between samples or final placement."
)


def _failure(reason):
    return {"passed": False, "checks": {}, "metrics": {}, "reason": reason, "scope": SCOPE}


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Expected a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative number")
    return value


def _array(value, shape):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"Expected finite numeric array of shape {shape}")
    return array.astype(float)


def evaluate_transport(trace: list[dict], tip_names, expected_phases) -> dict:
    """Require ordered transport phases and continuous bilateral grasp until release.

    ``relative_position`` and ``relative_rotation`` describe the object in the
    gripper-end frame. Contacts contain ``other_body`` and ``normal_force_n``.
    The first release sample closes the timing interval but its contact state
    is deliberately excluded: opening is allowed at that point.
    """
    try:
        for names in (tip_names, expected_phases):
            if (not isinstance(names, (list, tuple)) or not names
                    or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("Tips and expected phases must be nonempty distinct names")
        if len(tip_names) != 2 or any(phase in ("lift_hold", "release") for phase in expected_phases):
            raise ValueError("Expected two tips and transport phases excluding lift_hold/release")
        if not isinstance(trace, list) or not trace:
            raise ValueError("Missing transport trace")
        times, phases = [], []
        for row in trace:
            if not isinstance(row, dict) or not isinstance(row["phase"], str) or not row["phase"]:
                raise ValueError("Every sample needs a phase")
            times.append(_number(row["time_s"]))
            phases.append(row["phase"])
            if len(times) > 1 and times[-1] <= times[-2]:
                raise ValueError("Trace times must strictly increase")
        release = phases.index("release")
        baseline = max(i for i, phase in enumerate(phases) if phase == "lift_hold")
        if baseline >= release:
            raise ValueError("The last lift_hold must precede the first release")
        observed = []
        for phase in phases[baseline + 1:release]:
            if not observed or phase != observed[-1]:
                observed.append(phase)

        positions, rotations, tip_forces, support_forces = [], [], [], []
        for row in trace[baseline:release]:
            positions.append(_array(row["relative_position"], (3,)))
            rotation = _array(row["relative_rotation"], (3, 3))
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
                raise ValueError("Relative rotation must be a proper rotation matrix")
            rotations.append(rotation)
            contacts = row["contacts"]
            if not isinstance(contacts, list):
                raise ValueError("Contacts must be a list")
            by_tip = {tip: [] for tip in tip_names}
            external = []
            for contact in contacts:
                if (not isinstance(contact, dict) or not isinstance(contact["other_body"], str)
                        or not contact["other_body"]):
                    raise ValueError("Every contact needs an other_body name")
                force = _number(contact["normal_force_n"])
                if contact["other_body"] in by_tip:
                    by_tip[contact["other_body"]].append(force)
                else:
                    external.append(force)
            tip_forces.append([math.fsum(by_tip[tip]) for tip in tip_names])
            support_forces.append(math.fsum(external))
        translation = np.linalg.norm(np.asarray(positions) - positions[0], axis=1)
        rotation_drift = [float(np.arccos(np.clip((np.trace(r @ rotations[0].T) - 1.) / 2., -1., 1.)))
                          for r in rotations]
        metrics = {
            "baseline_time_s": times[baseline], "release_time_s": times[release],
            "transport_samples": release - baseline - 1, "observed_phases": observed,
            "max_sample_gap_s": float(np.diff(times[baseline:release + 1]).max()),
            "min_tip_normal_force_n": dict(zip(tip_names, np.min(tip_forces, axis=0).tolist(), strict=True)),
            "max_external_support_force_n": max(support_forces),
            "max_relative_translation_m": float(translation.max()),
            "max_relative_rotation_rad": max(rotation_drift),
        }
        if not all(math.isfinite(metrics[key]) for key in (
            "max_sample_gap_s", "max_relative_translation_m", "max_relative_rotation_rad",
        )):
            raise ValueError("Non-finite derived transport measurement")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _failure(f"Invalid transport evidence: {error}")

    checks = {
        "expected_phase_order": observed == list(expected_phases),
        "sampling_gap_at_most_50ms": metrics["max_sample_gap_s"] <= .05 + 1e-9,
        "bilateral_contact_all_samples": min(metrics["min_tip_normal_force_n"].values()) >= .5,
        "no_external_support": metrics["max_external_support_force_n"] < .1,
        "relative_translation_at_most_5mm": metrics["max_relative_translation_m"] <= .005 + 1e-9,
        "relative_rotation_at_most_5deg": metrics["max_relative_rotation_rad"] <= math.radians(5.) + 1e-9,
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "metrics": metrics, "scope": SCOPE,
            "reason": "Suspended transport passed" if passed else "Failed: " + ", ".join(
                name for name, okay in checks.items() if not okay
            )}
