"""Reproducible offline blue-part grasp test, with explicit model experiments.

This is a fixed-layout physics diagnostic, not a dataset collector. No network
connection, simulator takeover, object weld, or object pose edits are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .blue_env import BluePickEnv
from .geometry import body_bounds, quat_matrix
from .gripper_control import right_gripper_command

HERE = Path(__file__).resolve().parent
BLUE = "[17143530386317]_ScrewHole"
END = "unitree_g1_1_arm_r_end_link"
TIPS = tuple(f"unitree_g1_1_gripper_r_{side}_link4" for side in ("inner", "outer"))
EXCLUSIONS = tuple(
    (f"unitree_g1_1_gripper_r_{side}_link{a}", f"unitree_g1_1_gripper_r_{side}_link{b}")
    for side in ("inner", "outer") for a, b in ((1, 2), (4, 2))
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_model(source: Path, destination: Path, profile: str, *, fix_waist: bool = False) -> dict:
    """Write an independent XML copy and describe every physics modification."""
    if profile not in ("original", "internal-exclusions", "experimental-contact"):
        raise ValueError(f"Unknown model profile: {profile}")
    source = source.resolve(strict=True)
    if source == destination.resolve():
        raise ValueError("The source model must not be overwritten")
    tree = ET.parse(source)
    root = tree.getroot()
    if root.findall(".//include"):
        raise ValueError("A standalone scene export is required")
    compiler = root.find("compiler")
    asset_hashes = {}
    for kind, directory in (("mesh", "meshdir"), ("texture", "texturedir")):
        asset_dir = source.parent
        if compiler is not None and compiler.get(directory):
            asset_dir /= compiler.get(directory)
        for node in root.findall(f"./asset/{kind}[@file]"):
            file = Path(node.get("file"))
            asset = next((p.resolve() for p in (
                asset_dir / file, source.parent / "assets" / file,
            ) if p.is_file()), None)
            if asset is None:
                raise FileNotFoundError(file)
            asset_hashes[f"{kind}:{node.get('name')}"] = sha256(asset)
            node.set("file", str(asset))
    changes = []
    if fix_waist:
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        joints = {joint.get("name") for joint in root.findall(".//body/joint")}
        for axis in ("yaw", "roll", "pitch"):
            name = f"unitree_g1_1_waist_{axis}_joint"
            if name not in joints:
                raise ValueError(f"Waist fixture joint missing: {name}")
            attrs = {"name": f"blue_test_fixed_waist_{axis}", "joint1": name,
                     "polycoef": "0 0 0 0 0", "solref": "0.002 1"}
            ET.SubElement(equality, "joint", **attrs)
            changes.append({"test_fixture_joint_equality": attrs})
    bodies = {body.get("name"): body for body in root.iter("body")}
    if profile != "original":
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        existing = {frozenset((n.get("body1"), n.get("body2"))) for n in contact.findall("exclude")}
        for a, b in EXCLUSIONS:
            if a not in bodies or b not in bodies:
                raise ValueError(f"Calibrated gripper body missing: {a}, {b}")
            if frozenset((a, b)) not in existing:
                ET.SubElement(contact, "exclude", body1=a, body2=b)
                changes.append({"exclude_body_pair": [a, b]})
    if profile == "experimental-contact":
        for tip in TIPS:
            geoms = bodies[tip].findall("geom")
            if len(geoms) != 1:
                raise ValueError(f"Expected one calibrated mesh on {tip}")
            geom = geoms[0]
            edits = {"condim": "4", "friction": "2 0.01 0.1", "solref": "0.005 1", "priority": "1"}
            changes.append({"body": tip, "before": {k: geom.get(k) for k in edits}, "after": edits})
            geom.attrib.update(edits)
    tree.write(destination, encoding="utf-8")
    return {"profile": profile, "source_xml": str(source), "source_sha256": sha256(source),
            "derived_sha256": sha256(destination), "asset_sha256": asset_hashes, "changes": changes}


def validate_layout(env: BluePickEnv, plan: dict) -> None:
    """Reject stale joint paths if a user moves/replaces scene objects."""
    for name, expected in plan["initial_body_poses"].items():
        pos, quat = env.read_pose(name)
        angle = np.linalg.norm(Rotation.from_matrix(
            quat_matrix(quat) @ quat_matrix(expected["quaternion_wxyz"]).T,
        ).as_rotvec())
        if np.linalg.norm(pos - expected["position"]) > .001 or angle > .005:
            raise ValueError(f"Scene changed at {name}; regenerate the IK path before running")
    if not np.allclose(env.control_from_qpos(), plan["initial_actuated_qpos"], atol=.001, rtol=0):
        raise ValueError("Robot initial joints changed; this diagnostic path is stale")


def collision_primitives(env: BluePickEnv) -> list[dict]:
    return [{"geom_type": g["Type"], "geom_size": g["Size"], "geom_pos": g["Pos"],
             "geom_quat": g["Quat"], "geom_contype": g["Contype"],
             "geom_conaffinity": g["Conaffinity"]}
            for g in env.model.get_geom_dict().values() if g["BodyName"] == BLUE]


def sample(env: BluePickEnv, phase: str, geoms: dict, blue_geoms: list, floor: float) -> dict:
    pos, quat = env.read_pose(BLUE)
    end_pos, end_quat = env.read_pose(END)
    rotation = quat_matrix(end_quat)
    contacts = env.query_contact_simple()
    relevant = [i for i, c in enumerate(contacts) if BLUE in (geoms[c["geom1"]], geoms[c["geom2"]])]
    forces = env.query_contact_force(relevant)
    readings = []
    for i in relevant:
        c = contacts[i]
        if not np.isfinite([forces[i][0], c["dist"]]).all():
            raise ValueError("Non-finite target contact measurement")
        bodies = (geoms[c["geom1"]], geoms[c["geom2"]])
        readings.append({"other_body": bodies[1] if bodies[0] == BLUE else bodies[0],
                         "normal_force_n": float(forces[i][0]), "distance_m": float(c["dist"])})
    lower, _ = body_bounds(blue_geoms, quat_matrix(quat), pos)
    return {
        "phase": phase, "time_s": float(env.data.time),
        "blue_position": pos.tolist(), "blue_quaternion_wxyz": quat.tolist(),
        "end_position": end_pos.tolist(), "end_quaternion_wxyz": end_quat.tolist(),
        "relative_position": (rotation.T @ (pos - end_pos)).tolist(),
        "relative_rotation": (rotation.T @ quat_matrix(quat)).tolist(),
        "lowest_collider_clearance_m": float(lower[2] - floor), "contacts": readings,
        "control_saturated": bool(env.last_control_saturated.any()),
        "saturated_actuator_names": [env.actuator_joint_names[i]
                                     for i in np.flatnonzero(env.last_control_saturated)],
        "force_saturated": bool(env.last_force_saturated.any()),
    }


def evaluate_hold(trace: list[dict], initial_position: list) -> dict:
    """Check the final hold and reject deep target contact anywhere in the trace."""
    hold = [r for r in trace if r["phase"] == "lift_hold"]
    if not hold:
        return {"passed": False, "reason": "No lift hold was executed"}
    window = [r for r in hold if r["time_s"] >= hold[-1]["time_s"] - 1.0 - 1e-8]
    if len(window) < 2 or window[-1]["time_s"] - window[0]["time_s"] < .99:
        return {"passed": False, "reason": "Less than one second of hold evidence"}
    grip = np.array([r["relative_position"] for r in window])
    rotations = [np.array(r["relative_rotation"]) for r in window]
    rotation_drift = max(np.linalg.norm(Rotation.from_matrix(r @ rotations[0].T).as_rotvec()) for r in rotations)
    bilateral = all(all(sum(c["normal_force_n"] for c in r["contacts"] if c["other_body"] == tip) >= .5
                        for tip in TIPS) for r in window)
    external_support = [sum(max(0., c["normal_force_n"]) for c in r["contacts"]
                            if c["other_body"] not in TIPS) for r in window]
    unsupported = max(external_support) < .1
    peak = max(
        ((max(0., -c["distance_m"]), r["phase"], r["time_s"])
         for r in trace for c in r["contacts"]),
        key=lambda item: item[0], default=(0., None, None),
    )
    trajectory_metrics = {
        "max_target_penetration_m": peak[0],
        "peak_penetration_phase": peak[1], "peak_penetration_time_s": peak[2],
    }
    metrics = {
        "hold_samples": len(window), "hold_span_s": window[-1]["time_s"] - window[0]["time_s"],
        "min_origin_lift_m": min(r["blue_position"][2] - initial_position[2] for r in window),
        "min_collider_clearance_m": min(r["lowest_collider_clearance_m"] for r in window),
        "relative_position_drift_m": float(np.max(np.linalg.norm(grip - grip[0], axis=1))),
        "relative_rotation_drift_rad": float(rotation_drift),
        "max_target_penetration_m": max([max(0., -c["distance_m"]) for r in window for c in r["contacts"]] or [0.]),
        "bilateral_contact_all_samples": bilateral, "no_other_support_all_samples": unsupported,
        "max_total_external_support_n": max(external_support),
    }
    checks = {
        "origin_lift_at_least_50mm": metrics["min_origin_lift_m"] >= .05,
        "all_colliders_at_least_20mm_above_initial_support": metrics["min_collider_clearance_m"] >= .02,
        "relative_translation_under_5mm": metrics["relative_position_drift_m"] <= .005,
        "relative_rotation_under_5deg": rotation_drift <= np.deg2rad(5),
        "penetration_under_2mm": metrics["max_target_penetration_m"] <= .002,
        "trajectory_target_penetration_under_2mm": peak[0] <= .002,
        "bilateral_contact": bilateral, "no_environment_support": unsupported,
    }
    return {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "metrics": metrics,
            "trajectory_metrics": trajectory_metrics,
            "scope": ("Final one-second hold checks plus full-trace target penetration at 50 Hz; "
                      "not proof of every 500 Hz physics substep")}


def run(source: Path, profile: str, output: Path, plan_path: Path | None = None, *, fix_waist: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = None
    trace, states_before, states_after, commands, servos = [], [], [], [], []
    times_after = []
    try:
        plan_path = plan_path or HERE / "blue_pick_plan.json"
        plan = json.loads(plan_path.read_text())
        if plan.get("schema_version") != 2:
            raise ValueError("Expected a plan with explicit joint2 gripper drive semantics")
        manifest = prepare_model(source, output / "scene.xml", profile, fix_waist=fix_waist)
        if manifest["source_sha256"] != plan["source_sha256"] or manifest["asset_sha256"] != plan["asset_sha256"]:
            raise ValueError("Scene XML or asset contents changed; recalibrate this fixed-layout diagnostic first")
        manifest.update({"plan_sha256": sha256(plan_path), "python": platform.python_version(),
                         "mujoco": mujoco.__version__, "base_fixed": True,
                         "waist_fixed_test_fixture": fix_waist,
                         "gripper_controller": "joint2_reference",
                         "gripper_drive_phases": {p["name"]: p["drive"] for p in plan["phases"]
                                                  if p["joints"] == "gripper"},
                         "bias_compensation": True, "position_feedback_gain": 5.,
                         "physics_dt_s": .002, "control_dt_s": .02, "training_data": False})
        manifest["sampling"] = "mj_forward after each control step; poses, contacts and Jacobians at recorded qpos/time"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        env = BluePickEnv(str(output / "scene.xml"), bias_compensation=True, position_feedback_gain=5.)
        validate_layout(env, plan)
        env.fix_base()
        ids = {part: [env.actuator_joint_names.index(n) for n in plan[key]]
               for part, key in (("arm", "arm_joint_names"), ("gripper", "gripper_joint_names"))}
        if "aux_joint_names" in plan:
            ids["aux"] = [env.actuator_joint_names.index(n) for n in plan["aux_joint_names"]]
        geoms = {g["GeomId"]: g["BodyName"] for g in env.model.get_geom_dict().values()}
        blue_geoms = collision_primitives(env)
        placement = None
        if "placement" in plan:
            from .placement_sample import placement_geometry, sample_placement
            placement = placement_geometry(env, plan["placement"], BLUE)
            phase_names = [phase["name"] for phase in plan["phases"]]
            transport_phases = phase_names[phase_names.index("lift_hold") + 1:phase_names.index("release")]
            if not transport_phases:
                raise ValueError("Placement plan needs suspended transport phases before release")
        start_pos, start_quat = env.read_pose(BLUE)
        floor = body_bounds(blue_geoms, quat_matrix(start_quat), start_pos)[0][2]
        control = env.control_from_qpos()
        for phase in plan["phases"]:
            if phase["name"] == "release":
                env.right_gripper_hold.release()
            if phase["joints"] == "gripper":
                points = np.array([right_gripper_command(plan["gripper_joint_names"], drive)
                                   for drive in phase["drive"]])
            else:
                points = np.asarray(phase["q"])
            count = round(phase["duration_s"] / .02)
            for i in range(1, count + 1):
                fraction = i / count
                fraction = fraction**2 * (3 - 2 * fraction)
                coordinate = fraction * (len(points) - 1)
                index = min(int(coordinate), len(points) - 2)
                weight = coordinate - index
                control[ids[phase["joints"]]] = (1 - weight) * points[index] + weight * points[index + 1]
                state_before = env.data.qpos.ravel().copy()
                env.step(control)
                # mj_step advances qpos/time after computing cached body/contact data.
                # Refresh once before both samplers so the recorded row has one time.
                env.mj_forward()
                states_before.append(state_before)
                states_after.append(env.data.qpos.ravel().copy())
                commands.append(env.last_command.copy())
                servos.append(env.last_servo_control.copy())
                times_after.append(float(env.data.time))
                if not np.isfinite(states_after[-1]).all():
                    raise RuntimeError("Non-finite physics state")
                row = sample(env, phase["name"], geoms, blue_geoms, floor)
                if placement is not None:
                    row.update(sample_placement(env, placement))
                trace.append(row)
            if phase["name"] == "close":
                env.right_gripper_hold.latch(float(phase["drive"][-1]))
            if phase["name"] == "open":
                # Use the settled object, rather than the editor's hovering pose.
                start_pos, start_quat = env.read_pose(BLUE)
                floor = body_bounds(blue_geoms, quat_matrix(start_quat), start_pos)[0][2]
            print(f"{phase['name']}: time={row['time_s']:.2f}s, blue_z={row['blue_position'][2]:.4f}", flush=True)
            if placement is not None and phase["name"] == "lift_hold":
                held = evaluate_hold(trace, start_pos.tolist())
                required = ("origin_lift_at_least_50mm", "bilateral_contact", "no_environment_support")
                if not all(held.get("checks", {}).get(key, False) for key in required):
                    raise RuntimeError("Blue part is not held clear of the table; transfer was not executed")
        result = evaluate_hold(trace, start_pos.tolist())
        if placement is not None:
            from .placement_result import evaluate_placement
            from .transport_result import evaluate_transport
            carried = evaluate_transport(trace, TIPS, transport_phases)
            placed = evaluate_placement(trace, edge_clearance_m=placement["edge_clearance_m"])
            result = {"passed": result["passed"] and carried["passed"] and placed["passed"],
                      "grasp_quality": result, "transport": carried, "placement": placed}
        result.update({"profile": profile, "training_data": False,
                       "waist_fixed_test_fixture": fix_waist,
                       "gripper_controller": "joint2_reference",
                       "control_saturation_steps": sum(r["control_saturated"] for r in trace),
                       "force_saturation_steps": sum(r["force_saturated"] for r in trace)})
    except Exception as exc:
        result = {"passed": False, "error": str(exc), "profile": profile, "training_data": False,
                  "waist_fixed_test_fixture": fix_waist, "gripper_controller": "joint2_reference"}
        raise
    finally:
        try:
            (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            np.savez_compressed(output / "controls.npz", qpos_before=states_before, qpos_after=states_after,
                                nominal_action=commands, servo_last_substep=servos,
                                time_after=times_after)
            if "result" in locals():
                if states_after:
                    waist = [name for name in env.actuator_joint_names if "_waist_" in name]
                    indices = [env.joint_qpos_indices[name] for name in waist]
                    positions = np.asarray(states_after)[:, indices]
                    result["max_abs_waist_joint_rad"] = dict(zip(
                        waist, np.max(np.abs(positions), axis=0).tolist(), strict=True,
                    ))
                (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if env is not None:
                env.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, required=True, help="Standalone current scene export with meshes")
    parser.add_argument(
        "--profile", choices=("original", "internal-exclusions", "experimental-contact"), default="original",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path, help="Explicit scene-bound diagnostic plan; default is the blue pick only")
    parser.add_argument("--fix-waist", action="store_true",
                        help="Explicit offline fixture: constrain three waist joints at zero; not balance control")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or HERE / "outputs" / f"blue_{args.profile}_{stamp}"
    result = run(args.model_xml, args.profile, output, args.plan, fix_waist=args.fix_waist)
    print(json.dumps({"output": str(output.resolve()), **result}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
