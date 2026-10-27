"""Plan tray transfer in an owned, offline MuJoCo model.

No dynamics, simulator connection, actuator command, or training-data recording
occurs here. The target is moved only in this private collision-check model to
predict the swept volume of a maintained grasp; the real object is never welded.
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import body_bounds, candidate_placement, primitive_bounds

END = "unitree_g1_1_arm_r_end_link"
TIPS = {f"unitree_g1_1_gripper_r_{side}_link4" for side in ("inner", "outer")}


def load_owned_model(source_xml: str | Path) -> mujoco.MjModel:
    """Resolve exported assets without changing any physical model property."""
    source = Path(source_xml).resolve(strict=True)
    root = ET.parse(source).getroot()
    if root.findall(".//include"):
        raise ValueError("A standalone scene XML export is required")
    compiler = root.find("compiler")
    for kind, attribute in (("mesh", "meshdir"), ("texture", "texturedir")):
        directory = source.parent
        if compiler is not None and compiler.get(attribute):
            directory /= compiler.get(attribute)
        for item in root.findall(f"./asset/{kind}[@file]"):
            filename = Path(item.get("file"))
            options = (directory / filename, source.parent / "assets" / filename)
            found = next((path.resolve() for path in options if path.is_file()), None)
            if found is None:
                raise FileNotFoundError(filename)
            item.set("file", str(found))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def body_primitives(model: mujoco.MjModel, body: int) -> list[dict]:
    return [{"geom_type": int(model.geom_type[g]), "geom_size": model.geom_size[g],
             "geom_pos": model.geom_pos[g], "geom_quat": model.geom_quat[g],
             "geom_contype": int(model.geom_contype[g]),
             "geom_conaffinity": int(model.geom_conaffinity[g])}
            for g in range(model.ngeom) if model.geom_bodyid[g] == body
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def measure_tray(model: mujoco.MjModel, data: mujoco.MjData, body_name: str | None = None) -> dict:
    """Infer the conservative rectangular cavity of this slatted tray."""
    if body_name is None:
        names = [model.body(i).name for i in range(model.nbody)
                 if model.body(i).name.endswith("basket_01")]
        if len(names) != 1:
            raise ValueError("Specify one tray body_name; basket_01 is ambiguous or missing")
        body_name = names[0]
    body = model.body(body_name).id
    rotation = data.xmat[body].reshape(3, 3)
    if not np.allclose(rotation[:, 2], [0, 0, 1], atol=.01):
        raise ValueError("Planning currently requires an approximately level tray")
    geoms = body_primitives(model, body)
    boxes = [g for g in geoms if g["geom_type"] == int(mujoco.mjtGeom.mjGEOM_BOX)]
    floors = [g for g in boxes if min(g["geom_size"][:2]) > .1
              and g["geom_size"][2] < .02]
    if len(floors) != 1:
        raise ValueError("Expected one broad, thin tray-floor collider")
    floor = floors[0]
    floor_lo, floor_hi = primitive_bounds(floor)
    lower, upper = floor_lo[:2].copy(), floor_hi[:2].copy()
    bounds = [primitive_bounds(g) for g in geoms if g is not floor]
    for lo, hi in bounds:
        if hi[2] <= floor_hi[2] + .02:
            continue
        center = (lo + hi) / 2
        for axis in range(2):
            if abs(center[axis]) < .75 * max(abs(floor_lo[axis]), abs(floor_hi[axis])):
                continue
            if center[axis] < 0:
                lower[axis] = max(lower[axis], hi[axis])
            else:
                upper[axis] = min(upper[axis], lo[axis])
    rim = max(hi[2] for _, hi in bounds)
    position = data.xpos[body].copy()
    return {"body_name": body_name, "floor_size_xyz_m": (2 * floor["geom_size"]).tolist(),
            "inner_xy_min": lower.tolist(), "inner_xy_max": upper.tolist(),
            "floor_top_z": float(floor_hi[2]), "rim_top_z": float(rim),
            "position_world": position.tolist(), "rotation_world": rotation.tolist(),
            "floor_top_world_z": float(position[2] + floor_hi[2]),
            "rim_top_world_z": float(position[2] + rim), "edge_clearance_m": .02,
            "release_clearance_m": .005}


class TransportPlanner:
    """Kinematic IK and carried-object collision checks, with no stepping."""

    def __init__(self, source_xml: str | Path, full_qpos, plan: dict):
        self.model = load_owned_model(source_xml)
        self.data = mujoco.MjData(self.model)
        self.base = np.asarray(full_qpos, dtype=float).copy()
        if self.base.shape != (self.model.nq,) or not np.isfinite(self.base).all():
            raise ValueError("full_qpos must be a finite snapshot matching this scene")
        self.data.qpos[:] = self.base
        mujoco.mj_forward(self.model, self.data)
        self.target_name = plan["target_body"]
        self.target = self.model.body(self.target_name).id
        self.end = self.model.body(END).id
        self.joint_names = plan["arm_joint_names"]
        joints = [self.model.joint(name).id for name in self.joint_names]
        self.qaddr = self.model.jnt_qposadr[joints]
        self.low = self.model.jnt_range[joints, 0] + .02
        self.high = self.model.jnt_range[joints, 1] - .02
        target_joints = range(self.model.body_jntadr[self.target],
                              self.model.body_jntadr[self.target] + self.model.body_jntnum[self.target])
        free = [j for j in target_joints if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError("The carried target must have exactly one free joint")
        self.target_qaddr = int(self.model.jnt_qposadr[free[0]])
        self.start_q = self.base[self.qaddr].copy()
        self.start_end_pos = self.data.xpos[self.end].copy()
        self.end_rotation = self.data.xmat[self.end].reshape(3, 3).copy()
        self.relative_position = self.end_rotation.T @ (
            self.data.xpos[self.target] - self.start_end_pos)
        self.relative_rotation = self.end_rotation.T @ self.data.xmat[self.target].reshape(3, 3)
        self.object_rotation = self.end_rotation @ self.relative_rotation
        self.object_geoms = body_primitives(self.model, self.target)
        self.tray = measure_tray(self.model, self.data, plan.get("placement", {}).get("body_name"))
        clearance = float(plan.get("placement", {}).get("release_clearance_m",
                                                        self.tray["release_clearance_m"]))
        if not np.isfinite(clearance) or not 0. <= clearance <= .02:
            raise ValueError("Release clearance must be finite and between 0 and 0.02m")
        self.tray["release_clearance_m"] = clearance
        self.arm_bodies = set()
        root_body = int(self.model.jnt_bodyid[joints[0]])
        for body in range(self.model.nbody):
            ancestor = body
            while ancestor:
                if ancestor == root_body:
                    self.arm_bodies.add(body)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        self.checks = 0
        self.max_position_error = 0.
        self.max_rotation_error = 0.
        self.max_joint_increment = 0.

    def fk(self, q):
        self.data.qpos[:] = self.base
        self.data.qpos[self.qaddr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def attach_for_static_check(self):
        rotation = self.data.xmat[self.end].reshape(3, 3)
        pos = self.data.xpos[self.end] + rotation @ self.relative_position
        quat_xyzw = Rotation.from_matrix(rotation @ self.relative_rotation).as_quat()
        address = self.target_qaddr
        self.data.qpos[address:address + 3] = pos
        self.data.qpos[address + 3:address + 7] = quat_xyzw[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)

    def collisions(self, q, *, carried=True, target_fixed=None):
        self.fk(q)
        if carried:
            self.attach_for_static_check()
        else:
            if target_fixed is not None:
                self.data.qpos[self.target_qaddr:self.target_qaddr + 7] = target_fixed
            mujoco.mj_forward(self.model, self.data)
        bad = []
        for contact in self.data.contact[:self.data.ncon]:
            ids = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            names = [self.model.body(body).name for body in ids]
            if not (any(body in self.arm_bodies for body in ids) or self.target in ids):
                continue
            if all("gripper_r_" in name for name in names):
                continue
            if self.target in ids and any(name in TIPS for name in names):
                continue
            if contact.dist < -.0001:
                bad.append({"bodies": names, "distance_m": float(contact.dist)})
        self.checks += 1
        return bad

    def solve(self, position, seed, rotation=None):
        position = np.asarray(position)
        rotation = self.end_rotation if rotation is None else np.asarray(rotation)

        def residual(q):
            self.fk(q)
            rot_error = Rotation.from_matrix(
                self.data.xmat[self.end].reshape(3, 3) @ rotation.T).as_rotvec()
            return np.r_[self.data.xpos[self.end] - position, .2 * rot_error, .002 * (q - seed)]

        fit = least_squares(residual, np.clip(seed, self.low, self.high),
                            bounds=(self.low, self.high), max_nfev=150,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        self.fk(fit.x)
        pe = float(np.linalg.norm(self.data.xpos[self.end] - position))
        re = float(np.linalg.norm(Rotation.from_matrix(
            self.data.xmat[self.end].reshape(3, 3) @ rotation.T).as_rotvec()))
        self.max_position_error = max(self.max_position_error, pe)
        self.max_rotation_error = max(self.max_rotation_error, re)
        if pe > .001 or re > .01:
            raise ValueError(f"IK target unreachable: position error {pe:.5f}m, rotation {re:.5f}rad")
        return fit.x

    def segment(self, name, start_q, start_position, end_position, duration_s, *, carried=True,
                target_fixed=None, start_rotation=None, end_rotation=None):
        distance = np.linalg.norm(np.asarray(end_position) - start_position)
        start_rotation = self.end_rotation if start_rotation is None else np.asarray(start_rotation)
        end_rotation = start_rotation if end_rotation is None else np.asarray(end_rotation)
        turn = Rotation.from_matrix(end_rotation @ start_rotation.T).as_rotvec()
        count = max(21, int(np.ceil(distance / .004)) + 1,
                    int(np.ceil(np.linalg.norm(turn) / .005)) + 1)
        q = np.asarray(start_q).copy()
        points = [q.tolist()]
        for fraction in np.linspace(0, 1, count)[1:]:
            position = (1 - fraction) * start_position + fraction * np.asarray(end_position)
            rotation = Rotation.from_rotvec(fraction * turn).as_matrix() @ start_rotation
            next_q = self.solve(position, q, rotation)
            jump = float(np.max(np.abs(next_q - q)))
            self.max_joint_increment = max(self.max_joint_increment, jump)
            if jump > .15:
                raise ValueError(f"Discontinuous IK in {name}: {jump:.3f}rad")
            # Inspect interpolation, not only IK endpoints.
            for weight in (0.25, .5, .75, 1.):
                bad = self.collisions((1 - weight) * q + weight * next_q,
                                      carried=carried, target_fixed=target_fixed)
                if bad:
                    raise ValueError(f"Collision in {name} at {fraction:.3f}: {bad[0]}")
            points.append(next_q.tolist())
            q = next_q
        return {"name": name, "joints": "arm", "duration_s": duration_s, "q": points}, q


def plan_transport_at_rotation(source_xml: str | Path, full_qpos, plan: dict,
                               wrist_rotation_world_deg) -> dict:
    """Try one explicit wrist attitude with a smooth, collision-checked transition."""
    planner = TransportPlanner(source_xml, full_qpos, plan)
    initial_rotation = planner.end_rotation.copy()
    turn = Rotation.from_euler("xyz", wrist_rotation_world_deg, degrees=True)
    planner.end_rotation = turn.as_matrix() @ initial_rotation
    planner.object_rotation = planner.end_rotation @ planner.relative_rotation
    tray = planner.tray
    tray_pos = np.asarray(tray["position_world"])
    tray_rot = np.asarray(tray["rotation_world"])
    object_rot_in_tray = tray_rot.T @ planner.object_rotation
    lo, hi = body_bounds(planner.object_geoms, object_rot_in_tray)
    half = (hi[:2] - lo[:2]) / 2
    lower = np.asarray(tray["inner_xy_min"]) + half + .025
    upper = np.asarray(tray["inner_xy_max"]) - half - .025
    if np.any(lower >= upper):
        raise ValueError("Object does not fit inside the tray with 25mm edge margin")
    current = tray_rot.T @ (planner.data.xpos[planner.target] - tray_pos)
    closest = np.clip(current[:2] + (lo[:2] + hi[:2]) / 2, lower, upper)
    grid = [closest] + [np.array([x, y]) for x in np.linspace(lower[0], upper[0], 5)
                       for y in np.linspace(lower[1], upper[1], 5)]
    grid = sorted(grid, key=lambda xy: np.linalg.norm(xy - closest))
    # Turning the held object also moves its origin about the wrist. Include the
    # whole turn when choosing clearance, not just the final object's bounds.
    turning_low_z = []
    for fraction in np.linspace(0, 1, 21):
        rotation = Rotation.from_rotvec(fraction * turn.as_rotvec()).as_matrix() @ initial_rotation
        object_position = planner.start_end_pos + rotation @ planner.relative_position
        turning_low_z.append(body_bounds(planner.object_geoms,
                                         rotation @ planner.relative_rotation,
                                         object_position)[0][2])
    raise_height = max(.04, tray["rim_top_world_z"] + .035 - min(turning_low_z))
    high_position = planner.start_end_pos + [0, 0, raise_height]
    raise_phase, high_q = planner.segment("transport_raise", planner.start_q,
                                           planner.start_end_pos, high_position, 4.,
                                           start_rotation=initial_rotation)
    if np.linalg.norm(turn.as_rotvec()) > 1e-10:
        orient, high_q = planner.segment("transport_raise", high_q, high_position,
                                         high_position, 2., start_rotation=initial_rotation,
                                         end_rotation=planner.end_rotation)
        raise_phase["q"].extend(orient["q"][1:])
        raise_phase["duration_s"] += orient["duration_s"]
    raised_statistics = (planner.max_position_error, planner.max_rotation_error,
                         planner.max_joint_increment)
    failures = []
    for center in grid:
        # Failed alternatives are reported separately; accepted-path accuracy
        # must not contain the residual of a rejected, unreachable endpoint.
        (planner.max_position_error, planner.max_rotation_error,
         planner.max_joint_increment) = raised_statistics
        proposal = candidate_placement(planner.object_geoms, object_rot_in_tray, center,
                                       tray["floor_top_z"], tray["release_clearance_m"])
        object_position = tray_pos + tray_rot @ proposal["body_origin_tray"]
        release_position = object_position - planner.end_rotation @ planner.relative_position
        over_position = np.r_[release_position[:2], high_position[2]]
        try:
            over, over_q = planner.segment("transport_over_tray", high_q, high_position,
                                          over_position, 7.)
            lower_phase, release_q = planner.segment("transport_lower", over_q, over_position,
                                                     release_position, 6.)
            planner.collisions(release_q)
            fixed = planner.data.qpos[planner.target_qaddr:planner.target_qaddr + 7].copy()
            retreat_position = release_position + [0, 0, .13]
            retreat, retreat_q = planner.segment("retreat", release_q, release_position,
                                                 retreat_position, 4., carried=False,
                                                 target_fixed=fixed)
        except ValueError as exc:
            failures.append({"center_xy": center.tolist(), "reason": str(exc)})
            continue
        drive = float(plan.get("transfer", {}).get("gripper_joint2_drive", 1.35))
        held = [0., 0., .28, drive, 0., 0., -.28, drive]
        release = {"name": "release", "joints": "gripper", "duration_s": 2.,
                   "servo_targets": [held, [0., 0., 0., -1., 0., 0., 0., -1.]]}
        settle = {"name": "settle_placed", "joints": "arm", "duration_s": 3.,
                  "q": [retreat_q.tolist(), retreat_q.tolist()]}
        report = {"planning_only": True, "dynamics_verified": False,
                  "basis": "Measured post-lift full qpos; constant measured object/end transform",
                  "static_object_pose_changes_only": True, "object_weld": False,
                  "target_body": planner.target_name, "tray": tray,
                  "target_center_xy_in_tray": center.tolist(),
                  "release_object_position_world": object_position.tolist(),
                  "release_object_bounds_in_tray": proposal,
                  "end_waypoints_world": {"raise": high_position.tolist(),
                                           "over_tray": over_position.tolist(),
                                           "release": release_position.tolist(),
                                           "retreat": retreat_position.tolist()},
                  "relative_position_in_end": planner.relative_position.tolist(),
                  "relative_rotation_in_end": planner.relative_rotation.tolist(),
                  "wrist_rotation_world_deg": list(wrist_rotation_world_deg),
                  "wrist_rotation_transition": "Raise with measured attitude, then smoothly turn before crossing rim",
                  "collision_samples": planner.checks,
                  "max_ik_position_error_m": planner.max_position_error,
                  "max_ik_rotation_error_rad": planner.max_rotation_error,
                  "max_joint_increment_rad": planner.max_joint_increment,
                  "rejected_candidates": failures,
                  "limitations": ["Opening servo dynamics and grasp retention are not evaluated",
                                  "Retreat check uses held finger pose and a fixed released-object prediction",
                                  "Collision sampling cannot prove continuous-time clearance"]}
        return {"phases": [raise_phase, over, lower_phase, release, retreat, settle],
                "placement": {key: tray[key] for key in ("body_name", "floor_size_xyz_m",
                              "inner_xy_min", "inner_xy_max", "edge_clearance_m", "release_clearance_m")},
                "report": report}
    raise ValueError(f"No sampled tray placement is reachable/collision-free: {failures[:3]}")


def plan_transport(source_xml: str | Path, full_qpos, plan: dict) -> dict:
    """Build continuous transfer phases from the measured post-lift state.

    The near-side placement in this scene requires a small pitch correction to
    avoid the measured grasp attitude's wrist workspace boundary. These are
    explicit target attitudes, not relaxed IK accuracy or discontinuous jumps.
    """
    candidates = plan.get("transfer", {}).get("wrist_rotation_candidates_world_deg",
                                               [[0., -5., 0.], [0., 0., 0.], [0., 5., 0.]])
    if not candidates:
        raise ValueError("At least one wrist attitude candidate is required")
    failures = []
    for angles in candidates:
        values = np.asarray(angles, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Wrist attitude offsets must be finite xyz angles in degrees")
        try:
            transport = plan_transport_at_rotation(source_xml, full_qpos, plan, values.tolist())
        except ValueError as exc:
            failures.append({"wrist_rotation_world_deg": values.tolist(), "reason": str(exc)})
            continue
        transport["report"]["rejected_wrist_attitudes"] = failures
        return transport
    raise ValueError(f"No continuous tray transport is reachable/collision-free: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True,
                        help="Previously recorded controls.npz containing qpos_after")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.plan.read_text())
    plan = copy.deepcopy(original)
    for phase in plan["phases"]:
        if phase["name"] == "tighten":
            phase["servo_targets"][-1][3] = 1.35
            phase["servo_targets"][-1][7] = 1.35
        elif phase["name"] == "lift_hold":
            phase["duration_s"] = 2.
    state = np.load(args.state, allow_pickle=False)["qpos_after"][-1]
    transport = plan_transport(plan["source_xml"], state, plan)
    plan["phases"] = plan["phases"][:next(i for i, p in enumerate(plan["phases"])
                                         if p["name"] == "lift_hold") + 1]
    plan["reference_transport"] = transport["phases"]
    plan["placement"] = transport["placement"]
    plan["transfer"] = {"replan_after_phase": "lift_hold", "gripper_joint2_drive": 1.35,
                        "planning_reference_state": str(args.state.resolve()),
                        "reference_drive_was": 1.2, "training_data": False}
    plan["purpose"] = "Stronger retained grasp followed by measured-state tray transport; development preview"
    plan["dynamic_validation_design"].update({
        "task": "tray_transfer", "final_joint2_target": 1.35, "lift_hold_duration_s": 2.,
        "standalone_force_validation": False,
        "success_criterion": "retain_during_transport_then_release_inside_tray_and_settle",
        "reason": ("User requested additional grip drive and direct tray-transfer development "
                   "without another standalone hold test."),
    })
    transport["report"].update({
        "reference_snapshot_joint2_drive": 1.2,
        "requested_execution_joint2_drive": 1.35,
        "reference_is_not_physics_evidence_for_requested_drive": True,
        "runtime_replanning_required": True,
    })
    plan["transport_static_report"] = transport["report"]
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **transport["report"]}, indent=2))


if __name__ == "__main__":
    main()
