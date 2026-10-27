"""Execute a scene-bound grasp/transfer plan, optionally recording synchronized demonstrations."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .blue_env import BluePickEnv
from .blue_pick import TIPS, evaluate_hold, prepare_model, sha256
from .geometry import body_bounds, quat_matrix
from .grasp_alignment import analyze_contacts
from .gripper_control import RIGHT_GRIPPER_JOINTS, right_gripper_command
from .scene_transfer_result import TRANSFER_PHASES

IMPLEMENTATION_FILES = (
    "scene_pick.py",
    "blue_env.py",
    "gripper_control.py",
    "blue_pick.py",
    "geometry.py",
    "grasp_alignment.py",
    "plan_tray_transfer.py",
    "scene_transfer_result.py",
    "placement_sample.py",
    "placement_result.py",
    "return_home.py",
    "episode_recording.py",
    "native_capture.py",
)
CONTROL_CONFIG = {
    "base_fixed": True,
    "waist_fixed": False,
    "gripper_hold_latched": True,
    "gripper_controller": "eight_servo_targets_latched_v2",
    "bias_compensation": True,
    "position_feedback_gain": 5.0,
    "physics_dt": 0.002,
    "frame_skip": 10,
    "control_dt": 0.02,
}


def penetration_requires_stop(contacts: list[dict], *, contact_diagnostics: bool) -> bool:
    """The 4mm heuristic is optional for intended fingertip contact, not a hardware limit."""
    return any(
        c["distance_m"] < -.004 and not (contact_diagnostics and c["other_body"] in TIPS)
        for c in contacts
    )


def evaluate_scene_pick(trace: list[dict], initial_position: list, *, contact_diagnostics: bool = False) -> dict:
    """User-selected retention check; contact overlap remains visible as a diagnostic."""
    result = evaluate_hold(trace, initial_position)
    result["success_criterion"] = "retention" if contact_diagnostics else "strict_contact_diagnostic"
    if not contact_diagnostics or "checks" not in result:
        return result
    result["diagnostic_checks"] = {
        key: result["checks"].pop(key)
        for key in ("penetration_under_2mm", "trajectory_target_penetration_under_2mm")
    }
    liftoff_clearance = .002
    first = next((i for i, r in enumerate(trace) if r["lowest_collider_clearance_m"] >= liftoff_clearance), None)
    airborne = trace[first:] if first is not None else []
    retained = bool(airborne) and all(
        r["lowest_collider_clearance_m"] > 0
        and all(sum(c["normal_force_n"] for c in r["contacts"] if c["other_body"] == tip) >= .1 for tip in TIPS)
        and sum(max(0., c["normal_force_n"]) for c in r["contacts"] if c["other_body"] not in TIPS) < .1
        for r in airborne
    )
    result["checks"]["retained_since_liftoff"] = retained
    result["retention_metrics"] = {
        "liftoff_clearance_m": liftoff_clearance,
        "samples": len(airborne),
        "first_liftoff_time_s": airborne[0]["time_s"] if airborne else None,
        "span_s": airborne[-1]["time_s"] - airborne[0]["time_s"] if airborne else 0.,
    }
    result["passed"] = bool(all(result["checks"].values()))
    return result


def ensure_release_over_tray(sample: dict, edge_clearance_m: float, *, acceptance: str = "stable") -> None:
    """Open only when the measured object is inside and close to the tray floor."""
    lower = np.asarray(sample["object_bounds_in_tray_min"], dtype=float)
    upper = np.asarray(sample["object_bounds_in_tray_max"], dtype=float)
    inner_min = np.asarray(sample["tray_inner_xy_min"], dtype=float)
    inner_max = np.asarray(sample["tray_inner_xy_max"], dtype=float)
    floor = float(sample["tray_floor_top_z"])
    if (lower.shape != (3,) or upper.shape != (3,) or inner_min.shape != (2,) or inner_max.shape != (2,)
            or not np.isfinite(np.r_[lower, upper, inner_min, inner_max, floor, edge_clearance_m]).all()):
        raise ValueError("Release needs finite measured object and tray bounds")
    if acceptance not in ("stable", "released_inside_tray"):
        raise ValueError("Unknown placement acceptance")
    margin = -.001 if acceptance == "released_inside_tray" else edge_clearance_m
    inside = np.all(lower[:2] >= inner_min + margin) and np.all(upper[:2] <= inner_max - margin)
    if not inside or not -.002 <= lower[2] - floor <= .02:
        raise RuntimeError("Object is not inside near the tray floor; keeping the gripper closed")


def evaluate_transfer_with_home(trace, initial_position, return_start, home_target, placement,
                                *, contact_diagnostics: bool) -> dict:
    """Keep strict grasp evidence separate from the requested task outcome."""
    from .placement_result import evaluate_placement
    from .return_home import evaluate_home_return
    from .scene_transfer_result import evaluate_scene_transfer
    acceptance = placement.get("acceptance", "stable")
    strict = evaluate_scene_transfer(
        trace[:return_start], initial_position, edge_clearance_m=placement["edge_clearance_m"],
        contact_diagnostics=contact_diagnostics,
    )
    home = evaluate_home_return(
        trace[return_start:], home_target, edge_clearance_m=placement["edge_clearance_m"], acceptance=acceptance,
    )
    if acceptance == "released_inside_tray":
        placed = evaluate_placement(trace[:return_start], acceptance=acceptance)
        checks = {"valid_transfer_evidence": bool(strict.get("checks")),
                  "released_inside_tray": placed["passed"], "returned_home": home["passed"]}
        return {"passed": bool(all(checks.values())), "checks": checks, "placement": placed,
                "return_home": home, "returned_home": home["passed"],
                "success_criterion": "released_inside_tray_and_return_home",
                "strict_quality_evaluation": strict}
    strict["return_home"] = home
    strict["checks"]["returned_home"] = home["passed"]
    strict["passed"] = bool(strict["passed"] and home["passed"])
    return strict


def gripper_phase_points(phase: dict, close_drive: float, limits: np.ndarray) -> np.ndarray:
    """Validate actuator targets; explicit vectors use RIGHT_GRIPPER_JOINTS order."""
    if ("drive" in phase) == ("servo_targets" in phase):
        raise ValueError("Gripper phase needs either drive or servo_targets")
    if "servo_targets" in phase:
        points = np.asarray(phase["servo_targets"], dtype=float)
    else:
        points = np.array([
            right_gripper_command(RIGHT_GRIPPER_JOINTS, close_drive if v == "close" else float(v))
            for v in phase["drive"]
        ])
    if points.ndim != 2 or points.shape[1] != 8 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Expected at least two finite eight-servo waypoints")
    if np.any(points < limits[:, 0]) or np.any(points > limits[:, 1]):
        raise ValueError("Gripper waypoint exceeds the original actuator limits")
    return points


def gripper_latch_phase(plan: dict) -> str:
    """Require a single final gripper phase before lifting, with no silent overrides."""
    name = plan.get("gripper_latch_after_phase", "close")
    phases = plan["phases"]
    names = [phase["name"] for phase in phases]
    if len(set(names)) != len(names):
        raise ValueError("Every plan phase must have a unique name")
    if plan.get("transfer") and (
        plan["transfer"].get("replan_after_phase") != "lift_hold" or names[-1] != "lift_hold"
    ):
        raise ValueError("Runtime transfer plan must end at lift_hold before replanning")
    matches = [i for i, phase in enumerate(phases) if phase["name"] == name]
    lift = [i for i, phase in enumerate(phases) if phase["name"] == "lift"]
    if len(matches) != 1 or len(lift) != 1 or matches[0] >= lift[0]:
        raise ValueError("Gripper latch phase must occur exactly once before lift")
    if phases[matches[0]]["joints"] != "gripper" or any(
        phase["joints"] == "gripper" for phase in phases[matches[0] + 1:]
    ):
        raise ValueError("Latch must follow the final gripper phase")
    return name


def implementation_sources(directory: Path) -> dict[str, bytes]:
    """Snapshot the local controller, geometry and success-check implementation."""
    return {name: (directory / name).read_bytes() for name in IMPLEMENTATION_FILES}


def implementation_hashes(sources: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}


def validate_offline_evidence(result: dict, prior: dict, expected: dict, *, preview: bool = False) -> None:
    """Require matching evidence; explicit human preview may inspect quality failures."""
    if result.get("online") is not False or prior.get("online") is not False:
        raise ValueError("Offline evidence must be explicitly offline")
    if preview:
        checks = result.get("checks")
        if (
            type(result.get("passed")) is not bool
            or result.get("error")
            or not isinstance(checks, dict)
            or not checks
            or any(type(value) is not bool for value in checks.values())
        ):
            raise ValueError("Preview needs a completed offline evaluation, not an aborted run")
    elif result.get("passed") is not True:
        raise ValueError("Offline evidence must be an explicitly offline passing run")
    if prior.get("profile") != "original" or prior.get("changes") != []:
        raise ValueError("Offline evidence must use original physics without modifications")
    for key in ("base_fixed", "waist_fixed", "gripper_hold_latched"):
        if prior.get(key) is not CONTROL_CONFIG[key]:
            raise ValueError(f"Offline evidence has incompatible {key}")
    for key in (
        "plan_sha256",
        "source_sha256",
        "asset_sha256",
        "close_drive",
        "contact_diagnostics",
        "target_body",
        "control_config",
        "implementation_sha256",
    ):
        if key not in prior or prior[key] != expected[key]:
            raise ValueError(f"Offline evidence does not match {key}")
    if result.get("target_body") != expected["target_body"]:
        raise ValueError("Offline evidence result has a different target body")
    mode = expected["contact_diagnostics"]
    if type(result.get("contact_diagnostics")) is not bool or result["contact_diagnostics"] is not mode:
        raise ValueError("Offline evidence result has a different contact_diagnostics mode")
    task_mode = expected.get("task_mode", "single_pick")
    if prior.get("task_mode", "single_pick") != task_mode or result.get("task_mode", "single_pick") != task_mode:
        raise ValueError("Offline evidence has a different task mode")
    criterion = "retention" if mode else "strict_contact_diagnostic"
    if task_mode == "tray_transfer":
        criterion = "transfer_" + criterion
    if result.get("success_criterion") != criterion:
        raise ValueError("Offline evidence result has a different success criterion")


def validate_code_archive(directory: Path, expected_hashes: dict[str, str]) -> None:
    try:
        archived = implementation_hashes(implementation_sources(directory / "code"))
    except OSError as exc:
        raise ValueError("Offline evidence code archive is incomplete") from exc
    if archived != expected_hashes:
        raise ValueError("Offline evidence code archive changed")


def archive_implementation(directory: Path, sources: dict[str, bytes]) -> None:
    code = directory / "code"
    code.mkdir()
    for name, content in sources.items():
        (code / name).write_bytes(content)
    (directory / "run.py").write_bytes(sources["scene_pick.py"])


def parse_args():
    """Load the plan and validate CLI combinations before any scene or output access."""
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--online", action="store_true")
    p.add_argument(
        "--preview", action="store_true",
        help="With --online, visually inspect a completed offline run even if quality checks failed; no training data",
    )
    p.add_argument("--offline-evidence", type=Path)
    p.add_argument(
        "--development-preview", action="store_true",
        help="With --online --preview, run a new tray-transfer plan without a prior dynamics run",
    )
    p.add_argument(
        "--initial-close-drive", "--close-drive", dest="close_drive", type=float,
        help="Initial close only; defaults to the plan. The tighten phase specifies the carrying grip separately",
    )
    p.add_argument(
        "--contact-diagnostics", action="store_true",
        help="Judge continuous retention; intended fingertip overlap is recorded, not a stop/pass criterion",
    )
    p.add_argument("--record-episode", type=Path, help="New local directory for synchronized native RGB/state/action")
    p.add_argument("--task-index", type=int, choices=(0, 1, 2), default=0, help="English paraphrase for this episode")
    args = p.parse_args()
    if args.record_episode and (not args.online or args.preview or args.development_preview):
        p.error("--record-episode requires --online and cannot be combined with preview flags")
    if args.preview and not args.online:
        p.error("--preview requires --online")
    if args.development_preview and not (args.online and args.preview):
        p.error("--development-preview requires --online --preview")
    plan = json.loads(args.plan.read_text())
    if args.close_drive is None:
        args.close_drive = float(plan.get("dynamic_validation_design", {}).get("close_drive", 2.0))
    transfer = plan.get("transfer")
    if args.record_episode and not transfer:
        p.error("Episode collection requires the complete tray-transfer task")
    if args.development_preview and not transfer:
        p.error("--development-preview is for a tray-transfer development plan")
    return args, plan


def main():
    args, plan = parse_args()
    transfer = plan.get("transfer")
    task_mode = "tray_transfer" if transfer else "single_pick"
    latch_phase = gripper_latch_phase(plan)
    source = Path(plan["source_xml"])
    target = plan["target_body"]
    END = "unitree_g1_1_arm_r_end_link"
    if sha256(source) != plan["source_sha256"]:
        raise ValueError("Scene XML changed")
    plan_hash = sha256(args.plan)
    sources = implementation_sources(Path(__file__).resolve().parent)
    if args.online:
        if args.offline_evidence is None and not args.development_preview and not args.record_episode:
            raise ValueError("Online needs --offline-evidence (passing unless --preview is explicit)")
        if args.offline_evidence is not None:
            prior = json.loads((args.offline_evidence / "result.json").read_text())
            prior_m = json.loads((args.offline_evidence / "manifest.json").read_text())

    def check_live_export():
        import xml.etree.ElementTree as ET

        import grpc
        from orca_gym.protos import mjc_message_pb2 as pb
        from orca_gym.protos import mjc_message_pb2_grpc as rpc

        with grpc.insecure_channel("127.0.0.1:50051") as ch:
            stub = rpc.GrpcServiceStub(ch)
            raw = stub.LoadLocalEnv(
                pb.LoadLocalEnvRequest(req_type=pb.LoadLocalEnvRequest.XML_FILE_CONTENT), timeout=15
            )
            if raw.status != 0 or hashlib.sha256(raw.xml_content).hexdigest() != plan["source_sha256"]:
                raise ValueError("Live scene XML differs from validated source")
            for node in ET.fromstring(raw.xml_content).findall("./asset/*[@file]"):
                raw_asset = stub.LoadContentFile(pb.LoadContentFileRequest(file_name=node.get("file")), timeout=15)
                expected = sha256(source.parent / node.get("file"))
                if raw_asset.status != 0 or hashlib.sha256(raw_asset.content).hexdigest() != expected:
                    raise ValueError("Live mesh/asset changed")

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = prepare_model(source, args.output / "scene.xml", "original")
    manifest.update(
        plan_sha256=plan_hash,
        online=args.online,
        preview=args.preview,
        development_preview=args.development_preview,
        task_mode=task_mode,
        base_fixed=True,
        waist_fixed=False,
        training_data=False,
        recording_requested=bool(args.record_episode),
        close_drive=args.close_drive,
        contact_diagnostics=args.contact_diagnostics,
        physics_dt=0.002,
        control_dt=0.02,
        target_body=target,
        gripper_hold_latched=True,
        control_config=dict(CONTROL_CONFIG),
        implementation_sha256=implementation_hashes(sources),
    )
    if args.online:
        if args.offline_evidence is not None:
            validate_offline_evidence(prior, prior_m, manifest, preview=args.preview)
            validate_code_archive(args.offline_evidence, manifest["implementation_sha256"])
        check_live_export()
        if args.preview:
            print(
                "人工预览：从布局初始状态执行当前运控，不生成训练数据。",
                flush=True,
            )
    if args.contact_diagnostics:
        print("夹持优先：指尖压入量仅作诊断记录；检查完整抬升、不掉落及持续接触。", flush=True)
    archive_implementation(args.output, sources)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(args.plan, args.output / "plan.json")
    e = BluePickEnv(
        str(source),
        online=args.online,
        timestep=CONTROL_CONFIG["physics_dt"],
        frame_skip=CONTROL_CONFIG["frame_skip"],
        bias_compensation=CONTROL_CONFIG["bias_compensation"],
        position_feedback_gain=CONTROL_CONFIG["position_feedback_gain"],
    )
    trace = []
    states = []
    commands = []
    servos = []
    velocities = []
    recorder = None
    cameras = None
    result = {
        "passed": False,
        "online": args.online,
        "preview": args.preview,
        "development_preview": args.development_preview,
        "task_mode": task_mode,
        "contact_diagnostics": args.contact_diagnostics,
        "target_body": target,
        "training_data": False,
    }
    try:
        if args.online:
            check_live_export()
        if not np.allclose(e.data.qpos.ravel(), plan["initial_full_qpos"], atol=1e-7, rtol=0):
            raise ValueError("Initial configuration changed")
        e.fix_base()
        control = e.control_from_qpos()
        gi = [e.actuator_joint_names.index(n) for n in RIGHT_GRIPPER_JOINTS]
        li = [e.actuator_joint_names.index(n) for n in plan["left_joint_names"]]
        ai = [e.actuator_joint_names.index(n) for n in plan["arm_joint_names"]]
        gripper_points = {
            phase["name"]: gripper_phase_points(phase, args.close_drive, e.control_limits[gi])
            for phase in plan["phases"] if phase["joints"] == "gripper"
        }
        print(
            f"初次闭合目标 {args.close_drive:.2f}；搬运保持目标 "
            f"{gripper_points[latch_phase][-1].tolist()}；两者作用阶段不同。", flush=True,
        )
        control[gi] = right_gripper_command(RIGHT_GRIPPER_JOINTS, -1.0)
        geom = {g["GeomId"]: g["BodyName"] for g in e.model.get_geom_dict().values()}
        com_local = e.model.get_body_dict()[target]["IPos"]
        prim = [
            {
                "geom_type": g["Type"],
                "geom_size": g["Size"],
                "geom_pos": g["Pos"],
                "geom_quat": g["Quat"],
                "geom_contype": g["Contype"],
                "geom_conaffinity": g["Conaffinity"],
            }
            for g in e.model.get_geom_dict().values()
            if g["BodyName"] == target
        ]
        start_pos, start_quat = e.read_pose(target)
        floor = body_bounds(prim, quat_matrix(start_quat), start_pos)[0][2]
        phases = [{"name": "settle", "joints": "arm", "q": [control[ai].tolist()] * 2, "duration_s": 3.0}]
        left = plan["left_waypoints"]
        phases += [
            {"name": n, "joints": "left", "q": [a, b], "duration_s": d}
            for n, a, b, d in zip(["left_retract", "left_withdraw", "left_lower"], left[:-1], left[1:], [3.0, 4.0, 4.0])
        ]
        phases += [{"name": "left_hold", "joints": "left", "q": [left[-1]] * 2, "duration_s": 1.0}]
        phases += plan["phases"]
        placement = None
        return_start = None
        home_target = None
        if transfer:
            from .placement_sample import placement_geometry, sample_placement
            placement = placement_geometry(e, plan["placement"], target)
        for phase in phases:
            name = phase["name"]
            print(f"开始阶段 {name}，仿真时间 {float(e.data.time):.2f}s", flush=True)
            kind = phase["joints"]
            ids = {"arm": ai, "left": li, "gripper": gi}[kind]
            if name == "release":
                if placement is None:
                    raise ValueError("Release requires a tray placement plan")
                measured = sample_placement(e, placement)
                ensure_release_over_tray(measured, placement["edge_clearance_m"],
                                         acceptance=placement.get("acceptance", "stable"))
                e.right_gripper_hold.release()
            if kind == "gripper":
                points = gripper_points[name]
            else:
                points = np.array(phase["q"])
            for i in range(1, round(phase["duration_s"] / 0.02) + 1):
                t = i / round(phase["duration_s"] / 0.02)
                t = t * t * (3 - 2 * t)
                x = t * (len(points) - 1)
                j = min(int(x), len(points) - 2)
                w = x - j
                control[ids] = (1 - w) * points[j] + w * points[j + 1]
                if recorder is None:
                    e.step(control)
                else:
                    from .native_capture import record_control_step
                    record_control_step(e, control, ai, gi, recorder, cameras, name)
                    if len(recorder.rows) % 100 == 0:
                        print(f"已录制 {len(recorder.rows)} 帧双RGB，阶段 {name}", flush=True)
                e.mj_forward()
                pos, quat = e.read_pose(target)
                ep, eq = e.read_pose(END)
                rot = quat_matrix(eq)
                contacts = e.query_contact_simple()
                sel = [k for k, c in enumerate(contacts) if target in (geom[c["geom1"]], geom[c["geom2"]])]
                forces = e.query_contact_force(sel)
                readings = []
                forbidden = []
                for k in sel:
                    c = contacts[k]
                    pair = [geom[c["geom1"]], geom[c["geom2"]]]
                    other = pair[1] if pair[0] == target else pair[0]
                    f = float(forces[k][0])
                    dist = float(c["dist"])
                    if not np.isfinite([f, dist]).all():
                        raise RuntimeError("Nonfinite target contact")
                    normal = np.asarray(c["frame"]).reshape(3, 3)[0] * (-1 if pair[0] == target else 1)
                    readings.append(
                        {
                            "other_body": other,
                            "normal_force_n": f,
                            "distance_m": dist,
                            "position_world_m": np.asarray(c["pos"]).tolist(),
                            "normal_on_object_world": normal.tolist(),
                        }
                    )

                def moving(n):
                    return any(
                        v in n
                        for v in (
                            "_right_shoulder_",
                            "_right_elbow_",
                            "_right_wrist_",
                            "_left_shoulder_",
                            "_left_elbow_",
                            "_left_wrist_",
                            "_gripper_r_",
                            "_gripper_l_",
                            "_arm_r_end_",
                            "_arm_l_end_",
                            "_camera_left_",
                            "_camera_right_",
                        )
                    )

                for c in contacts:
                    a, b = geom[c["geom1"]], geom[c["geom2"]]
                    if not (moving(a) or moving(b)) or c["dist"] >= -0.0001:
                        continue
                    if any(all("_gripper_" + side + "_" in n for n in [a, b]) for side in ["l", "r"]):
                        continue
                    if target in (a, b) and (a in TIPS or b in TIPS):
                        continue
                    forbidden.append({"bodies": [a, b], "distance_m": float(c["dist"])})
                lower, _ = body_bounds(prim, quat_matrix(quat), pos)
                measured_control = e.control_from_qpos()
                row = {
                    "phase": name,
                    "time_s": float(e.data.time),
                    "blue_position": pos.tolist(),
                    "object_position": pos.tolist(),
                    "object_quaternion": quat.tolist(),
                    "relative_position": (rot.T @ (pos - ep)).tolist(),
                    "relative_rotation": (rot.T @ quat_matrix(quat)).tolist(),
                    "lowest_collider_clearance_m": float(lower[2] - floor),
                    "contacts": readings,
                    "forbidden_contacts": forbidden,
                    "grasp_alignment": analyze_contacts(pos, quat, com_local, readings, TIPS),
                    "control_saturated": bool(e.last_control_saturated.any()),
                    "force_saturated": bool(e.last_force_saturated.any()),
                    "waist_rad": {
                        n: float(measured_control[k])
                        for k, n in enumerate(e.actuator_joint_names)
                        if "_waist_" in n
                    },
                }
                row["gripper"] = {
                    "hold_active": e.right_gripper_hold.active,
                    "joint_names": list(RIGHT_GRIPPER_JOINTS),
                    "nominal_command": e.last_command[gi].tolist(),
                    "servo_last_substep": e.last_servo_control[gi].tolist(),
                    "qpos": e.data.qpos.ravel()[e.actuator_qpos_indices[gi]].tolist(),
                    "qvel": e.data.qvel.ravel()[e.actuator_dof_indices[gi]].tolist(),
                    "actuator_force": e.data.actuator_force.ravel()[gi].tolist(),
                }
                row["right_arm_qpos"] = e.data.qpos.ravel()[e.actuator_qpos_indices[ai]].tolist()
                row["right_arm_qvel"] = e.data.qvel.ravel()[e.actuator_dof_indices[ai]].tolist()
                if placement is not None:
                    row.update(sample_placement(e, placement))
                trace.append(row)
                states.append(e.data.qpos.ravel().copy())
                commands.append(e.last_command.copy())
                servos.append(e.last_servo_control.copy())
                velocities.append(e.data.qvel.ravel().copy())
                if not np.isfinite(states[-1]).all():
                    raise RuntimeError("Nonfinite state")
                if forbidden:
                    raise RuntimeError(f"Arm collision in {name}: {forbidden}")
                if (
                    name not in ("settle", "left_retract", "left_withdraw", "left_lower", "left_hold")
                    and penetration_requires_stop(readings, contact_diagnostics=args.contact_diagnostics)
                ):
                    raise RuntimeError(f"Target penetration exceeds 4mm in {name}")
                if args.online and recorder is None:
                    e.render()
                    time.sleep(0.02)
            if name == latch_phase:
                e.right_gripper_hold.latch_command(control[gi])
            if name == "lift_hold" and transfer:
                from .plan_tray_transfer import plan_transport
                if (not row["gripper"]["hold_active"] or row["lowest_collider_clearance_m"] < .002
                        or any(sum(c["normal_force_n"] for c in readings if c["other_body"] == tip) < .1
                               for tip in TIPS)
                        or sum(max(0., c["normal_force_n"]) for c in readings if c["other_body"] not in TIPS) >= .1):
                    raise RuntimeError("Object is not held clear of its support; cannot begin transport")
                print("抓取完成，正在按实际持物姿态计算横移、下放与退手路径……", flush=True)
                continuation = plan_transport(str(source), e.data.qpos.ravel().copy(), plan)
                added = continuation["phases"]
                expected_names = list(TRANSFER_PHASES[1:])
                if [p["name"] for p in added] != expected_names:
                    raise ValueError("Transport planner returned an unexpected phase order")
                for following in added:
                    if following["joints"] == "gripper":
                        gripper_points[following["name"]] = gripper_phase_points(
                            following, args.close_drive, e.control_limits[gi]
                        )
                phases.extend(added)
                (args.output / "transport_plan.json").write_text(json.dumps(continuation, indent=2))
                print("抓取后已按实际持物姿态生成托盘搬运路径。", flush=True)
            if name == "settle_placed" and transfer:
                from .placement_result import evaluate_placement
                from .return_home import plan_home_return
                placed = evaluate_placement(trace, edge_clearance_m=placement["edge_clearance_m"],
                                            acceptance=placement.get("acceptance", "stable"))
                if not placed["passed"]:
                    raise RuntimeError(
                        f"Object does not meet tray placement acceptance; cannot begin home return: {placed}"
                    )
                returning = plan_home_return(str(source), e.data.qpos.ravel().copy(), plan)
                return_start = len(trace)
                home_target = returning["home_joint_target"]
                phases.extend(returning["phases"])
                (args.output / "return_home_plan.json").write_text(json.dumps(returning, indent=2))
                print("物体已通过放置检查，夹爪保持张开，右臂返回布局默认位置。", flush=True)
            if name == "left_hold":
                start_pos, start_quat = e.read_pose(target)
                floor = body_bounds(prim, quat_matrix(start_quat), start_pos)[0][2]
                if np.linalg.norm(start_pos - plan["settled_target_position"]) > 0.002:
                    raise RuntimeError("Settled target differs from planned position")
                angle = np.linalg.norm(
                    Rotation.from_matrix(
                        quat_matrix(start_quat) @ quat_matrix(plan["settled_target_quaternion"]).T
                    ).as_rotvec()
                )
                if angle > 0.01:
                    raise RuntimeError("Settled target differs from planned orientation")
                if args.record_episode:
                    from .episode_recording import TASK_TEXTS, EpisodeRecorder
                    from .native_capture import NativeCameraPair
                    cameras = NativeCameraPair(e)
                    recorder = EpisodeRecorder(
                        args.record_episode, TASK_TEXTS[args.task_index], plan["arm_joint_names"],
                        RIGHT_GRIPPER_JOINTS, fps=50.0,
                        metadata={
                            "source_xml": str(source.resolve()), "source_sha256": plan["source_sha256"],
                            "plan_sha256": plan_hash, "implementation_sha256": manifest["implementation_sha256"],
                            "diagnostics_directory": str(args.output.resolve()),
                            "target_body": target, "camera_names": cameras.camera_names,
                            "native_camera_properties": cameras.properties,
                            "image_preprocessing": (
                                "native RGB -> aspect-preserving bilinear resize and black pad to 224x224; "
                                "gamma unchanged"
                            ),
                            "initial_target_position": start_pos.tolist(),
                            "initial_target_quaternion_wxyz": start_quat.tolist(),
                            "initial_state_full_qpos": e.data.qpos.ravel().tolist(),
                            "physics_dt": CONTROL_CONFIG["physics_dt"], "control_dt": .02,
                            "physics_backend": "OrcaGymEuler / MuJoCo CPU",
                            "expert_gripper_hold": "latched calibrated targets until explicit tray release",
                            "includes_return_home": True,
                            "success_criterion": (
                                "released_inside_tray_and_return_home"
                                if placement is not None and placement.get("acceptance") == "released_inside_tray"
                                else "transfer_retention" if args.contact_diagnostics
                                else "transfer_strict_contact_diagnostic"
                            ),
                        },
                    )
                    print(f"开始同步采集：{args.record_episode}\nTask: {recorder.task}", flush=True)
            print(
                name,
                "t",
                round(row["time_s"], 2),
                "target",
                np.round(pos, 4).tolist(),
                "tip forces",
                [round(sum(c["normal_force_n"] for c in readings if c["other_body"] == tip), 2) for tip in TIPS],
                flush=True,
            )
        if placement is None:
            result = evaluate_scene_pick(trace, start_pos.tolist(), contact_diagnostics=args.contact_diagnostics)
        else:
            result = evaluate_transfer_with_home(
                trace, start_pos.tolist(), return_start, home_target, placement,
                contact_diagnostics=args.contact_diagnostics,
            )
        result.update(
            online=args.online,
            preview=args.preview,
            development_preview=args.development_preview,
            task_mode=task_mode,
            contact_diagnostics=args.contact_diagnostics,
            target_body=target,
            training_data=False,
            scope=("Single object pick, carry, tray placement and right-arm return" if transfer
                   else "Single object pick only")
                  + "; original contact and inertia. 50Hz samples.",
            forbidden_contact_samples=sum(bool(r["forbidden_contacts"]) for r in trace),
            control_saturated_steps=sum(r["control_saturated"] for r in trace),
            force_saturated_steps=sum(r["force_saturated"] for r in trace),
            max_waist_rad=max(abs(v) for r in trace for v in r["waist_rad"].values()),
        )
    except Exception as exc:
        result = {
            "passed": False,
            "error": str(exc),
            "online": args.online,
            "preview": args.preview,
            "development_preview": args.development_preview,
            "task_mode": task_mode,
            "contact_diagnostics": args.contact_diagnostics,
            "target_body": target,
            "training_data": False,
        }
        print("ERROR", exc, flush=True)
    finally:
        try:
            if recorder is not None:
                result["training_data"] = bool(result.get("passed"))
                recorded = recorder.finish(bool(result.get("passed")), result.copy())
                result["episode"] = {
                    "directory": str(recorder.directory.resolve()), "frame_count": recorded["frame_count"],
                    "success": recorded["success"], "task": recorded["task"], "training_ready": False,
                }
                (recorder.directory / "camera_frames.json").write_text(json.dumps(cameras.frame_metadata))
            (args.output / "trace.json").write_text(json.dumps(trace, indent=2))
            np.savez_compressed(
                args.output / "controls.npz",
                qpos_after=states,
                qvel_after=velocities,
                nominal_action=commands,
                servo_last_substep=servos,
            )
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
        finally:
            if cameras is not None:
                cameras.close()
            e.close()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
