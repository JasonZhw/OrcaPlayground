#!/usr/bin/env python3
"""药品分类场景探针；读取模型与短时观测，不重置仿真、不发送动作。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from scene_utils import REQUIRED_ACTORS, ROBOT, analyze_layout, match_body_names, validate_observation

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = PROJECT / "data/medication_sorting/scenes/drag_classfication.json"


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化：{type(value)}")


def save_json(path, data):
    path.write_text(json.dumps(data, default=json_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def software_versions():
    result = {"python": sys.version.split()[0], "executable": sys.executable}
    for package in ("orca-lab", "orca-gym", "torch", "warp-lang", "nvidia-curobo", "h5py", "av"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


async def inspect_live(args, output, report):
    import grpc
    from google.protobuf.json_format import MessageToDict
    from orca_gym.core.orca_gym_remote import OrcaGymRemote
    # OrcaGymRemote 同样使用这套官方协议模块；不访问 env 私有属性。
    import mjc_message_pb2 as pb
    from mjc_message_pb2_grpc import GrpcServiceStub

    async def call(operation):
        return await asyncio.wait_for(operation, timeout=args.timeout)

    channel = grpc.aio.insecure_channel(args.addr, options=[("grpc.max_receive_message_length", 256*1024*1024)])
    try:
        await call(channel.channel_ready())
        client_stub = GrpcServiceStub(channel)
        client = OrcaGymRemote(client_stub)
        model = await call(client.query_model_info())
        if model["nq"] <= 0:
            raise RuntimeError("物理模型为空：请在 OrcaLab 中加载场景并点击运行")
        bodies = await call(client.query_all_bodies())
        joints = await call(client.query_all_joints())
        actuators = await call(client.query_all_actuators())
        sites = await call(client.query_all_sites())
        geoms = await call(client.query_all_geoms())
        asset_paths = {
            asset["name"]: asset.get("asset_path")
            for asset in report.get("offline_layout", {}).get("assets", [])
        }
        candidates = {
            actor: match_body_names(actor, bodies, asset_path=asset_paths.get(actor))
            for actor in REQUIRED_ACTORS
        }
        ids = {bodies[name]["ID"] for name in candidates[ROBOT]}
        robot_joints = {name: info for name, info in joints.items() if info["BodyID"] in ids}
        robot_actuators = {name: info for name, info in actuators.items() if info["JointName"] in robot_joints}
        robot_sites = {name: info for name, info in sites.items() if info["BodyID"] in ids}
        missing = [name for name, names in candidates.items() if not names]
        camera_errors, cameras = {}, {}
        try:
            response = await call(client_stub.GetCameraNames(pb.GetCameraNamesRequest()))
            if response.status != pb.GetCameraNamesResponse.SUCCESS:
                raise RuntimeError(response.error_message)
            for name in response.camera_names:
                try:
                    props = await call(client_stub.GetCameraProperties(pb.GetCameraPropertiesRequest(camera_name=name)))
                    if props.status != pb.GetCameraPropertiesResponse.SUCCESS:
                        raise RuntimeError(props.error_message)
                    cameras[name] = MessageToDict(props, preserving_proto_field_name=True)
                except (RuntimeError, TimeoutError, grpc.RpcError) as exc:
                    camera_errors[name] = str(exc)
        except (RuntimeError, TimeoutError, grpc.RpcError) as exc:
            camera_errors["enumeration"] = str(exc)
        selected_bodies = sorted({name for names in candidates.values() for name in names})
        positions = await call(client.query_body_xpos_xmat_xquat(selected_bodies)) if selected_bodies else {}
        report["live"] = {
            "model": model, "bodies": bodies, "joints": joints, "actuators": actuators,
            "sites": sites, "geoms": geoms, "body_candidates": candidates,
            "missing_physics_actors": missing, "robot_joints": robot_joints,
            "robot_actuators": robot_actuators, "robot_sites": robot_sites,
            "body_poses": positions, "contacts": await call(client.query_contact_simple()),
            "cameras": cameras, "camera_errors": camera_errors,
            "sensor_enumeration": "当前 Remote 协议没有枚举所有传感器名称的接口，未猜测传感器名称。",
            "binding_note": "候选名需要结合资产结构确认；本探针不建立控制映射。",
        }
        save_json(output / "scene_manifest.json", report)
        if missing or not robot_joints or not robot_actuators:
            raise RuntimeError(f"场景物理结构尚未完整匹配：缺失 Actor={missing}，G1 关节={len(robot_joints)}，执行器={len(robot_actuators)}；请核对场景和资产物理组件")

        if args.record_seconds > 0:
            count = 0
            started = time.monotonic()
            deadline = started + args.record_seconds
            with (output / "observations.jsonl").open("x") as journal:
                while time.monotonic() < deadline:
                    request_ns = time.time_ns()
                    qpos, qvel, qacc = await call(client.query_all_qpos_qvel_qacc())
                    validate_observation(qpos, qvel, qacc, nq=model["nq"], nv=model["nv"])
                    row = {"index": count, "request_wall_time_ns": request_ns, "response_wall_time_ns": time.time_ns(),
                           "elapsed_wall_seconds": time.monotonic() - started,
                           "qpos": qpos, "qvel": qvel, "qacc": qacc}
                    journal.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")
                    journal.flush()
                    count += 1
                    await asyncio.sleep(min(1/args.hz, max(0, deadline - time.monotonic())))
            report["observation_recording"] = {
                "sample_count": count, "elapsed_wall_seconds": time.monotonic() - started,
                "data_kind": "observation_only", "robot_actions_recorded": False,
                "rgb_recorded": False, "image_state_alignment_verified": False,
                "trainable_for_pi05": False, "success": None,
            }
        report["status"] = "live_scene_inspected"
        return 0
    finally:
        await channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--addr", default="127.0.0.1:50051")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--offline", action="store_true", help="只检查 Layout，不连接仿真")
    parser.add_argument("--record-seconds", type=float, default=0, help="记录只含状态的诊断样本，不作为训练数据")
    parser.add_argument("--hz", type=float, default=10)
    parser.add_argument("--output", type=Path, help="新建实验目录；不得覆盖既有结果")
    args = parser.parse_args()
    if not 0 <= args.record_seconds <= 60 or not 0 < args.hz <= 100 or not 0 < args.timeout <= 30:
        parser.error("时长范围 0–60 秒、采样频率 0–100 Hz、连接超时 0–30 秒")
    if args.offline and args.record_seconds:
        parser.error("离线模式不能录制真实观测")
    output = args.output or PROJECT / "data/medication_sorting/reports" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "software_versions": software_versions(), "address": args.addr,
              "layout_file": str(args.layout.resolve()), "robot_control_sent": False, "grasp_verified": False}
    exit_code = 0
    try:
        contents = args.layout.read_bytes()
        report["layout_sha256"] = hashlib.sha256(contents).hexdigest()
        report["offline_layout"] = analyze_layout(json.loads(contents))
        if report["offline_layout"]["missing_task_actors"]:
            raise ValueError(f"Layout 缺少任务对象：{report['offline_layout']['missing_task_actors']}")
        if args.offline:
            report["status"] = "offline_checked"
        else:
            host, port = args.addr.rsplit(":", 1)
            with socket.create_connection((host, int(port)), timeout=args.timeout):
                pass
            exit_code = asyncio.run(inspect_live(args, output, report))
    except PermissionError as exc:
        report.update(status="blocked_by_execution_permissions", error=str(exc))
        exit_code = 2
    except ConnectionRefusedError as exc:
        report.update(status="simulation_not_listening", error=str(exc))
        exit_code = 2
    except KeyboardInterrupt:
        report.update(status="interrupted")
        exit_code = 130
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        save_json(output / "scene_manifest.json", report)
        print(f"状态：{report['status']}")
        if "error" in report:
            print(f"原因：{report['error']}")
        print(f"报告：{output / 'scene_manifest.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
