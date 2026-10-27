"""Read the live OrcaLab scene and compute two-object placement candidates.

Run from the repository root with python -m examples.euler.tray_transfer.inspect_scene.
Uses public query RPCs only, without constructing or resetting an environment.
"""

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import grpc

from .task import build_report


def read_snapshot(endpoint, timeout):
    from google.protobuf.json_format import MessageToDict
    from orca_gym.protos import mjc_message_pb2 as pb
    from orca_gym.protos import mjc_message_pb2_grpc as rpc

    # Keep an explicit allowlist: extending this inspector must not start/step/reset the scene.
    queries = ["QueryModelInfo", "QueryAllBodies", "QueryAllJoints", "QueryAllActuators",
               "QueryAllGeoms", "QueryAllSites", "QueryAllEqualityConstraints",
               "GetCameraNames", "GetBodyManipulationAnchored"]
    responses = {}
    started = datetime.now(timezone.utc).isoformat()
    with grpc.insecure_channel(endpoint) as channel:
        stub = rpc.GrpcServiceStub(channel)
        for query in queries:
            result = getattr(stub, query)(getattr(pb, query + "Request")(), timeout=timeout)
            responses[query] = MessageToDict(result, preserving_proto_field_name=True,
                                             always_print_fields_with_no_presence=True)
        names = [body["name"] for body in responses["QueryAllBodies"]["body_info"]]
        result = stub.QueryBodyPosMatQuat(pb.QueryBodyPosMatQuatRequest(body_name_list=names), timeout=timeout)
        responses["QueryBodyPosMatQuat"] = MessageToDict(
            result, preserving_proto_field_name=True, always_print_fields_with_no_presence=True
        )
    return {
        "observed_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint, "read_only": True,
        "consistency": "Sequential queries; not an atomic simulation frame. Use a stationary scene.",
        "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("task.json"))
    parser.add_argument("--snapshot", type=Path, help="Inspect a saved JSON instead of connecting to OrcaLab")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        if args.snapshot:
            snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        else:
            snapshot = read_snapshot(args.endpoint, args.timeout)
    except grpc.RpcError as exc:
        print(f"OrcaLab query failed at {args.endpoint}: {exc.code().name}: {exc.details()}", file=sys.stderr)
        return 2
    output = args.output_dir or Path(__file__).with_name("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        report = build_report(snapshot, config)
    except (ValueError, KeyError) as exc:
        print(f"Scene/config mismatch: {exc}\nSnapshot: {(output / 'scene_snapshot.json').resolve()}", file=sys.stderr)
        return 2
    report["observed_at_utc"] = snapshot.get("observed_at_utc")
    report["source"] = str(args.snapshot.resolve()) if args.snapshot else args.endpoint
    report["task_config"] = config
    report["runtime"] = {"python": sys.executable}
    for package in ["orcalab", "orca-gym", "numpy", "torch", "nvidia-curobo"]:
        try:
            report["runtime"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["runtime"][package] = None
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Geometry preflight: {'PASS' if report['geometry_passed'] else 'FAIL'} (not a motion/grasp test)")
    print("Order: " + " -> ".join(report["object_order"]))
    for item in report["objects"]:
        size = " x ".join(f"{value * 100:.2f}" for value in item["collision_size_body_m"])
        print(f"{item['id']}: {size} cm, {item['mass_kg']:.3f} kg; body={item['body_name']}")
        print("  candidate body origin in world: " + str(item["placement"]["body_origin_world"]))
    for error in report["geometry_errors"]:
        print("ERROR: " + error)
    print("Unverified: " + ", ".join(report["unverified"]))
    print(f"Report: {(output / 'report.json').resolve()}")
    return 0 if report["geometry_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
