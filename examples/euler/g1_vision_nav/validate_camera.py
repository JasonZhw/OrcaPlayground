"""Rebuild Demo 1 and validate the fixed G1 head RGB stream online."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from layout_scene import publish_layout_with_g1

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = PROJECT_ROOT.parent / "demo1_v1.json"
DEFAULT_SAMPLE_PATH = Path("/tmp/g1_camera_head_rgb.png")
DEFAULT_SPAWN_XYZ = (-18.5, -13.5, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 头部相机验证：重建 Layout → 原地站立 → RGB 实时流")
    parser.add_argument("--addr", default="127.0.0.1:50051", help="OrcaLab gRPC 地址")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--spawn-x", type=float, default=DEFAULT_SPAWN_XYZ[0])
    parser.add_argument("--spawn-y", type=float, default=DEFAULT_SPAWN_XYZ[1])
    parser.add_argument("--spawn-z", type=float, default=DEFAULT_SPAWN_XYZ[2])
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=500, help="50 Hz 控制步数")
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_PATH)
    return parser.parse_args()


def _yaw_quaternion_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half_yaw = math.radians(yaw_deg) * 0.5
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _load_components():
    root_string = str(PROJECT_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    lesson_dir = PROJECT_ROOT / "examples" / "euler" / "07_locomotion"
    lesson_dir_string = str(lesson_dir)
    if lesson_dir_string not in sys.path:
        sys.path.insert(0, lesson_dir_string)

    from g1_base_env import G1_FRAME_SKIP, G1_MODEL_XML, G1_TIME_STEP
    from online_verifier import OnlineVerifier

    from envs.euler.g1_vision_nav.config import DEFAULT_CONFIG
    from envs.euler.g1_vision_nav.g1_camera_validation_env import (
        G1CameraValidationEnv,
    )

    return (
        DEFAULT_CONFIG,
        G1CameraValidationEnv,
        G1_FRAME_SKIP,
        G1_MODEL_XML,
        G1_TIME_STEP,
        OnlineVerifier,
    )


def main() -> None:
    args = parse_args()
    (
        config,
        camera_env_class,
        frame_skip,
        model_xml,
        time_step,
        verifier_class,
    ) = _load_components()
    spawn_position = (args.spawn_x, args.spawn_y, args.spawn_z)

    print("[WARNING] 即将清空 OrcaLab 当前运行场景，并从 Layout JSON 重新生成。")
    print(f"[INFO] Layout: {args.layout.expanduser().resolve()}")
    print(f"[INFO] G1: {config.scene.robot_asset_path}")
    print(f"[INFO] Camera: {config.camera.entity_name}, RGB ws://localhost:{config.camera.rgb_port}")

    actor_count = publish_layout_with_g1(
        grpc_addr=args.addr,
        layout_path=args.layout,
        g1_actor_name=config.scene.robot_actor_name,
        g1_asset_path=config.scene.robot_asset_path,
        g1_position_xyz=spawn_position,
        g1_rotation_wxyz=_yaw_quaternion_wxyz(args.spawn_yaw),
    )
    print(f"[INFO] 已发布 {actor_count} 个 Layout 物体 + G1；等待服务重启……")
    time.sleep(3.0)

    env = camera_env_class(
        frame_skip=frame_skip,
        orcagym_addr=args.addr,
        agent_names=[config.scene.robot_actor_name],
        time_step=time_step,
        model_xml_path=model_xml,
        camera_config=config.camera,
        sample_path=args.output.expanduser().resolve(),
    )
    verifier = verifier_class("Demo 1: G1 camera_head RGB 验证")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
