"""Run RGB-driven workbench avoidance with frozen G1 locomotion."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from layout_scene import publish_layout_with_g1

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = PROJECT_ROOT.parent / "demo_v2.json"
DEFAULT_SPAWN_XYZ = (8.0, 0.4636681079864502, 0.0)
INSPECTION_POINT_XY = (19.2, 4.3)
ROUTES = {
    "left": (
        (9.0, 3.5),
        (10.0, 3.5),
        (13.0, 7.1),
        INSPECTION_POINT_XY,
    ),
    "right": ((13.0, -5.0), INSPECTION_POINT_XY),
}
DEFAULT_STEPS = {"left": 8000, "right": 8000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 左/右路径巡检：直线 vx/yaw 跟随 + RGB 连续避障修正 + 柜前拍照"
    )
    parser.add_argument("--addr", default="127.0.0.1:50051", help="OrcaLab gRPC 地址")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--spawn-x", type=float, default=DEFAULT_SPAWN_XYZ[0])
    parser.add_argument("--spawn-y", type=float, default=DEFAULT_SPAWN_XYZ[1])
    parser.add_argument("--spawn-z", type=float, default=DEFAULT_SPAWN_XYZ[2])
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--route", choices=("left", "right"), default="left")
    parser.add_argument("--max-speed", type=float, default=0.60, help="空旷巡航速度上限，最大 0.60m/s")
    parser.add_argument("--num-steps", type=int, default=None, help="50 Hz 控制步数")
    parser.add_argument("--output", type=Path, default=None, help="到达柜前后保存的 RGB PNG")
    parser.add_argument(
        "--camera-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示独立的 G1 camera_head 第一视角窗口（默认开启）",
    )
    parser.add_argument(
        "--camera-window-port",
        type=int,
        default=8765,
        help="第一视角浏览器窗口的本地端口（默认 8765）",
    )
    parser.add_argument("--publish-wait", type=float, default=3.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
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
    from envs.euler.g1_vision_nav.g1_visual_avoidance_env import G1VisualAvoidanceEnv
    from envs.euler.g1_vision_nav.visual_avoidance import VisualAvoidanceNavigator

    return (
        DEFAULT_CONFIG,
        G1VisualAvoidanceEnv,
        VisualAvoidanceNavigator,
        G1_FRAME_SKIP,
        G1_MODEL_XML,
        G1_TIME_STEP,
        OnlineVerifier,
    )


def _create_env_with_retry(environment_class, *, timeout_s: float, **kwargs):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return environment_class(**kwargs)
        except Exception as exc:
            transient = any(
                token in str(exc)
                for token in (
                    "MuJoCo has not been initialized while starting up",
                    "failed to connect to all addresses",
                    "Connection refused",
                    "UNAVAILABLE",
                )
            )
            if not transient or time.monotonic() >= deadline:
                raise
            print("[INFO] MuJoCo service is restarting; retrying in 2s...")
            time.sleep(2.0)


def main() -> None:
    args = parse_args()
    if args.num_steps is None:
        args.num_steps = DEFAULT_STEPS[args.route]
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if not 0.0 < args.max_speed <= 0.60:
        raise ValueError("--max-speed must be in (0, 0.60]")
    if args.publish_wait < 0.0 or args.startup_timeout <= 0.0:
        raise ValueError("publish/startup timing must be positive")
    if not 0 <= args.camera_window_port <= 65535:
        raise ValueError("--camera-window-port must be in [0, 65535]")
    waypoints = ROUTES[args.route]
    output = args.output or Path(f"/tmp/g1_cabinet_{args.route}_rgb.png")
    (
        config,
        environment_class,
        navigator_class,
        frame_skip,
        model_xml,
        time_step,
        verifier_class,
    ) = _load_components()
    command_limits = replace(
        config.command_limits,
        max_forward_mps=float(args.max_speed),
    )
    navigator = navigator_class(
        navigation=config.navigation,
        limits=command_limits,
        visual=config.visual_avoidance,
    )

    print("[INFO] 保留当前 Layout，只替换上一次代码生成的 G1。")
    print(f"[INFO] Layout: {args.layout.expanduser().resolve()}")
    print(f"[INFO] G1: {config.scene.robot_asset_path}")
    print(f"[INFO] Spawn: ({args.spawn_x:.3f}, {args.spawn_y:.3f}, {args.spawn_z:.3f}), yaw={args.spawn_yaw:.1f}deg")
    print(
        f"[INFO] Route {args.route}: "
        + " -> ".join(f"({x:.2f}, {y:.2f})" for x, y in waypoints)
    )
    print(f"[INFO] Clear cruise: {command_limits.max_forward_mps:.2f}m/s")
    print("[INFO] RGB safety colors: dark + green + yellow; source=camera_head:7072")
    print(
        "[INFO] First-person window: "
        + (
            f"ON (browser, http://127.0.0.1:{args.camera_window_port})"
            if args.camera_window
            else "OFF"
        )
    )
    print(f"[INFO] Inspection photo: {output.expanduser().resolve()}")
    print(
        f"[INFO] Runtime window: {args.num_steps} control steps ~= {args.num_steps / config.timing.locomotion_hz:.1f}s"
    )

    publish_layout_with_g1(
        grpc_addr=args.addr,
        layout_path=args.layout,
        g1_actor_name=config.scene.robot_actor_name,
        g1_asset_path=config.scene.robot_asset_path,
        g1_position_xyz=(args.spawn_x, args.spawn_y, args.spawn_z),
        g1_rotation_wxyz=_yaw_quaternion_wxyz(args.spawn_yaw),
        include_layout_actors=False,
    )
    if args.publish_wait > 0.0:
        print(f"[INFO] G1 已注册；等待服务重启 {args.publish_wait:.1f}s...")
        time.sleep(args.publish_wait)

    env = _create_env_with_retry(
        environment_class,
        timeout_s=args.startup_timeout,
        frame_skip=frame_skip,
        orcagym_addr=args.addr,
        agent_names=[config.scene.robot_actor_name],
        time_step=time_step,
        model_xml_path=model_xml,
        waypoints_xy_world=waypoints,
        camera_config=config.camera,
        timing_config=config.timing,
        command_limits=command_limits,
        navigation_config=config.navigation,
        navigator=navigator,
        sample_path=output.expanduser().resolve(),
        show_camera_window=args.camera_window,
        camera_window_port=args.camera_window_port,
    )
    verifier = verifier_class(f"Demo 1: G1 RGB {args.route} 路线巡检")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
