"""Demo 1: cross a green table obstacle with RGB-guided G1 navigation.

Pipeline:
    open Layout -> AddActor(g1_navi) -> activate camera_head:7072
    -> point-goal command -> green-table avoidance correction
    -> frozen locomotion ONNX -> mixed-actuator G1 control
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = PROJECT_ROOT.parent / "demo1_v1.json"
DEFAULT_SAMPLE_PATH = Path("/tmp/g1_green_table_goal_rgb.png")
DEFAULT_SPAWN_XYZ = (4.7, -8.0, 0.0)
GREEN_TABLE_XY = (4.7, -5.0)
DEFAULT_GOAL_XY = (4.7, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="demo1_v1：G1 从绿色桌子南侧出发，视觉绕行后到达北侧点目标"
    )
    parser.add_argument("--addr", default="127.0.0.1:50051", help="OrcaLab gRPC 地址")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--spawn-x", type=float, default=DEFAULT_SPAWN_XYZ[0])
    parser.add_argument("--spawn-y", type=float, default=DEFAULT_SPAWN_XYZ[1])
    parser.add_argument("--spawn-z", type=float, default=DEFAULT_SPAWN_XYZ[2])
    parser.add_argument("--spawn-yaw", type=float, default=90.0)
    parser.add_argument("--goal-x", type=float, default=DEFAULT_GOAL_XY[0])
    parser.add_argument("--goal-y", type=float, default=DEFAULT_GOAL_XY[1])
    parser.add_argument("--num-steps", type=int, default=4000, help="50 Hz 控制步数")
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument(
        "--camera-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="显示 camera_head 的浏览器第一视角窗口（默认关闭）",
    )
    parser.add_argument(
        "--camera-window-port",
        type=int,
        default=8765,
        help="第一视角浏览器窗口的本地端口（默认 8765）",
    )
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
    from layout_scene import publish_layout_with_g1
    from online_verifier import OnlineVerifier

    from envs.euler.g1_vision_nav.config import DEFAULT_CONFIG
    from envs.euler.g1_vision_nav.g1_green_table_env import G1GreenTableEnv
    from envs.euler.g1_vision_nav.green_table_navigator import GreenTableNavigator

    return (
        DEFAULT_CONFIG,
        G1GreenTableEnv,
        GreenTableNavigator,
        G1_FRAME_SKIP,
        G1_MODEL_XML,
        G1_TIME_STEP,
        OnlineVerifier,
        publish_layout_with_g1,
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 固定演示条件：起点在桌子南侧，目标点在桌子北侧。直线路径会
    #    穿过绿色桌子，因此只有 RGB 真正介入才能完成任务。
    # ------------------------------------------------------------------
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if not 0 <= args.camera_window_port <= 65535:
        raise ValueError("--camera-window-port must be in [0, 65535]")
    goal_xy = (float(args.goal_x), float(args.goal_y))
    (
        config,
        environment_class,
        navigator_class,
        frame_skip,
        model_xml,
        time_step,
        verifier_class,
        publish_scene,
    ) = _load_components()
    navigator = navigator_class(
        navigation=config.navigation,
        limits=config.command_limits,
        visual=config.visual_avoidance,
    )

    print("[INFO] 保留 OrcaLab 当前运行的 demo1_v1 Layout，只发布代码生成的 G1。")
    print(f"[INFO] Layout: {args.layout.expanduser().resolve()}")
    print(f"[INFO] G1: {config.scene.robot_asset_path}")
    print(
        f"[INFO] Spawn: ({args.spawn_x:.2f}, {args.spawn_y:.2f}, {args.spawn_z:.2f}), "
        f"yaw={args.spawn_yaw:.1f}deg"
    )
    print(f"[INFO] Green table: ({GREEN_TABLE_XY[0]:.2f}, {GREEN_TABLE_XY[1]:.2f})")
    print(f"[INFO] Opposite-side goal: ({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")
    print("[INFO] Perception: camera_head RGB 7072 / green-table detector")
    print(
        "[INFO] First-person window: "
        + (
            f"ON (browser, http://127.0.0.1:{args.camera_window_port})"
            if args.camera_window
            else "OFF"
        )
    )
    print(
        f"[INFO] Runtime window: {args.num_steps} control steps ~= {args.num_steps / config.timing.locomotion_hz:.1f}s"
    )

    # ------------------------------------------------------------------
    # 2. 只通过 AddActor 发布 G1。官方 Lesson 8 说明：AddActor 会把
    #    CameraCaptureComponent 注册到 Studio，随后才能激活 7072 RGB。
    #    手动打开的 demo1_v1 环境不会被重新发布。
    # ------------------------------------------------------------------
    actor_count = publish_scene(
        grpc_addr=args.addr,
        layout_path=args.layout,
        g1_actor_name=config.scene.robot_actor_name,
        g1_asset_path=config.scene.robot_asset_path,
        g1_position_xyz=(args.spawn_x, args.spawn_y, args.spawn_z),
        g1_rotation_wxyz=_yaw_quaternion_wxyz(args.spawn_yaw),
        include_layout_actors=False,
    )
    print(f"[INFO] 已发布 G1（脚本重建 Layout actor 数={actor_count}）；等待服务重启……")
    time.sleep(3.0)

    # ------------------------------------------------------------------
    # 3. 进入闭环：50 Hz 运控每步执行，RGB/位姿生成上层速度指令，
    #    最后由冻结的 ONNX 策略转换为 G1 的关节控制。
    # ------------------------------------------------------------------
    env = environment_class(
        frame_skip=frame_skip,
        orcagym_addr=args.addr,
        agent_names=[config.scene.robot_actor_name],
        time_step=time_step,
        model_xml_path=model_xml,
        goal_xy_world=goal_xy,
        camera_config=config.camera,
        timing_config=config.timing,
        command_limits=config.command_limits,
        navigation_config=config.navigation,
        navigator=navigator,
        sample_path=args.output.expanduser().resolve(),
        show_camera_window=args.camera_window,
        camera_window_port=args.camera_window_port,
    )
    verifier = verifier_class("Demo 1 v1: G1 绿色桌子定点导航与视觉避障")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
