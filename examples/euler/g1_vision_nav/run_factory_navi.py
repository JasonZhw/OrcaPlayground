"""Run the G1 waypoint-based factory inspection Demo.

Pipeline:
    open factory_navi Layout with g1 -> connect UI camera -> follow left/right waypoints
    -> apply camera_head RGB obstacle correction -> frozen locomotion ONNX
    -> reach the electrical cabinet -> save one inspection RGB image
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = PROJECT_ROOT.parent / "factory_navi.json"
DEFAULT_PHOTO_DIR = Path("envs/euler/g1_vision_nav")
INSPECTION_POINT_XY = (20.0, 3.8)
ROUTES = {
    "left": (
        (9.0, 3.5),
        (10.0, 3.5),
        (12.6, 7.5),
        (14.5, 7.8),
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
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH, help="场景路径，仅作提示；请在 UI 手动打开")
    parser.add_argument("--actor-name", default="g1", help="Layout 中的机器人名称")
    parser.add_argument("--rgb-port", type=int, default=7070, help="UI 头部相机 ColorPort")
    parser.add_argument("--route", choices=("left", "right"), default="left")
    parser.add_argument("--max-speed", type=float, default=1.2, help="空旷巡航速度上限，最大 1.2m/s（需实跑验证）")
    parser.add_argument("--num-steps", type=int, default=None, help="50 Hz 控制步数")
    parser.add_argument("--output", type=Path, default=None, help="巡检照片路径；相对路径按项目根目录解析")
    parser.add_argument(
        "--camera-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="提供网页 RGB 第一视角与信息叠加（默认开启，但不自动打开浏览器）",
    )
    parser.add_argument(
        "--camera-window-port",
        type=int,
        default=8765,
        help="第一视角浏览器窗口的本地端口（默认 8765）",
    )
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    return parser.parse_args()


def _load_components():
    root_string = str(PROJECT_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    lesson_dir = PROJECT_ROOT / "examples" / "euler" / "07_locomotion"
    lesson_dir_string = str(lesson_dir)
    if lesson_dir_string not in sys.path:
        sys.path.insert(0, lesson_dir_string)

    from g1_base_env import G1_FRAME_SKIP, G1_MODEL_XML, G1_TIME_STEP

    from envs.euler.g1_vision_nav.config import DEFAULT_CONFIG
    from envs.euler.g1_vision_nav.factory_inspection_navigator import (
        FactoryInspectionNavigator,
    )
    from envs.euler.g1_vision_nav.g1_vision_nav_env import (
        FactoryReporter,
        G1FactoryInspectionEnv,
    )

    return (
        DEFAULT_CONFIG,
        G1FactoryInspectionEnv,
        FactoryInspectionNavigator,
        G1_FRAME_SKIP,
        G1_MODEL_XML,
        G1_TIME_STEP,
        FactoryReporter,
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
            print("[连接] 仿真服务尚未就绪，2 秒后重试；请确认已选择 manual launch。")
            time.sleep(2.0)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Demo contract: choose one fixed world-frame route. Both routes end
    #    at the same cabinet inspection point; changing --route changes only
    #    the ordered waypoints, not the low-level locomotion policy.
    # ------------------------------------------------------------------
    args = parse_args()
    if args.num_steps is None:
        args.num_steps = DEFAULT_STEPS[args.route]
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if not 0.0 < args.max_speed <= 1.2:
        raise ValueError("--max-speed must be in (0, 1.2]")
    if args.startup_timeout <= 0.0:
        raise ValueError("启动等待时间必须大于 0")
    if not args.actor_name.strip():
        raise ValueError("Actor 名称不能为空")
    if not 0 <= args.camera_window_port <= 65535:
        raise ValueError("--camera-window-port must be in [0, 65535]")
    waypoints = ROUTES[args.route]
    output = (args.output or DEFAULT_PHOTO_DIR / f"g1_cabinet_{args.route}_rgb.png").expanduser()
    # 无论从哪个目录启动，都把相对照片路径解析到项目根目录。
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    (
        config,
        environment_class,
        navigator_class,
        frame_skip,
        model_xml,
        time_step,
        verifier_class,
    ) = _load_components()
    config = replace(config, camera=replace(
        config.camera, rgb_port=args.rgb_port, depth_port=None,
        enable_depth_preview=False,
    ))
    command_limits = replace(
        config.command_limits,
        max_forward_mps=float(args.max_speed),
    )
    navigator = navigator_class(
        navigation=config.navigation,
        limits=command_limits,
        visual=config.visual_avoidance,
    )

    print("运行Demo：G1 工厂定点导航")
    # 从配置读取文件名，不绑定用户目录；此提示不查询 UI 当前打开的场景。
    print(f"场景布局文件：{args.layout.name}")
    photo_display = output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output
    print(f"照片保存位置：{photo_display}")
    print("[转向] 从机器人正上方俯视：+ 左转/逆时针，- 右转/顺时针；rad 是角度差，rad/s 是转向速度。")
    print(
        f"[速度] 空旷上限 {command_limits.max_forward_mps:.2f} m/s；"
        f"轻度避障 {config.visual_avoidance.obstacle_approach_forward_mps:.2f}，"
        f"近障 {config.visual_avoidance.near_obstacle_forward_mps:.2f} m/s；急转弯仍限速。"
    )
    if not args.camera_window:
        print("网页预览已关闭；RGB 接收和巡检拍照继续启用。")

    # 只连接当前场景；机器人位置、朝向和相机由 UI 决定。
    # 同一份 RGB 供导航、拍照和网页第一视角；不接收深度，不修改 UI 相机开关。
    env = _create_env_with_retry(
        environment_class,
        timeout_s=args.startup_timeout,
        frame_skip=frame_skip,
        orcagym_addr=args.addr,
        agent_names=[args.actor_name],
        time_step=time_step,
        model_xml_path=model_xml,
        waypoints_xy_world=waypoints,
        camera_config=config.camera,
        timing_config=config.timing,
        command_limits=command_limits,
        navigation_config=config.navigation,
        navigator=navigator,
        sample_path=output,
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
