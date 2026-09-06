"""Demo 1: cross a green table obstacle with RGB-guided G1 navigation.

Pipeline:
    -> point-goal command -> green-table avoidance correction
    -> frozen locomotion ONNX ->  G1 actuators control
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = PROJECT_ROOT.parent / "green_table.json"
DEFAULT_SAMPLE_PATH = Path("/tmp/g1_green_table_goal_rgb.png")
GREEN_TABLE_XY = (4.7, -5.0)
DEFAULT_GOAL_XY = (4.7, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="green_table：G1 从绿色桌子南侧出发，视觉绕行后到达北侧点目标")
    parser.add_argument("--addr", default="127.0.0.1:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_LAYOUT_PATH,
        help="当前手动打开的 Layout 路径，仅作提示，不加载或修改场景",
    )
    parser.add_argument("--actor-name", default="g1", help="Layout 中已有 G1 的 Actor 名称")
    parser.add_argument("--rgb-port", type=int, default=7070, help="UI 头部相机 ColorPort")
    parser.add_argument("--goal-x", type=float, default=DEFAULT_GOAL_XY[0])
    parser.add_argument("--goal-y", type=float, default=DEFAULT_GOAL_XY[1])
    parser.add_argument("--num-steps", type=int, default=4000, help="50 Hz 控制步数")
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_PATH)
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
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 固定演示条件：起点在桌子南侧，目标点在桌子北侧。直线路径会
    #    穿过绿色桌子，因此只有 RGB 真正介入才能完成任务。
    # ------------------------------------------------------------------
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if not args.actor_name.strip():
        raise ValueError("--actor-name must not be empty")
    if not 1 <= args.rgb_port <= 65535:
        raise ValueError("--rgb-port must be in [1, 65535]")
    goal_xy = (float(args.goal_x), float(args.goal_y))
    (
        config,
        environment_class,
        navigator_class,
        frame_skip,
        model_xml,
        time_step,
        verifier_class,
    ) = _load_components()
    camera_config = replace(config.camera, rgb_port=args.rgb_port)
    navigator = navigator_class(
        navigation=config.navigation,
        limits=config.command_limits,
        visual=config.visual_avoidance,
    )
    print("[INFO] 运行 Demo：G1 绿色桌子定点导航与视觉避障")
    print(f"[INFO] 场景布局文件: {args.layout}")
    print(
        f"[INFO] Runtime window: {args.num_steps} control steps ~= {args.num_steps / config.timing.locomotion_hz:.1f}s"
    )

    # ------------------------------------------------------------------
    # 2. 进入闭环：已有 G1 的 50 Hz 运控每步执行，先站立推进仿真。
    #    首帧到达后 RGB/位姿生成上层速度指令，
    #    最后由冻结的 ONNX 策略转换为 G1 的关节控制。
    # ------------------------------------------------------------------
    env = environment_class(
        frame_skip=frame_skip,
        orcagym_addr=args.addr,
        agent_names=[args.actor_name],
        time_step=time_step,
        model_xml_path=model_xml,
        goal_xy_world=goal_xy,
        camera_config=camera_config,
        timing_config=config.timing,
        command_limits=config.command_limits,
        navigation_config=config.navigation,
        navigator=navigator,
        sample_path=args.output.expanduser().resolve(),
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
