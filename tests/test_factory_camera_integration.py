"""Factory UI camera/preview contracts, without connecting to the simulator."""

import importlib.util
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from urllib.request import urlopen

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/euler/07_locomotion"))
sys.path.insert(0, str(ROOT / "examples/euler/g1_vision_nav"))

from envs.euler.g1_vision_nav.camera_stream import CameraFrame, CameraNotReadyError, CameraPreviewWindow, G1CameraStreams
from envs.euler.g1_vision_nav.config import CameraConfig
from envs.euler.g1_vision_nav.g1_camera_stream_env import G1CameraStreamEnv
from envs.euler.g1_vision_nav.g1_pick_locomotion_env import G1PickLocomotionEnv
from envs.euler.g1_vision_nav.g1_vision_nav_env import G1VisionNavEnv


def test_navigation_observation_aims_at_current_waypoint_not_path_projection(monkeypatch):
    from types import SimpleNamespace

    from envs.euler.g1_vision_nav.command_bridge import CommandLimiter
    from envs.euler.g1_vision_nav.config import NavigationConfig
    from envs.euler.g1_vision_nav.point_goal_navigator import WaypointRoute

    env = object.__new__(G1VisionNavEnv)
    env.route = WaypointRoute(((5., 0.), (5., 5.)), waypoint_tolerance_m=.4,
                             passage_tolerance_m=.4, goal_tolerance_m=.2)
    env.route.update((0., 0.))
    env.navigation_config = NavigationConfig()
    env.camera_config = CameraConfig()
    env.command_limiter = CommandLimiter()
    env._latest_frame = CameraFrame(np.full((60, 90, 3), 120, np.uint8), 1)
    monkeypatch.setattr(G1VisionNavEnv, 'data', property(lambda self: SimpleNamespace(time=1.)))
    env._read_planar_state = lambda: {
        'pelvis_position_world': np.array([2., 1., .75]), 'base_heading_rad': 0.,
        'goal_xy_robot_m': np.array([3., -1.]), 'base_velocity_xy_mps': np.zeros(2),
        'base_yaw_rate_rps': 0.,
    }
    obs = env._build_navigation_observation()
    np.testing.assert_allclose(obs.goal_xy_robot_m, [3., -1.])
    assert env.route.current_index == 0


@pytest.fixture
def factory_entry():
    spec = importlib.util.spec_from_file_location("factory_entry", ROOT / "examples/euler/g1_vision_nav/run_factory_navi.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def test_entry_defaults_and_no_spawn(monkeypatch, factory_entry):
    entry = factory_entry
    monkeypatch.setattr(sys, "argv", ["factory"])
    args = entry.parse_args()
    assert args.actor_name == "g1"
    assert args.rgb_port == 7070
    assert not hasattr(args, "depth_port")
    assert args.camera_window is True
    assert not hasattr(args, "spawn_x")
    assert "publish_layout_with_g1" not in Path(entry.__file__).read_text()
    assert args.max_speed == 1.2
    assert args.layout.name == "factory_navi.json"
    assert args.output is None


@pytest.mark.parametrize("route, output", [
    ("left", None), ("right", None),
    ("left", "photos/test.png"), ("left", "/tmp/custom-inspection.png"),
])
def test_entry_wires_speed_camera_route_and_photo(monkeypatch, tmp_path, capsys, factory_entry, route, output):
    from envs.euler.g1_vision_nav.config import DEFAULT_CONFIG
    entry = factory_entry
    argv = ["factory", "--route", route]
    if output is not None:
        argv += ["--output", output]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)
    env = Mock()
    env.run_lesson.return_value = {"summary": {"all_passed": True}}
    create_env = Mock(return_value=env)
    monkeypatch.setattr(entry, "_create_env_with_retry", create_env)
    monkeypatch.setattr(entry, "_load_components", lambda: (
        DEFAULT_CONFIG, Mock(), Mock(), 20, "model.xml", .001, Mock(),
    ))
    entry.main()
    kwargs = create_env.call_args.kwargs
    assert kwargs["command_limits"].max_forward_mps == 1.2
    assert kwargs["agent_names"] == ["g1"]
    assert kwargs["camera_config"].rgb_port == 7070
    assert kwargs["camera_config"].depth_port is None
    assert not kwargs["camera_config"].enable_depth_preview
    assert kwargs["waypoints_xy_world"] == entry.ROUTES[route]
    expected_path = Path(output) if output is not None else Path(f"envs/euler/g1_vision_nav/g1_cabinet_{route}_rgb.png")
    if not expected_path.is_absolute():
        expected_path = ROOT / expected_path
    assert kwargs["sample_path"] == expected_path.resolve()
    printed = capsys.readouterr().out
    assert "场景布局文件：factory_navi.json" in printed
    assert "正上方俯视" in printed
    assert "+ 左转/逆时针" in printed
    assert "- 右转/顺时针" in printed
    assert "1.20 m/s" in printed
    if output is None:
        assert f"照片保存位置：envs/euler/g1_vision_nav/g1_cabinet_{route}_rgb.png" in printed
        assert "/tmp/g1_cabinet_" not in printed
    assert kwargs["show_camera_window"] is True
    env.close.assert_called_once()


def test_entry_rejects_speed_above_cruise_limit(monkeypatch, factory_entry):
    monkeypatch.setattr(sys, "argv", ["factory", "--max-speed", "1.21"])
    with pytest.raises(ValueError):
        factory_entry.main()


def test_preview_never_opens_browser_by_default():
    assert CameraPreviewWindow(enabled=True).open_browser is False


def test_real_entry_components_keep_environment_inheritance(factory_entry):
    from online_verifier import OnlineVerifier

    from envs.euler.g1_vision_nav.g1_vision_nav_env import FactoryReporter, G1FactoryInspectionEnv

    config, env_class, _, _, _, _, reporter = factory_entry._load_components()
    assert config.command_limits.max_forward_mps == 1.2
    assert env_class is G1FactoryInspectionEnv
    assert env_class.__mro__[:4] == (
        G1FactoryInspectionEnv, G1VisionNavEnv, G1CameraStreamEnv, G1PickLocomotionEnv,
    )
    assert reporter is FactoryReporter
    assert issubclass(reporter, OnlineVerifier)


def camera_env(monkeypatch):
    monkeypatch.setattr(G1PickLocomotionEnv, "verify_step", lambda *args: None)
    env = object.__new__(G1CameraStreamEnv)
    env.camera_config = CameraConfig()
    env._first_frame_index = -1
    env._sample_saved = False
    env._latest_frame = None
    env._camera_wait_started = time.monotonic()
    env._last_camera_wait_log_step = None
    env._camera_streams = Mock()
    env._camera_preview = Mock()
    return env


def test_fresh_frame_shared_by_preview_and_navigation_then_outage(monkeypatch):
    env = camera_env(monkeypatch)
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3), 1, time.monotonic())
    env._camera_preview_lines = lambda f: [f"frame={f.index}"]
    env._camera_streams.get_rgb.return_value = frame
    verifier = Mock()
    env.verify_step(1, verifier)
    assert env.camera_motion_ready
    assert env._latest_frame is frame
    assert env._camera_preview.show.call_args.args[0] is frame.image
    env._camera_streams.get_depth_preview.assert_not_called()
    env._camera_streams.get_rgb.side_effect = CameraNotReadyError("offline")
    env.verify_step(2, verifier)
    assert not env.camera_motion_ready
    assert "CAMERA OFFLINE" in env._camera_preview.show.call_args.args[1][0]
    for step in range(3, 20):
        env.verify_step(step, verifier)
    assert sum(c.args[0] == "camera_waiting" for c in verifier.observe.call_args_list) == 1


def test_depth_preview_keeps_rgb_navigation_and_inspection_photo(monkeypatch, tmp_path):
    from PIL import Image

    env = camera_env(monkeypatch)
    env.camera_config = replace(env.camera_config, depth_port=7071, enable_depth_preview=True)
    rgb = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3), 2, time.monotonic())
    depth = CameraFrame(np.full((3, 4, 3), 20, dtype=np.uint8), 7, time.monotonic())
    env._camera_preview_lines = lambda f: [f"RGB frame={f.index}"]
    env._camera_streams.get_rgb.return_value = rgb
    env._camera_streams.get_depth_preview.return_value = depth
    env.sample_path = tmp_path / "inspection.png"
    env.verify_step(1, Mock())
    assert env._latest_frame is rgb and env.camera_motion_ready
    shown, lines = env._camera_preview.show.call_args.args
    np.testing.assert_array_equal(shown, depth.image * 6)
    assert any("DEPTH" in line and "7" in line for line in lines)
    assert any("not meters" in line for line in lines)
    env.after_loop(Mock())
    np.testing.assert_array_equal(np.asarray(Image.open(env.sample_path)), rgb.image)

    # 深度断流仅影响预览，不能把旧深度或 RGB 伪装成实时深度。
    env._camera_streams.get_depth_preview.side_effect = CameraNotReadyError("stale")
    env.verify_step(2, Mock())
    shown, lines = env._camera_preview.show.call_args.args
    assert env.camera_motion_ready
    assert not shown.any()
    assert any("DEPTH OFFLINE" in line for line in lines)

    # RGB 断流仍必须停走，但实时深度预览应继续更新。
    env._camera_streams.get_depth_preview.side_effect = None
    env._camera_streams.get_rgb.side_effect = CameraNotReadyError("offline")
    env.verify_step(3, Mock())
    shown, lines = env._camera_preview.show.call_args.args
    assert not env.camera_motion_ready
    np.testing.assert_array_equal(shown, depth.image * 6)
    assert any("RGB OFFLINE" in line for line in lines)


def test_depth_snapshot_preserves_rgb_channels_and_rejects_stale_frames():
    streams = G1CameraStreams(CameraConfig(depth_port=7071, enable_depth_preview=True))
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    receiver = Mock()
    streams._depth_camera = receiver
    receiver.snapshot.return_value = CameraFrame(image, 1, time.monotonic())
    np.testing.assert_array_equal(streams.get_depth_preview().image, image)
    receiver.snapshot.return_value = CameraFrame(image, 1, time.monotonic() - 10)
    with pytest.raises(CameraNotReadyError, match="stale"):
        streams.get_depth_preview()


def test_depth_preparation_skipped_between_preview_frames(monkeypatch):
    env = camera_env(monkeypatch)
    env.camera_config = replace(env.camera_config, depth_port=7071, enable_depth_preview=True)
    env._camera_preview.is_due = False
    env._show_camera_preview()
    env._camera_streams.get_depth_preview.assert_not_called()
    env._camera_preview.show.assert_not_called()


def test_preview_due_is_non_consuming_and_disabled_after_close(monkeypatch):
    preview = CameraPreviewWindow(enabled=True)
    monkeypatch.setattr("envs.euler.g1_vision_nav.camera_stream.time.monotonic", lambda: 10.0)
    preview._next_encode_time = 10.1
    assert not preview.is_due
    preview._next_encode_time = 10.0
    assert preview.is_due and preview.is_due
    preview.close()
    assert not preview.is_due


def test_both_camera_receivers_stop_even_if_rgb_stop_fails():
    streams = G1CameraStreams()
    streams._started = True
    streams._rgb_camera = Mock()
    streams._depth_camera = Mock()
    streams._rgb_camera.stop.side_effect = RuntimeError("RGB stop failed")
    with pytest.raises(RuntimeError, match="RGB stop failed"):
        streams.stop()
    streams._depth_camera.stop.assert_called_once()


@pytest.mark.parametrize("active", [True, False])
def test_capture_ownership_does_not_override_ui(monkeypatch, tmp_path, active):
    env = object.__new__(G1CameraStreamEnv)
    env._owns_capture = False
    env.get_current_frame = Mock(side_effect=[12] if active else [-1, 0])
    env.begin_save_video = Mock()
    env.stop_save_video = Mock()
    env.set_camera_sensor_info = Mock(side_effect=AssertionError("must not configure UI"))
    env._camera_streams = Mock()
    env._camera_preview = Mock()
    monkeypatch.setattr("envs.euler.g1_vision_nav.g1_camera_stream_env.tempfile.mkdtemp", lambda **kw: str(tmp_path))
    monkeypatch.setattr(G1PickLocomotionEnv, "close", lambda self: None)
    env._start_capture(Mock())
    assert env.begin_save_video.call_count == int(not active)
    env.close()
    assert env.stop_save_video.call_count == int(not active)
    env.set_camera_sensor_info.assert_not_called()


def test_missing_frame_stops_before_any_route_or_pose_access():
    env = object.__new__(G1VisionNavEnv)
    env.camera_config = CameraConfig()
    env._latest_frame = None
    env._goal_reached = False
    env._startup_steps = 0
    env._emergency_stop = False
    command = env._requested_command(20)
    assert not command.walk_enabled
    assert command.forward_mps == command.lateral_mps == command.yaw_rate_rps == 0


def test_real_http_preview_serves_jpeg_without_opening_browser(monkeypatch):
    opened = Mock()
    monkeypatch.setattr("envs.euler.g1_vision_nav.camera_stream.webbrowser.open_new", opened)
    rgb = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
    before = rgb.copy()
    preview = CameraPreviewWindow(enabled=True, port=0, width=32, height=24)
    try:
        preview.start()
        preview.show(rgb, ["g1 | cmd=0.10"])
        with urlopen(preview.url, timeout=2) as page:
            assert b"/stream.mjpg" in page.read()
        # 两次连接模拟浏览器关闭后重开；只关闭客户端不关闭相机/服务器。
        for _ in range(2):
            with urlopen(preview.url + "/stream.mjpg", timeout=2) as stream:
                assert b"Content-Type: image/jpeg" in stream.read(128)
        assert preview.enabled and preview.has_frame
        np.testing.assert_array_equal(rgb, before)
        opened.assert_not_called()
    finally:
        preview.close()
    assert preview.url is None


@pytest.mark.parametrize("fail_receiver", [False, True])
def test_cleanup_always_closes_capture_and_parent(monkeypatch, fail_receiver):
    env = object.__new__(G1CameraStreamEnv)
    events = []
    def stop_receiver():
        events.append("receiver")
        if fail_receiver:
            raise RuntimeError("receiver stop")
    env._camera_preview = Mock()
    env._camera_streams = Mock(stop=stop_receiver)
    env._owns_capture = True
    env.stop_save_video = lambda: events.append("capture")
    monkeypatch.setattr(G1PickLocomotionEnv, "close", lambda self: events.append("parent"))
    if fail_receiver:
        with pytest.raises(RuntimeError, match="receiver stop"):
            env.close()
    else:
        env.close()
    assert events == ["receiver", "capture", "parent"]
    env._camera_preview.close.assert_called_once()


def test_reporter_keeps_checks_without_success_spam(capsys, tmp_path):
    from envs.euler.g1_vision_nav.g1_vision_nav_env import FactoryReporter
    reporter = FactoryReporter("test", report_dir=str(tmp_path))
    for step in range(50):
        reporter.check(f"height_{step}", True, np.float32(0.8), "正常", "高度正常")
    assert capsys.readouterr().out == ""
    reporter.check("height_50", False, 0.2, "正常", "高度异常")
    reporter.check("height_100", False, 0.2, "正常", "高度异常")
    assert capsys.readouterr().out.count("[异常]") == 1
    reporter.check("height_150", True, 0.8, "正常", "高度正常")
    assert "[恢复]" in capsys.readouterr().out
    assert len(reporter.checks) == 53
    assert reporter.report()["summary"]["failed"] == 2


@pytest.mark.parametrize("rate, expected", [
    (.65, "+0.65 rad/s（左转/逆时针）"),
    (-.65, "-0.65 rad/s（右转/顺时针）"),
    (0., "+0.00 rad/s（不转向）"),
])
def test_turn_rate_log_explains_sign_and_unit(rate, expected):
    from envs.euler.g1_vision_nav.g1_vision_nav_env import format_turn_rate
    assert format_turn_rate(rate) == expected


def test_simulation_gait_clock_ignores_wall_clock_and_pauses_while_standing() -> None:
    from envs.euler.g1_vision_nav.g1_pick_locomotion_env import SimulationGaitClock

    clock = SimulationGaitClock(gait_period_s=0.8)

    assert clock.update(0.0, walking=False) == 0.0
    assert clock.update(1.0, walking=False) == 0.0
    assert np.isclose(clock.update(1.2, walking=True), 0.25)
    assert np.isclose(clock.update(1.2, walking=True), 0.25)
    assert np.isclose(clock.update(1.4, walking=True), 0.50)


def test_factory_arrival_waits_for_blue_and_saves_confirmed_frame_once(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from PIL import Image

    from envs.euler.g1_vision_nav.command_bridge import CommandLimiter
    from envs.euler.g1_vision_nav.config import NavigationConfig
    from envs.euler.g1_vision_nav.factory_inspection_navigator import BluePanelInspection
    from envs.euler.g1_vision_nav.g1_vision_nav_env import G1FactoryInspectionEnv
    from envs.euler.g1_vision_nav.point_goal_navigator import WaypointRoute

    env = object.__new__(G1FactoryInspectionEnv)
    env.inspection = BluePanelInspection()
    env.navigation_config = NavigationConfig()
    env.camera_config = CameraConfig()
    env.command_limiter = CommandLimiter()
    env.route = WaypointRoute(((19.2, 4.3),), waypoint_tolerance_m=.4,
                             passage_tolerance_m=.4, goal_tolerance_m=.2)
    env.route.update((19.2, 4.3))
    env._goal_reached = env._sample_saved = env._emergency_stop = env._terminal_safety_stop = False
    env._startup_steps = 0
    env._inspection_last_log = None
    env.locomotion = Mock()
    env.sample_path = tmp_path / 'cabinet.png'
    sim = SimpleNamespace(time=0.)
    monkeypatch.setattr(G1FactoryInspectionEnv, 'data', property(lambda self: sim))
    state = {'pelvis_position_world': np.array([19.2, 4.3, .75]),
             'base_heading_rad': 0., 'base_velocity_xy_mps': np.zeros(2), 'base_yaw_rate_rps': 0.}
    env._read_planar_state = lambda: state
    rgb = np.full((200, 300, 3), 120, np.uint8)
    rgb[45:115, 175:265] = (25, 65, 170)  # 面板偏右也拍，不再为了居中转向。
    env._latest_frame = CameraFrame(rgb, 1, time.monotonic())
    assert not env._requested_command(5).walk_enabled
    assert not env._goal_reached
    assert not env._should_save_rgb_sample()
    state['pelvis_position_world'][0] += .44  # 已到点后转向漂移，不能永久禁拍。
    assert env.arrival_confirmed
    for i in range(1, 8):
        sim.time = i * .1
        env._latest_frame = CameraFrame(rgb, i + 1, time.monotonic())
        assert not env._requested_command(5 + i * 5).walk_enabled
    assert env.inspection.ready
    verifier = Mock()
    monkeypatch.setattr(G1VisionNavEnv, 'verify_step', lambda *args: None)
    env.verify_step(100, verifier)
    for step in (200, 300, 400):
        env.verify_step(step, verifier)
    prompts = [call.args[1] for call in verifier.observe.call_args_list]
    assert prompts[-1] == "[巡检] 电气柜照片已保存，巡检完成。"
    assert len(prompts) == 2  # 一条照片路径、一条成功结果，不刷取景中间状态。
    assert not any('蓝色' in text or '停车区' in text or '占比' in text for text in prompts)
    assert env._sample_saved and env._goal_reached
    env.locomotion.set_commands.assert_called_once_with(lin_vel=(0., 0.), ang_vel=0., stand=0)
    assert not env.command_limiter.previous.walk_enabled
    # 拍照后即使图像丢失或安全状态变化，也不再产生行走/搜索指令。
    for step in (50, 500, 5000):
        command = env._requested_command(step)
        assert not command.walk_enabled
        np.testing.assert_array_equal(command.as_array(), [0., 0., 0.])
    np.testing.assert_array_equal(np.asarray(Image.open(env.sample_path)), rgb)
    # 结束时断流也不能把已经拍好的照片改成别的帧或覆盖。
    env._latest_frame = None
    env._camera_streams = Mock()
    env._camera_streams.get_rgb.side_effect = CameraNotReadyError('offline after capture')
    G1CameraStreamEnv.after_loop(env, Mock())
    env._camera_streams.get_rgb.assert_not_called()
    np.testing.assert_array_equal(np.asarray(Image.open(env.sample_path)), rgb)


def test_factory_photo_not_saved_during_safety_stop_or_on_write_failure(tmp_path):
    from envs.euler.g1_vision_nav.factory_inspection_navigator import BluePanelInspection
    from envs.euler.g1_vision_nav.g1_vision_nav_env import G1FactoryInspectionEnv

    env = object.__new__(G1FactoryInspectionEnv)
    env.inspection = BluePanelInspection()
    env.inspection.ready = True
    env.inspection.photo_rgb = np.zeros((20, 20, 3), np.uint8)
    env.sample_path = tmp_path / 'photo.png'
    env._sample_saved = env._goal_reached = env._terminal_safety_stop = False
    env._emergency_stop = True
    env._save_inspection_photo(Mock())
    assert not env.sample_path.exists() and not env._goal_reached
    env._emergency_stop = False
    env._save_rgb_image = Mock(side_effect=OSError('disk full'))
    with pytest.raises(OSError, match='disk full'):
        env._save_inspection_photo(Mock())
    assert not env._goal_reached and not env._sample_saved
