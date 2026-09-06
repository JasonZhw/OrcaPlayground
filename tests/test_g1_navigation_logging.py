"""CPU regression tests: readable logs must not throttle navigation safety."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def logging_env(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/euler/07_locomotion"))
    from online_verifier import OnlineVerifier

    from envs.euler.g1_vision_nav.command_bridge import CommandLimiter
    from envs.euler.g1_vision_nav.config import CameraConfig, NavigationConfig
    from envs.euler.g1_vision_nav.g1_camera_stream_env import G1CameraStreamEnv
    from envs.euler.g1_vision_nav.g1_vision_nav_env import G1VisionNavEnv

    parent_steps = []
    monkeypatch.setattr(G1CameraStreamEnv, "verify_step", lambda self, step, verifier: parent_steps.append(step))

    class LoggingEnv(G1VisionNavEnv):
        def __init__(self):
            # No simulator connection: only the state source and camera hook are fixtures.
            self.navigation_config = NavigationConfig()
            self.camera_config = CameraConfig()
            self.command_limiter = CommandLimiter()
            self.goal_xy_world = np.array([4.7, 0.0])
            self._navigation_stride = 1
            self._startup_steps = 100
            self._initial_distance_m = None
            self._minimum_distance_m = np.inf
            self._goal_reached = False
            self._emergency_stop = False
            self._unsafe_collision = None
            self._latest_frame = SimpleNamespace(index=1)
            self.height = 0.75
            self.distance = 5.0
            self.reads = 0

        @property
        def camera_motion_ready(self):
            return True

        def _read_planar_state(self):
            self.reads += 1
            return {
                "pelvis_position_world": np.array([4.7, -self.distance, self.height]),
                "goal_xy_robot_m": np.array([self.distance, 0.0]),
                "goal_bearing_rad": 0.0,
                "base_yaw_world_rad": np.pi / 2,
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
            }

    return LoggingEnv(), OnlineVerifier("navigation logging", report_dir=str(tmp_path)), parent_steps


def test_navigation_summary_is_chinese_and_prints_every_100_steps(logging_env, capsys):
    env, verifier, parent_steps = logging_env
    for step in range(101):
        env.verify_step(step, verifier)

    summaries = [item for item in verifier.observations if item["name"].startswith("navigation_state_")]
    assert [item["name"] for item in summaries] == ["navigation_state_0", "navigation_state_100"]
    assert env.reads == 101
    assert parent_steps == list(range(101))
    checks = [item for item in verifier.checks if item["name"].startswith("navigation_safe_")]
    assert len(checks) == 3  # Keep the existing check cadence, independently of summaries.
    output = capsys.readouterr().out
    assert "距目标" in output and "朝向" in output
    assert "前进" in output and "转向" in output and "rad/s" in output
    assert "position=" not in output


def test_emergency_is_reported_immediately_between_summaries(logging_env, capsys):
    env, verifier, _ = logging_env
    env.verify_step(0, verifier)
    capsys.readouterr()
    env.height = 0.4
    env.verify_step(1, verifier)
    output = capsys.readouterr().out
    assert "急停" in output
    assert "下一控制周期" in output
    assert not env._requested_command(101).walk_enabled
    env.verify_step(2, verifier)
    assert "急停" not in capsys.readouterr().out


def test_final_report_does_not_claim_collision_free_when_check_disabled(logging_env, capsys):
    env, verifier, _ = logging_env
    env.verify_step(0, verifier)
    capsys.readouterr()
    env.verify_final(verifier)
    output = capsys.readouterr().out
    assert "碰撞急停未启用" in output
    assert "未发生非足部环境碰撞" not in output


@pytest.fixture
def camera_logging_env(monkeypatch, tmp_path):
    import time

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/euler/07_locomotion"))
    from online_verifier import OnlineVerifier

    from envs.euler.g1_vision_nav.camera_stream import CameraFrame, CameraNotReadyError
    from envs.euler.g1_vision_nav.config import CameraConfig
    from envs.euler.g1_vision_nav.g1_camera_stream_env import G1CameraStreamEnv
    from envs.euler.g1_vision_nav.g1_pick_locomotion_env import G1PickLocomotionEnv

    parent_steps = []
    monkeypatch.setattr(G1PickLocomotionEnv, "verify_step", lambda self, step, verifier: parent_steps.append(step))

    class Stream:
        available = True
        reads = 0

        def get_rgb(self):
            self.reads += 1
            if not self.available:
                raise CameraNotReadyError("waiting for first frame")
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:12] = 255
            return CameraFrame(image, self.reads, time.monotonic())

    stream = Stream()

    class CameraLoggingEnv(G1CameraStreamEnv):
        def __init__(self):
            self.camera_config = CameraConfig()
            self._first_frame_index = -1
            self._latest_frame = None
            self._camera_wait_started = time.monotonic()
            self._last_camera_wait_log_step = None
            self._camera_streams = stream

    return CameraLoggingEnv(), OnlineVerifier("camera logging", report_dir=str(tmp_path)), stream, parent_steps


def test_camera_status_is_quiet_but_frames_are_read_every_step(camera_logging_env, capsys):
    env, verifier, stream, parent_steps = camera_logging_env
    for step in range(501):
        env.verify_step(step, verifier)
    checks = [item for item in verifier.checks if item["name"].startswith("rgb_frame_fresh_")]
    assert len(checks) == 3  # At 0, 250 and 500, rather than every 25 steps.
    assert stream.reads == 501
    assert parent_steps == list(range(501))
    assert env.camera_motion_ready
    output = capsys.readouterr().out
    assert "[相机] 已收到画面" in output
    assert "[相机] 图像正常" in output
    assert "fresh decoded RGB" not in output


def test_camera_waiting_reminder_is_throttled(camera_logging_env, capsys):
    env, verifier, stream, _ = camera_logging_env
    stream.available = False
    for step in range(501):
        env.verify_step(step, verifier)
    assert capsys.readouterr().out.count("[相机] 等待图像") == 3
    assert stream.reads == 501
    assert not env.camera_motion_ready


def test_camera_outage_and_recovery_are_reported_without_waiting(camera_logging_env, capsys):
    env, verifier, stream, _ = camera_logging_env
    env.verify_step(248, verifier)
    capsys.readouterr()
    stream.available = False
    env.verify_step(249, verifier)
    assert "[相机] 图像中断" in capsys.readouterr().out
    assert not env.camera_motion_ready
    env.verify_step(250, verifier)
    assert capsys.readouterr().out == ""
    stream.available = True
    env.verify_step(251, verifier)
    assert "[相机] 图像已恢复" in capsys.readouterr().out
    assert env.camera_motion_ready
    env.verify_step(252, verifier)
    assert capsys.readouterr().out == ""
    stream.available = False
    env.verify_step(253, verifier)
    assert "[相机] 图像中断" in capsys.readouterr().out


def test_demo_entry_keeps_rgb_options_without_browser_options(monkeypatch):
    from examples.euler.g1_vision_nav.run_green_table_crossing import parse_args

    monkeypatch.setattr("sys.argv", ["run_green_table_crossing.py"])
    args = parse_args()
    assert args.actor_name == "g1"
    assert args.rgb_port == 7070
    assert not hasattr(args, "camera_window")
    assert not hasattr(args, "camera_window_port")


@pytest.mark.parametrize("option", ["--camera-window", "--no-camera-window", "--camera-window-port=8765"])
def test_demo_entry_rejects_removed_browser_options(monkeypatch, option):
    from examples.euler.g1_vision_nav.run_green_table_crossing import parse_args

    monkeypatch.setattr("sys.argv", ["run_green_table_crossing.py", option])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize("owns_capture", [False, True])
@pytest.mark.parametrize("receiver_fails", [False, True])
def test_camera_cleanup_without_browser_preserves_capture_ownership(
    camera_logging_env, monkeypatch, owns_capture, receiver_fails,
):
    from envs.euler.g1_vision_nav.g1_pick_locomotion_env import G1PickLocomotionEnv

    env, _, stream, _ = camera_logging_env
    env._owns_capture = owns_capture
    events = []

    def stop_receiver():
        events.append("receiver")
        if receiver_fails:
            raise RuntimeError("receiver shutdown failed")

    monkeypatch.setattr(stream, "stop", stop_receiver, raising=False)
    monkeypatch.setattr(env, "stop_save_video", lambda: events.append("capture"))
    monkeypatch.setattr(G1PickLocomotionEnv, "close", lambda self: events.append("environment"))
    if receiver_fails:
        with pytest.raises(RuntimeError, match="receiver shutdown failed"):
            env.close()
    else:
        env.close()
    assert events == (["receiver", "capture", "environment"] if owns_capture else ["receiver", "environment"])


def test_camera_photo_still_saved_without_browser(camera_logging_env, tmp_path):
    from PIL import Image

    env, verifier, _, _ = camera_logging_env
    env.sample_path = tmp_path / "inspection.png"
    env.verify_step(0, verifier)
    env.after_loop(verifier)
    with Image.open(env.sample_path) as photo:
        assert photo.size == (32, 24)
        assert photo.mode == "RGB"
