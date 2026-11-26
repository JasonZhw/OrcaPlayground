"""Release server streams on normal shutdown and partial camera startup failure."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, call, patch


SOURCE = Path(__file__).resolve().parents[1] / "simulation/camera_capture.py"
spec = importlib.util.spec_from_file_location("camera_capture_tests", SOURCE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CameraLifecycleTests(unittest.TestCase):
    def test_close_disables_both_server_streams_before_stopping_receivers(self):
        env = Mock()
        manager = env.get_recorder_manager.return_value
        with patch.object(capture.time, "sleep"):
            cameras = capture.Cameras(env)
        cameras.close()
        self.assertEqual(manager.mock_calls, [
            call.set_streaming_enabled("head_cam", False),
            call.set_streaming_enabled("camera_right", False),
            call.stop_all(),
        ])

    def test_start_failure_stops_already_started_and_partial_streams(self):
        env = Mock()
        env.start_streaming.side_effect = [None, ConnectionError("camera startup failed")]
        with self.assertRaisesRegex(ConnectionError, "startup failed"):
            capture.Cameras(env)
        self.assertEqual(env.get_recorder_manager.return_value.mock_calls, [
            call.set_streaming_enabled("head_cam", False),
            call.set_streaming_enabled("camera_right", False),
            call.stop_all(),
        ])

    def test_close_attempts_other_camera_and_receiver_cleanup_after_error(self):
        env = Mock()
        manager = env.get_recorder_manager.return_value
        with patch.object(capture.time, "sleep"):
            cameras = capture.Cameras(env)
        manager.set_streaming_enabled.side_effect = [RuntimeError("server error"), None]
        with self.assertRaisesRegex(RuntimeError, "Failed to stop"):
            cameras.close()
        self.assertEqual(manager.set_streaming_enabled.call_count, 2)
        manager.stop_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
