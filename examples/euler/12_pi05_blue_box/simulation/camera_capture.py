"""Capture matched camera frames at a requested simulation index."""
import time

from orca_gym.recorder.recording_task import SingleFrameTask


class Cameras:
    def __init__(self, env):
        self.env = env
        self.names = ["head_cam", "camera_right"]
        self.manager = env.get_recorder_manager()
        for i, name in enumerate(self.names):
            env.start_streaming(
                name, capture_rgb=True, capture_depth=False, width=640, height=480,
                color_port=7070 + i * 2, max_buffer_frames=2500,
            )
        self.index = 100000
        time.sleep(0.5)

    def capture(self):
        self.index += 1
        futures = {}
        for name in self.names:
            task = SingleFrameTask(
                self.index,
                lambda entry, rgb: (entry.simulate_index, None if rgb is None else rgb.copy()),
            )
            futures[name] = self.manager.submit_task(name, task)
        self.env.render(simulate_index=self.index, request_idr=True)
        result = {}
        for name, future in futures.items():
            value = future.result(timeout=8)
            if value is None or value[0] != self.index or value[1] is None:
                raise RuntimeError(f"Missing synchronized RGB frame for {name} at {self.index}")
            result[name] = value[1]
        return result

    def close(self):
        self.manager.stop_all()
