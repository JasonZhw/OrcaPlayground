"""Offline indexed-camera and raw episode integrity tests (no Studio)."""

import asyncio
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from examples.euler.medication_sorting.execution import Grippers
from examples.euler.medication_sorting.recording import CAMERAS, RawEpisodeRecorder
from examples.euler.medication_sorting.workflow import Trial


class Bridge:
    def __init__(self, names=CAMERAS):
        self.names = names
        self.properties = {
            name: SimpleNamespace(
                capture_rgb=False,
                capture_depth=True,
                capture_normal=False,
                capture_object_color=False,
                color_port=8000 + i,
                streaming_enabled=True,
                width=64,
                height=32,
                vertical_fov=60.0,
                near_clip=0.01,
                far_clip=5.0,
                gamma=1.0,
            )
            for i, name in enumerate(names)
        }

    async def get_camera_names(self):
        return list(self.names)

    async def get_camera_properties(self, name):
        return self.properties[name]

    async def set_camera_properties(self, name, **kwargs):
        assert not self.properties[name].streaming_enabled
        for key, value in kwargs.items():
            setattr(self.properties[name], key, value)

    async def set_streaming_enabled(self, name, enabled):
        self.properties[name].streaming_enabled = enabled


class Receiver:
    def __init__(self):
        self.entries = {}
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_frame(self, index):
        return self.entries.get(index)


def make_packet():
    pixels = np.random.default_rng(10).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height = 64, 32
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 50)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "0"}
    packets = list(codec.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")))
    packets.extend(codec.encode(None))
    return b"".join(bytes(p) for p in packets)


def supply(receivers, packet, index=100, timestamp=500_000_000):
    for receiver in receivers.values():
        receiver.entries[index] = SimpleNamespace(
            simulate_index=index,
            timestamp_ns=timestamp,
            is_keyframe=True,
            nal_data=packet,
        )


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bridge, self.loop = Bridge(), asyncio.new_event_loop()
        self.receivers = {name: Receiver() for name in CAMERAS}
        self.recorder = RawEpisodeRecorder(
            Path(self.temp.name) / "episode",
            timeout=0.01,
            receiver_factory=lambda name, port, size: self.receivers[name],
        )
        self.recorder.start(self.bridge, self.loop)
        self.packet = make_packet()

    def tearDown(self):
        self.recorder.close(False, "fixture cleanup")
        self.loop.close()
        self.temp.cleanup()

    def test_native_video_dual_arm_arrays_and_metadata(self):
        recorder, bridge, receivers = self.recorder, self.bridge, self.receivers
        state, action = np.arange(30) * 0.01, np.arange(16) * 0.02
        for row in range(2):
            supply(receivers, self.packet, 100 + row, 500_000_000 + row * 20_000_000)
            recorder.append(state, action, 1.0 + row * 0.02, 100 + row, "pick")
        state[:] = 99  # Inputs must have been copied before later control mutation.
        result = recorder.close(True, "caller verified all placements")
        assert result["raw_valid"] and result["complete"]
        assert result["training_ready"] is False
        assert result["errors"] == []
        assert result["format"] == "medication_sorting_raw_v2"
        assert result["action_dim"] == 16
        assert result["action_names"][7] == "left_gripper_closure"
        assert result["gripper_mapping"]["version"] == "medication_closure_v1"
        with np.load(recorder.directory / "episode.npz", allow_pickle=False) as data:
            assert data["state"].shape == (2, 30)
            assert data["action"].shape == (2, 16)
            np.testing.assert_allclose(data["state"][0], np.arange(30) * 0.01)
            np.testing.assert_array_equal(data["frame_index"], [100, 101])
        for name in CAMERAS:
            camera = result["cameras"][name]
            assert camera["native_shape"] == [32, 64, 3]
            assert camera["decoded_frame_count"] == 2
            assert camera["resized"] is False
            entries = json.loads((recorder.directory / camera["frame_metadata"]).read_text())
            assert [e["frame_index"] for e in entries] == [100, 101]
            assert [e["timestamp_ns"] for e in entries] == [500_000_000, 520_000_000]
            assert bridge.properties[name].capture_rgb is False
            assert bridge.properties[name].capture_depth is True
            assert bridge.properties[name].streaming_enabled is True
            assert receivers[name].stopped
        assert recorder.close(False, "ignored second close") is result

    def test_missing_index_never_substitutes_latest_frame(self):
        recorder, receivers = self.recorder, self.receivers
        supply(receivers, self.packet, index=99)
        with self.assertRaisesRegex(TimeoutError, "Missing indexed RGB 100"):
            recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "pick")
        result = recorder.close(True, "caller thought successful")
        assert not result["raw_valid"] and not result["complete"]
        assert result["frame_count"] == 0
        assert any("Missing indexed RGB" in error for error in result["errors"])

    def test_trial_records_native_closure_before_physics_step(self):
        names, arms = [], {}
        for side, short in (("left", "l"), ("right", "r")):
            start = len(names)
            joints = [f"{side}_arm{i}" for i in range(7)]
            arms[side] = SimpleNamespace(joint_names=joints, qpos_ids=np.arange(start, start + 7))
            names.extend(joints)
            names.extend(f"g1_gripper_{short}_{part}_joint{j}" for part in ("inner", "outer") for j in (1, 3, 4, 2))
        measured = np.arange(30, dtype=float) / 100
        env = SimpleNamespace(
            joint_names=names,
            qids=np.arange(30),
            command=np.zeros(30),
            data=SimpleNamespace(qpos=measured.copy(), time=0.0),
            grippers=Grippers(names),
        )
        env.command[:7] = np.arange(7) / 10
        env.command[15:22] = np.arange(7) / 20
        env.grippers.set("left", 0.7)
        env.grippers.set("right", 1)
        executed = []

        def step(command):
            executed.append(command.copy())
            env.data.qpos[:] = 99
            env.data.time += 0.02

        async def render(*args, **kwargs):
            supply(self.receivers, self.packet, index=kwargs["simulate_index"])

        env.step = step
        scene = SimpleNamespace(arms=arms, bodies={}, set_qpos=lambda _: None)
        trial = Trial(env, scene, SimpleNamespace(render=render), self.loop, self.recorder.directory)
        trial.rec, trial.index = self.recorder, 100
        trial.tick(env.command, "hold")
        result = self.recorder.close(True, "fixture")
        assert result["raw_valid"]
        with np.load(self.recorder.directory / "episode.npz") as data:
            np.testing.assert_array_equal(data["state"][0], measured)
            np.testing.assert_allclose(data["action"][0], [*np.arange(7) / 10, 0.7, *np.arange(7) / 20, 1])
            assert data["sim_time"][0] == 0
        np.testing.assert_allclose(executed[0][7:15], [0, 0, 0.28, 0.9, 0, 0, -0.28, 0.9])
        np.testing.assert_allclose(executed[0][22:30], [0, 0, 0.28, 1.45, 0, 0, -0.28, 1.45])

    def test_native_recorder_rejects_legacy_actions_and_invalid_closure(self):
        supply(self.receivers, self.packet)
        with self.assertRaisesRegex(ValueError, "action16"):
            self.recorder.append(np.zeros(30), np.zeros(18), 0, 100, "hold")
        action = np.zeros(16)
        action[7] = 1.01
        with self.assertRaisesRegex(ValueError, "closure"):
            self.recorder.append(np.zeros(30), action, 0, 100, "hold")

    def test_invalid_native_frame_invalidates_episode(self):
        for field, value, match in (
            ("simulate_index", 999, "wrong frame index"),
            ("is_keyframe", False, "missing requested IDR"),
            ("timestamp_ns", -1, "invalid native timestamp"),
        ):
            with self.subTest(field=field):
                supply(self.receivers, self.packet)
                setattr(self.receivers["camera_left"].entries[100], field, value)
                with self.assertRaisesRegex(ValueError, match):
                    self.recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "pick")
        assert not self.recorder.close(True, "success requested")["raw_valid"]

    def test_repeated_native_timestamp_invalidates_episode(self):
        recorder, receivers = self.recorder, self.receivers
        supply(receivers, self.packet)
        recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "pick")
        supply(receivers, self.packet, index=101)
        with self.assertRaisesRegex(ValueError, "timestamp did not increase"):
            recorder.append(np.zeros(30), np.zeros(16), 0.02, 101, "pick")
        assert not recorder.close(True, "success requested")["raw_valid"]

    def test_irregular_control_time_invalidates_episode(self):
        recorder, receivers = self.recorder, self.receivers
        supply(receivers, self.packet)
        recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "pick")
        with self.assertRaisesRegex(ValueError, "advance by 1/fps"):
            recorder.append(np.zeros(30), np.zeros(16), 0.07, 101, "pick")
        assert not recorder.close(True, "success requested")["raw_valid"]

    def test_short_video_detected_on_close(self):
        recorder, receivers = self.recorder, self.receivers
        supply(receivers, self.packet)
        recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "pick")
        real_open = av.open

        class EmptyVideo:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, **kwargs):
                return iter(())

        with patch.object(
            av,
            "open",
            lambda path, *a, **kw: EmptyVideo() if "camera_left.mkv" in str(path) else real_open(path, *a, **kw),
        ):
            result = recorder.close(True, "success requested")
        assert not result["raw_valid"]
        assert any("count mismatch" in error for error in result["errors"])

    def test_physical_failure_keeps_diagnostic_video(self):
        recorder, receivers = self.recorder, self.receivers
        supply(receivers, self.packet)
        recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "pick")
        result = recorder.close(False, "box fell outside destination")
        assert not result["raw_valid"] and not result["training_ready"]
        assert result["frame_count"] == 1
        assert result["reason"] == "box fell outside destination"

    def test_single_object_text_does_not_advertise_four_object_variants(self):
        self.recorder.task = "Put the calcium bottle in the left purple bin."
        supply(self.receivers, self.packet)
        self.recorder.append(np.zeros(30), np.zeros(16), 0.0, 100, "place")
        result = self.recorder.close(True, "single object placed")
        assert result["task_variants"] == [self.recorder.task]
        assert result["cameras"]["head_cam"]["decoded_frame_count"] == 1

    def test_missing_camera_startup_can_be_finalized(self):
        loop = asyncio.new_event_loop()
        recorder = RawEpisodeRecorder(Path(self.temp.name) / "missing")
        try:
            with self.assertRaisesRegex(ValueError, "Missing native cameras"):
                recorder.start(Bridge(names=("head_cam",)), loop)
            result = recorder.close(False, "startup failed")
            assert not result["raw_valid"]
            assert (recorder.directory / "metadata.json").is_file()
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
