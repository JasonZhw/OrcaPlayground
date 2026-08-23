"""Online RGB-camera validation while ``g1_pick_usda`` stands safely."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from envs.euler.g1_vision_nav.camera_preview import CameraPreviewWindow
from envs.euler.g1_vision_nav.camera_stream import (
    CameraFrame,
    CameraNotReadyError,
    G1CameraStreams,
)
from envs.euler.g1_vision_nav.config import CameraConfig
from envs.euler.g1_vision_nav.g1_pick_locomotion_env import G1PickLocomotionEnv


class G1CameraValidationEnv(G1PickLocomotionEnv):
    """Validate the head RGB stream without asking G1 to walk."""

    FRAME_CHECK_INTERVAL = 25
    VIDEO_DIR = Path("/tmp/g1_camera_validation_video")

    def __init__(
        self,
        *args,
        camera_config: CameraConfig,
        sample_path: Path,
        show_camera_window: bool = True,
        camera_window_port: int = 8765,
        **kwargs,
    ) -> None:
        self.camera_config = camera_config
        self.sample_path = sample_path
        self._camera_streams: G1CameraStreams | None = None
        self._first_frame_index = -1
        self._first_frame_step = -1
        self._last_frame_index = -1
        self._latest_frame: CameraFrame | None = None
        self._saving_video = False
        self._camera_preview = CameraPreviewWindow(
            enabled=show_camera_window,
            port=camera_window_port,
        )
        super().__init__(*args, **kwargs)

    def before_loop(self, verifier) -> None:
        """Activate Studio RGB capture before the first render cycle."""
        camera_count = int(self.model.model_info.get("ncam", 0))
        verifier.observe(
            "camera_backend",
            f"MuJoCo ncam={camera_count}; 继续验证 OrcaLab Studio 侧 Camera Component",
        )
        self._activate_camera_viewport(verifier)

        camera = self.camera_config
        self.locomotion.set_commands(
            stand=0,
            lin_vel=(0.0, 0.0),
            ang_vel=0.0,
        )
        self.set_camera_sensor_info(
            actor_name=self.agent_name,
            capture_rgb=True,
            capture_depth=False,
            save_mp4_file=True,
            use_dds=False,
            capture_normal=False,
            capture_object_color=False,
            is_recording=True,
            width=camera.width,
            height=camera.height,
            vertical_fov=camera.vertical_fov_deg,
        )

        os.makedirs(self.VIDEO_DIR, exist_ok=True)
        self.begin_save_video(str(self.VIDEO_DIR), capture_mode=0)
        self._saving_video = True
        sync_frame = self.get_current_frame()
        verifier.check(
            "studio_camera_sync_enabled",
            sync_frame >= 0,
            sync_frame,
            ">=0",
            "Studio CameraSyncManager 已开始产帧",
        )

        streams = G1CameraStreams(camera)
        streams.start()
        self._camera_streams = streams
        self._camera_preview.start()
        verifier.observe(
            "camera_stream_connecting",
            f"已启动 camera_head RGB 客户端：port={camera.rgb_port}；控制循环开始 render 后异步等待首帧",
        )

    def verify_step(self, step: int, verifier) -> None:
        """Reuse standing checks and periodically verify live RGB progression."""
        super().verify_step(step, verifier)
        if not self._poll_first_frame(verifier, step=step):
            return
        if step == self._first_frame_step:
            return

        frame = self._require_streams().get_rgb()
        self._latest_frame = frame
        self._camera_preview.show(frame.image, self._camera_preview_lines(frame))
        if step <= 0 or step % self.FRAME_CHECK_INTERVAL != 0:
            return
        verifier.check(
            f"rgb_frame_increasing_{step}",
            frame.index > self._last_frame_index,
            frame.index,
            f">{self._last_frame_index}",
            f"RGB 帧号持续增长（step={step}）",
        )
        self._last_frame_index = frame.index

    def observe_step(self, step: int, verifier) -> None:
        """Keep this test stationary; locomotion phases belong to the prior test."""
        return None

    def after_loop(self, verifier) -> None:
        """Validate the last frame and save one RGB sample for visual inspection."""
        try:
            if not self._poll_first_frame(verifier):
                verifier.check(
                    "rgb_first_frame_received",
                    False,
                    "not received",
                    f"frame within {self.camera_config.first_frame_timeout_s}s",
                    "仿真持续渲染后仍未收到 RGB 首帧",
                )
                return

            frame = self._require_streams().get_rgb()
            self._latest_frame = frame
            self._check_frame(frame, verifier, "last")
            verifier.check(
                "rgb_stream_progressed",
                frame.index > self._first_frame_index,
                frame.index,
                f">{self._first_frame_index}",
                "验证期间 RGB 流持续更新",
            )

            self.sample_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.image, mode="RGB").save(self.sample_path)
            verifier.check(
                "rgb_sample_saved",
                self.sample_path.is_file() and self.sample_path.stat().st_size > 100,
                str(self.sample_path),
                "non-empty PNG",
                "保存 RGB 样本",
            )
            verifier.observe(
                "rgb_sample_ready",
                f"相机样本已保存：{self.sample_path}",
            )
        finally:
            self._stop_video_recording()

    def close(self) -> None:
        """Stop decoder threads and disable Studio camera capture."""
        self._camera_preview.close()
        if self._camera_streams is not None:
            self._camera_streams.stop()
            self._camera_streams = None
        self._stop_video_recording()
        try:
            self.set_camera_sensor_info(
                actor_name=self.agent_name,
                capture_rgb=False,
                capture_depth=False,
                save_mp4_file=False,
                use_dds=False,
                is_recording=False,
            )
        except (AttributeError, RuntimeError):
            pass
        super().close()

    def _stop_video_recording(self) -> None:
        if not self._saving_video:
            return
        self.stop_save_video()
        self._saving_video = False

    def _check_frame(self, frame: CameraFrame, verifier, label: str) -> None:
        image = frame.image
        expected_shape = (
            self.camera_config.height,
            self.camera_config.width,
            3,
        )
        verifier.check(
            f"rgb_shape_{label}",
            image.shape == expected_shape,
            image.shape,
            expected_shape,
            f"RGB {label} 帧分辨率",
        )
        verifier.check(
            f"rgb_dtype_{label}",
            image.dtype == np.uint8,
            str(image.dtype),
            "uint8",
            f"RGB {label} 帧数据类型",
        )
        dynamic_range = int(image.max()) - int(image.min())
        verifier.check(
            f"rgb_nonempty_{label}",
            dynamic_range >= 10 and float(image.std()) >= 1.0,
            {"dynamic_range": dynamic_range, "std": float(image.std())},
            "dynamic_range>=10 and std>=1.0",
            f"RGB {label} 帧不是空白图",
        )

    def _poll_first_frame(self, verifier, *, step: int | None = None) -> bool:
        if self._first_frame_index >= 0:
            return True
        try:
            frame = self._require_streams().get_rgb()
        except CameraNotReadyError:
            return False

        self._first_frame_index = frame.index
        self._first_frame_step = -1 if step is None else step
        self._last_frame_index = frame.index
        self._latest_frame = frame
        self._camera_preview.show(frame.image, self._camera_preview_lines(frame))
        self._check_frame(frame, verifier, "first")
        verifier.check(
            "rgb_first_frame_received",
            frame.index > 0,
            frame.index,
            ">0",
            "RGB WebSocket 已收到首帧",
        )
        verifier.observe(
            "camera_stream_started",
            f"camera_head RGB 已连接：shape={frame.image.shape}, frame={frame.index}",
        )
        return True

    def _camera_preview_lines(self, frame: CameraFrame) -> list[str]:
        return [
            f"{self.agent_name} / {self.camera_config.entity_name}",
            f"RGB frame: {frame.index}",
        ]

    def _activate_camera_viewport(self, verifier) -> None:
        actor_prefix = f"{self.agent_name}_"
        body_candidates = []
        for body_name in self.model.get_body_names():
            if "camera" not in body_name.casefold():
                continue
            if body_name.startswith(actor_prefix):
                body_candidates.append(body_name[len(actor_prefix) :])
            body_candidates.append(body_name)

        candidates = list(dict.fromkeys([self.camera_config.entity_name, *body_candidates]))
        errors: list[str] = []
        for entity_name in candidates:
            try:
                self.make_camera_viewport_active(self.agent_name, entity_name)
            except RuntimeError as exc:
                errors.append(f"{entity_name}: {exc}")
                continue
            verifier.check(
                "studio_camera_entity_resolved",
                True,
                entity_name,
                "valid Camera Component entity",
                "Studio 相机实体可激活",
            )
            verifier.observe(
                "camera_viewport_active",
                f"Studio 视口已切换到 {self.agent_name}/{entity_name}",
            )
            return

        verifier.check(
            "studio_camera_entity_resolved",
            False,
            errors,
            "one valid Camera Component entity",
            "未能激活固定 G1 的 Studio 相机实体",
        )

    def _require_streams(self) -> G1CameraStreams:
        if self._camera_streams is None:
            raise RuntimeError("RGB camera streams have not been started")
        return self._camera_streams
