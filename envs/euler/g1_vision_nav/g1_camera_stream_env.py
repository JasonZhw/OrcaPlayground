"""Receive the head RGB stream enabled by the user in OrcaLab Studio."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from envs.euler.g1_vision_nav.camera_stream import CameraFrame, CameraNotReadyError, G1CameraStreams
from envs.euler.g1_vision_nav.config import CameraConfig
from envs.euler.g1_vision_nav.g1_pick_locomotion_env import G1PickLocomotionEnv


class G1CameraStreamEnv(G1PickLocomotionEnv):
    """Use UI camera settings; own the receiver and any capture started here."""

    # 只降低状态报告频率；接收与过期判断仍逐个控制周期执行。
    FRAME_CHECK_INTERVAL = 250
    CAMERA_WAIT_LOG_INTERVAL = 250

    def __init__(
        self,
        *args,
        camera_config: CameraConfig,
        sample_path: Path,
        **kwargs,
    ) -> None:
        self.camera_config = camera_config
        self.sample_path = sample_path
        self._camera_streams: G1CameraStreams | None = None
        self._first_frame_index = -1
        self._latest_frame: CameraFrame | None = None
        self._camera_wait_started = 0.0
        self._last_camera_wait_log_step: int | None = None
        self._owns_capture = False
        self._capture_dir: str | None = None
        super().__init__(*args, **kwargs)

    @property
    def camera_motion_ready(self) -> bool:
        """Never navigate from a placeholder or an old decoded image."""
        return self._latest_frame is not None and self._latest_frame.is_fresh(self.camera_config.frame_timeout_s)

    def before_loop(self, verifier) -> None:
        # 必须先推进仿真/render，再异步等首帧；这里阻塞等图可能导致永远不出帧。
        # UI 负责相机配置；全局采集是另一个开关，不能随 AddActor 一起删除。
        self.locomotion.set_commands(stand=0, lin_vel=(0.0, 0.0), ang_vel=0.0)
        self._first_frame_index = -1
        self._latest_frame = None
        self._camera_wait_started = time.monotonic()
        self._last_camera_wait_log_step = None
        self._camera_streams = G1CameraStreams(self.camera_config)
        self._camera_streams.start()
        self._start_capture(verifier)
        verifier.observe(
            "camera_stream_connecting",
            f"[相机] 连接端口 {self.camera_config.rgb_port}，等待图像；暂时保持站立。",
        )

    def _start_capture(self, verifier) -> None:
        """启动全局采集，但不注册 Actor、不覆盖 UI 参数、不抢占已有录制。"""
        frame = self.get_current_frame()
        if frame >= 0:
            verifier.observe("camera_capture_reused", "[相机] 复用已有采集，退出时保持开启。")
            return
        # 独立目录避免覆盖旧录像。先记录所有权，使 RPC 超时后的退出也会尝试清理。
        self._capture_dir = tempfile.mkdtemp(prefix="g1-green-table-video-")
        self._owns_capture = True
        self.begin_save_video(self._capture_dir, capture_mode=0)
        frame = self.get_current_frame()
        if frame < 0:
            raise RuntimeError("全局相机采集未启动（frame=-1）；请检查 OrcaLab 采集日志")
        verifier.observe(
            "camera_capture_started",
            f"[相机] 采集已启动；录像目录：{self._capture_dir}",
        )

    def verify_step(self, step: int, verifier) -> None:
        super().verify_step(step, verifier)
        had_frame = self._latest_frame is not None
        try:
            frame = self._require_streams().get_rgb()
        except CameraNotReadyError as exc:
            self._latest_frame = None
            if had_frame:
                self._camera_wait_started = time.monotonic()
            last_log = self._last_camera_wait_log_step
            if had_frame or last_log is None or step < last_log or step - last_log >= self.CAMERA_WAIT_LOG_INTERVAL:
                self._last_camera_wait_log_step = step
                elapsed = time.monotonic() - self._camera_wait_started
                hint = ""
                if elapsed >= self.camera_config.first_frame_timeout_s:
                    hint = f" 请检查采集、相机开关和端口。详情：{exc}"
                status = "图像中断" if had_frame else "等待图像"
                verifier.observe("camera_waiting", f"[相机] {status}，暂停行走。{hint}", step=step)
            return

        self._latest_frame = frame
        self._last_camera_wait_log_step = None
        if self._first_frame_index < 0:
            self._first_frame_index = frame.index
            self._check_frame(frame, verifier, "first")
            verifier.observe(
                "camera_stream_started",
                f"[相机] 已收到画面：{frame.image.shape[1]}×{frame.image.shape[0]}，可供导航使用。",
            )
        elif not had_frame:
            verifier.observe(f"camera_stream_recovered_{step}", "[相机] 图像已恢复，可供导航使用。", step=step)
        if step % self.FRAME_CHECK_INTERVAL == 0:
            # 控制循环可能比渲染快；新鲜帧允许被重复使用，不要求每步新帧。
            verifier.check(
                f"rgb_frame_fresh_{step}", self.camera_motion_ready, frame.index, "未过期图像",
                "[相机] 图像正常" if self.camera_motion_ready else "[相机] 图像过期",
            )

    def observe_step(self, step: int, verifier) -> None:
        return None

    def after_loop(self, verifier) -> None:
        try:
            frame = self._require_streams().get_rgb()
        except CameraNotReadyError as exc:
            verifier.check("rgb_sample_saved", False, str(exc), "未过期图像", "[相机] 无可用图像，未保存照片。")
            return
        self._latest_frame = frame
        self._check_frame(frame, verifier, "last")
        verifier.check(
            "rgb_stream_progressed",
            frame.index > self._first_frame_index >= 0,
            frame.index,
            f">{self._first_frame_index}",
            "[相机] 结束帧号大于首帧号",
        )
        self.sample_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame.image).save(self.sample_path)
        verifier.observe("rgb_sample_ready", f"[相机] 照片已保存：{self.sample_path}")

    def close(self) -> None:
        # 任一清理步骤失败也要继续；只停止自己启动的全局采集，不改 UI 开关。
        try:
            if self._camera_streams is not None:
                self._camera_streams.stop()
                self._camera_streams = None
        finally:
            try:
                if self._owns_capture:
                    self.stop_save_video()
                    self._owns_capture = False
            finally:
                super().close()

    def _check_frame(self, frame: CameraFrame, verifier, label: str) -> None:
        image = frame.image
        verifier.check(
            f"rgb_shape_{label}",
            image.ndim == 3 and image.shape[2] == 3 and min(image.shape[:2]) > 0,
            image.shape,
            "高×宽×3 通道",
            "图像尺寸检查",
        )
        verifier.check(f"rgb_dtype_{label}", image.dtype == np.uint8, str(image.dtype), "uint8", "RGB 数据类型")
        dynamic_range = int(image.max()) - int(image.min())
        verifier.check(
            f"rgb_nonempty_{label}",
            dynamic_range >= 10 and float(image.std()) >= 1.0,
            {"dynamic_range": dynamic_range, "std": float(image.std())},
            "色值跨度≥10，标准差≥1",
            "图像色彩变化检查",
        )

    def _require_streams(self) -> G1CameraStreams:
        if self._camera_streams is None:
            raise CameraNotReadyError("RGB camera receiver has not been started")
        return self._camera_streams
