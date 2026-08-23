"""Online RGB and depth-preview streams for the G1 head camera."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from orca_gym.sensor.rgbd_camera import CameraWrapper

from envs.euler.g1_vision_nav.config import CameraConfig


class CameraNotReadyError(RuntimeError):
    """Raised when a camera frame is requested before its stream is ready."""


class _RetryingCameraWrapper(CameraWrapper):
    """Reconnect when Studio recreates the camera stream after AddActor."""

    RETRY_INTERVAL_S = 0.10

    def loop(self) -> None:
        while self.running:
            try:
                super().loop()
            except Exception:
                pass
            if self.running:
                time.sleep(self.RETRY_INTERVAL_S)


@dataclass(frozen=True)
class CameraFrame:
    """Owned image snapshot and its monotonically increasing stream index."""

    image: np.ndarray
    index: int


class G1CameraStreams:
    """Lifecycle wrapper around OrcaGym's WebSocket ``CameraWrapper``.

    The depth port currently exposes a decoded three-channel preview. It must not
    be interpreted as metric depth until a calibration or public depth conversion
    contract is supplied by OrcaGym.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._rgb_camera: CameraWrapper | None = None
        self._depth_camera: CameraWrapper | None = None
        self._started = False

    def start(self) -> None:
        """Start enabled streams in background decoder threads."""
        if self._started:
            return
        self._rgb_camera = _RetryingCameraWrapper(
            f"{self.config.entity_name}_rgb",
            self.config.rgb_port,
        )
        self._rgb_camera.start()
        self._started = True
        try:
            if self.config.enable_depth_preview:
                depth_port = self.config.depth_port
                if depth_port is None:
                    raise CameraNotReadyError("no depth stream is configured")
                self._depth_camera = _RetryingCameraWrapper(
                    f"{self.config.entity_name}_depth_preview",
                    depth_port,
                )
                self._depth_camera.start()
        except Exception:
            self.stop()
            raise

    def wait_until_ready(self, timeout_s: float | None = None) -> None:
        """Wait for the first frame from every enabled stream."""
        if not self._started or self._rgb_camera is None:
            raise CameraNotReadyError("camera streams have not been started")
        timeout = timeout_s if timeout_s is not None else self.config.first_frame_timeout_s
        if timeout <= 0.0:
            raise ValueError("timeout_s must be positive")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rgb_ready = self._rgb_camera.is_first_frame_received()
            depth_ready = self._depth_camera is None or self._depth_camera.is_first_frame_received()
            if rgb_ready and depth_ready:
                return
            time.sleep(0.02)
        raise CameraNotReadyError(
            "camera stream did not deliver its first frame; activate it with "
            "env.set_camera_sensor_info(...) and check the configured ports"
        )

    def get_rgb(self, size: tuple[int, int] | None = None) -> CameraFrame:
        """Return an owned RGB uint8 snapshot; ``size`` is ``(width, height)``."""
        camera = self._require_ready(self._rgb_camera, "RGB")
        image, index = camera.get_frame(format="rgb24", size=size)
        return CameraFrame(image=np.asarray(image, dtype=np.uint8).copy(), index=index)

    def get_depth_preview(self, size: tuple[int, int] | None = None) -> CameraFrame:
        """Return the depth stream's BGR preview without claiming metric units."""
        if not self.config.enable_depth_preview:
            raise CameraNotReadyError("depth preview is disabled in CameraConfig")
        camera = self._require_ready(self._depth_camera, "depth preview")
        image, index = camera.get_frame(format="bgr24", size=size)
        return CameraFrame(image=np.asarray(image, dtype=np.uint8).copy(), index=index)

    def stop(self) -> None:
        """Stop all decoder threads. Safe to call more than once."""
        if not self._started:
            return
        if self._rgb_camera is not None:
            self._rgb_camera.stop()
        if self._depth_camera is not None:
            self._depth_camera.stop()
        self._started = False

    def __enter__(self) -> "G1CameraStreams":
        self.start()
        try:
            self.wait_until_ready()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @staticmethod
    def _require_ready(camera: CameraWrapper | None, label: str) -> CameraWrapper:
        if camera is None or not camera.is_first_frame_received():
            raise CameraNotReadyError(f"{label} stream is not ready")
        return camera
