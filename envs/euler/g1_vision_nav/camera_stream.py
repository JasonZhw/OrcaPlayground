"""Online RGB and depth-preview streams for the G1 head camera."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace

import av
import cv2
import numpy as np
import websockets

from envs.euler.g1_vision_nav.config import CameraConfig


class CameraNotReadyError(RuntimeError):
    """Raised when a camera frame is requested before its stream is ready."""


@dataclass(frozen=True)
class CameraFrame:
    """Owned image snapshot and its monotonically increasing stream index."""

    image: np.ndarray
    index: int
    received_at: float = 0.0

    def is_fresh(self, timeout_s: float) -> bool:
        """Use decode time, not read time: repeatedly reading an old frame is stale."""
        return self.index > 0 and 0.0 <= time.monotonic() - self.received_at <= timeout_s


class CameraReceiver:
    """Consume OrcaGym's 8-byte-header + H.264 WebSocket format, without RPCs.

    Socket reads time out periodically so stop() can finish even when Studio
    stays connected without sending data. Connection failures retry while the
    simulation starts. Image, frame number and arrival time are one snapshot.
    """

    def __init__(self, name: str, port: int) -> None:
        self.name = name
        self.port = port
        self.last_error = "waiting for first frame"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                raise RuntimeError(f"camera receiver {self.name} did not stop")
        with self._lock:
            self._frame = None

    def is_first_frame_received(self) -> bool:
        with self._lock:
            return self._frame is not None

    def snapshot(self) -> CameraFrame:
        with self._lock:
            if self._frame is None:
                raise CameraNotReadyError(f"{self.name}: {self.last_error}")
            return replace(self._frame, image=self._frame.image.copy())

    def _run(self) -> None:
        asyncio.run(self._receive())

    async def _receive(self) -> None:
        index = 0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{self.port}",
                    open_timeout=1.0,
                    close_timeout=0.5,
                    max_size=16 * 1024 * 1024,
                ) as websocket:
                    decoder = av.CodecContext.create("h264", "r")
                    while not self._stop.is_set():
                        try:
                            data = await asyncio.wait_for(websocket.recv(), timeout=0.25)
                        except asyncio.TimeoutError:
                            continue
                        if not isinstance(data, bytes) or len(data) <= 8:
                            raise ValueError("expected an 8-byte header followed by H.264 bytes")
                        for packet in decoder.parse(data[8:]):
                            for decoded in decoder.decode(packet):
                                index += 1
                                frame = CameraFrame(decoded.to_ndarray(format="rgb24"), index, time.monotonic())
                                with self._lock:
                                    self._frame = frame
                                self.last_error = ""
            except (OSError, TimeoutError, ValueError, websockets.WebSocketException, av.error.FFmpegError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._frame = None
                self._stop.wait(0.25)


class G1CameraStreams:
    """Lifecycle wrapper for UI-managed OrcaGym WebSocket streams.

    The depth port currently exposes a decoded three-channel preview. It must not
    be interpreted as metric depth until a calibration or public depth conversion
    contract is supplied by OrcaGym.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._rgb_camera: CameraReceiver | None = None
        self._depth_camera: CameraReceiver | None = None
        self._started = False

    def start(self) -> None:
        """Start enabled streams in background decoder threads."""
        if self._started:
            return
        self._rgb_camera = CameraReceiver(f"{self.config.entity_name}_rgb", self.config.rgb_port)
        self._rgb_camera.start()
        self._started = True
        try:
            if self.config.enable_depth_preview:
                depth_port = self.config.depth_port
                if depth_port is None:
                    raise CameraNotReadyError("no depth stream is configured")
                self._depth_camera = CameraReceiver(
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
            "camera stream did not deliver its first frame; check Studio Enable, "
            "IsRecording, ColorPort and whether simulation/rendering is advancing"
        )

    def get_rgb(self, size: tuple[int, int] | None = None) -> CameraFrame:
        """Return an owned RGB uint8 snapshot; ``size`` is ``(width, height)``."""
        camera = self._require_ready(self._rgb_camera, "RGB")
        frame = camera.snapshot()
        if not frame.is_fresh(self.config.frame_timeout_s):
            raise CameraNotReadyError("RGB frame is stale; hold position until new frames arrive")
        if size is not None:
            frame = replace(frame, image=cv2.resize(frame.image, size))
        return frame

    def get_depth_preview(self, size: tuple[int, int] | None = None) -> CameraFrame:
        """Return the depth stream's BGR preview without claiming metric units."""
        if not self.config.enable_depth_preview:
            raise CameraNotReadyError("depth preview is disabled in CameraConfig")
        camera = self._require_ready(self._depth_camera, "depth preview")
        frame = camera.snapshot()
        image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        if size is not None:
            image = cv2.resize(image, size)
        return replace(frame, image=image)

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
    def _require_ready(camera: CameraReceiver | None, label: str) -> CameraReceiver:
        if camera is None or not camera.is_first_frame_received():
            raise CameraNotReadyError(f"{label} stream is not ready")
        return camera
