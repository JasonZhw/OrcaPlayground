"""Online RGB and depth-preview streams for the G1 head camera."""

from __future__ import annotations

import asyncio
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence

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
        """Return a fresh RGB-ordered preview, not a metric depth array."""
        if not self.config.enable_depth_preview:
            raise CameraNotReadyError("depth preview is disabled in CameraConfig")
        camera = self._require_ready(self._depth_camera, "depth preview")
        frame = camera.snapshot()
        if not frame.is_fresh(self.config.frame_timeout_s):
            raise CameraNotReadyError("depth preview frame is stale")
        if size is not None:
            frame = replace(frame, image=cv2.resize(frame.image, size))
        return frame

    def stop(self) -> None:
        """Stop all decoder threads. Safe to call more than once."""
        if not self._started:
            return
        try:
            if self._rgb_camera is not None:
                self._rgb_camera.stop()
        finally:
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


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>G1 相机预览</title>
<style>
html,body{margin:0;background:#111;color:#eee;font-family:sans-serif;height:100%}
body{display:flex;flex-direction:column;align-items:center;justify-content:center}
h2{margin:10px}.hint{margin:0 0 10px;color:#aaa}
img{width:min(96vw,960px);height:auto;border:1px solid #444;background:#000}
</style></head><body>
<h2>G1 相机预览 &mdash; camera_head</h2>
<img src="/stream.mjpg" alt="等待相机画面…">
</body></html>""".encode("utf-8")


class CameraPreviewWindow:
    """Serve the latest RGB frame to a browser without blocking control."""

    def __init__(
        self,
        *,
        enabled: bool,
        host: str = "127.0.0.1",
        port: int = 8765,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        open_browser: bool = False,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("preview port must be in [0, 65535]")
        if width <= 0 or height <= 0 or fps <= 0.0:
            raise ValueError("preview dimensions and fps must be positive")
        self.enabled = enabled
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.open_browser = open_browser
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._frame_serial = 0
        self._next_encode_time = 0.0
        self._warning_printed = False

    @property
    def url(self) -> str | None:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def has_frame(self) -> bool:
        with self._condition:
            return self._latest_jpeg is not None

    def start(self) -> None:
        """Start HTTP preview; opening the browser requires explicit opt-in."""
        if not self.enabled or self._server is not None:
            return
        try:
            handler = self._build_handler()
            try:
                server = ThreadingHTTPServer((self.host, self.port), handler)
            except OSError:
                server = ThreadingHTTPServer((self.host, 0), handler)
                print(
                    f"[网页] 端口 {self.port} 无法使用，改用 {server.server_address[1]}。"
                )
            server.daemon_threads = True
            self._server = server
            self._server_thread = threading.Thread(
                target=server.serve_forever,
                name="g1-camera-preview-http",
                daemon=True,
            )
            self._server_thread.start()
            print(f"[网页] 手动打开 {self.url} 查看第一视角与导航信息（不自动弹窗）。")
            if self.open_browser and self.url is not None:
                threading.Thread(
                    target=webbrowser.open_new,
                    args=(self.url,),
                    name="g1-camera-preview-browser",
                    daemon=True,
                ).start()
        except Exception as exc:
            self._disable_with_warning(exc)

    @property
    def is_due(self) -> bool:
        """Let the control-thread producer skip preparing frames between updates."""
        return self.enabled and time.monotonic() >= self._next_encode_time

    def show(self, rgb: np.ndarray, lines: Sequence[str] = (), *, force: bool = False) -> None:
        """Publish one frame when the configured preview FPS permits it."""
        if not self.enabled:
            return
        if self._server is None:
            self.start()
        if self._server is None:
            return
        now = time.monotonic()
        if not force and now < self._next_encode_time:
            return
        self._next_encode_time = now + 1.0 / self.fps

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("preview frame must be an HxWx3 uint8 RGB array")
        try:
            jpeg = self._encode_jpeg(image, lines)
        except Exception as exc:
            self._disable_with_warning(exc)
            return
        with self._condition:
            self._latest_jpeg = jpeg
            self._frame_serial += 1
            self._condition.notify_all()

    def close(self) -> None:
        """Stop the local preview server without affecting OrcaLab."""
        self.enabled = False
        with self._condition:
            self._condition.notify_all()
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

    def _encode_jpeg(self, rgb: np.ndarray, lines: Sequence[str]) -> bytes:
        import cv2

        canvas = cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        self._draw_hud(cv2, canvas, lines)
        success, encoded = cv2.imencode(
            ".jpg",
            canvas,
            (cv2.IMWRITE_JPEG_QUALITY, 88),
        )
        if not success:
            raise RuntimeError("OpenCV failed to encode the preview frame")
        return encoded.tobytes()

    @staticmethod
    def _draw_hud(cv2, canvas: np.ndarray, lines: Sequence[str]) -> None:
        line_height = 22
        panel_height = min(canvas.shape[0], 14 + line_height * len(lines))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0, canvas)
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                str(line),
                (10, 22 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 245, 255),
                1,
                cv2.LINE_AA,
            )

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                    return
                if self.path == "/stream.mjpg":
                    self._stream_frames()
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def _stream_frames(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                serial = -1
                try:
                    while owner.enabled:
                        with owner._condition:
                            owner._condition.wait_for(
                                lambda: owner._frame_serial != serial or not owner.enabled,
                                timeout=1.0,
                            )
                            if not owner.enabled:
                                return
                            jpeg = owner._latest_jpeg
                            serial = owner._frame_serial
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args) -> None:
                return None

        return PreviewHandler

    def _disable_with_warning(self, exc: Exception) -> None:
        if not self._warning_printed:
            print(
                f"[网页] 预览不可用，导航仍继续：{exc}"
            )
            self._warning_printed = True
        self.close()
