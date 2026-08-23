"""Non-blocking browser preview for the G1 head camera."""

from __future__ import annotations

import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence

import numpy as np

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>G1 First Person</title>
<style>
html,body{margin:0;background:#111;color:#eee;font-family:sans-serif;height:100%}
body{display:flex;flex-direction:column;align-items:center;justify-content:center}
h2{margin:10px}.hint{margin:0 0 10px;color:#aaa}
img{width:min(96vw,960px);height:auto;border:1px solid #444;background:#000}
</style></head><body>
<h2>G1 First Person &mdash; camera_head</h2>
<div class="hint">Closing or refreshing this page does not stop navigation.</div>
<img src="/stream.mjpg" alt="Waiting for camera_head RGB...">
</body></html>"""


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
        open_browser: bool = True,
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

    def start(self) -> None:
        """Start the local HTTP viewer and open it in the default browser."""
        if not self.enabled or self._server is not None:
            return
        try:
            handler = self._build_handler()
            try:
                server = ThreadingHTTPServer((self.host, self.port), handler)
            except OSError:
                server = ThreadingHTTPServer((self.host, 0), handler)
                print(
                    f"[WARNING] Preview port {self.port} is occupied; "
                    f"using {server.server_address[1]} instead."
                )
            server.daemon_threads = True
            self._server = server
            self._server_thread = threading.Thread(
                target=server.serve_forever,
                name="g1-camera-preview-http",
                daemon=True,
            )
            self._server_thread.start()
            print(f"[INFO] G1 first-person preview: {self.url}")
            if self.open_browser and self.url is not None:
                threading.Thread(
                    target=webbrowser.open_new,
                    args=(self.url,),
                    name="g1-camera-preview-browser",
                    daemon=True,
                ).start()
        except Exception as exc:
            self._disable_with_warning(exc)

    def show(self, rgb: np.ndarray, lines: Sequence[str] = ()) -> None:
        """Publish one frame when the configured preview FPS permits it."""
        if not self.enabled:
            return
        if self._server is None:
            self.start()
        if self._server is None:
            return
        now = time.monotonic()
        if now < self._next_encode_time:
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
                "[WARNING] First-person browser preview unavailable; "
                f"navigation will continue without it: {exc}"
            )
            self._warning_printed = True
        self.close()
