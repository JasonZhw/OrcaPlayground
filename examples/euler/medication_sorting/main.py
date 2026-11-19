from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

from pathlib import Path
from typing import Any


def _bootstrap_path():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "examples" / "euler" / "medication_sorting").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None


_repo_root = _bootstrap_path()

CAMERA_PORTS = (("head_cam", 7070), ("camera_left", 7074), ("camera_right", 7072))

MAX_IMAGE_SIDE = 512
JPEG_QUALITY = 75


def prepare_image(path: str) -> tuple[str, str] | None:
    """把图片缩到 <=512px 并压成 JPEG，返回 (base64, media_type)。

    实测：640x480 PNG/高质量 JPEG 单张约 40-90 KB，缩到 512px + q75 后约 15-25 KB，
    三张图请求体小 3 倍以上，延迟下降明显。
    """
    fp = Path(path)
    if not fp.exists():
        return None
    try:
        from io import BytesIO

        from PIL import Image

        im = Image.open(fp).convert("RGB")
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    except Exception:
        return None


# --------------------------------------------------------------------------
# 后端
# --------------------------------------------------------------------------


class AnthropicBackend:
    """Anthropic Messages API（原生多图）。也兼容 DeepSeek 的 /anthropic 端点。"""

    name = "anthropic"

    def __init__(self, *, api_key=None, model=None, base_url=None, timeout=120.0,
                 max_tokens=2500, retries=2):
        self.api_key = api_key or os.environ.get("FABLE_API_KEY", "")
        self.model = model or os.environ.get("FABLE_MODEL", "claude-fable-5-2")
        self.base_url = (base_url or os.environ.get("FABLE_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        self.timeout, self.max_tokens, self.retries = timeout, max_tokens, retries
        if not self.api_key:
            raise SystemExit("缺少 FABLE_API_KEY")

    def _image_block(self, path: str) -> dict | None:
        got = prepare_image(path)
        if got is None:
            return None
        b64, media = got
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}

    def ask(self, images: list[str], prompt: str) -> str:
        """通用多图问答：返回模型原始文本（空回复自动重试）。"""
        import urllib.request

        content: list[dict[str, Any]] = []
        for p in images:
            blk = self._image_block(p)
            if blk:
                content.append(blk)
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}]}
        headers = {"content-type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(f"{self.base_url}/v1/messages",
                                             data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                if text.strip():
                    return text
                # 兼容 thinking 模型：text 被 thinking 挤占时，从 thinking 内容里取 JSON
                think = "".join(b.get("thinking", "") for b in data.get("content", [])
                                if b.get("type") in ("thinking", "redacted_thinking"))
                if think.strip():
                    return think
            except Exception:
                pass
            time.sleep(0.5 * (attempt + 1))
        return ""


class OpenAICompatBackend:
    """任意 OpenAI 兼容端点（DeepSeek / Qwen-VL / GPT-4o 等）。"""

    name = "openai"

    def __init__(self, *, base_url=None, api_key=None, model=None, timeout=90.0):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise SystemExit("缺少 OPENAI_BASE_URL / OPENAI_MODEL")

    def ask(self, images: list[str], prompt: str) -> str:
        import urllib.request

        content: list[dict[str, Any]] = []
        for p in images:
            fp = Path(p)
            if fp.exists():
                b64 = base64.b64encode(fp.read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "max_tokens": 2500,
                "messages": [{"role": "user", "content": content}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# 相机取帧
# --------------------------------------------------------------------------


class FrameGrabber:
    """读三路相机的最新一帧。

    本地 orca_gym 需要先通过 bridge 打开相机流（否则 WebSocket 端口不监听）：
        set_streaming_enabled(False) -> set_camera_properties(color_port=...) -> set_streaming_enabled(True)
    之后用 VideoRecorder 订阅并 decode。
    """

    def __init__(self, bridge=None, loop=None, ports=CAMERA_PORTS):
        from orca_gym.recorder.video_recorder import VideoRecorder

        if bridge is not None and loop is not None:
            self.enable_streams(bridge, loop, ports)

        self.receivers = {}
        for name, port in ports:
            try:
                r = VideoRecorder(name, port, 64)
                r.start()
                self.receivers[name] = r
            except Exception as exc:
                print(f"[frames] {name}:{port} 启动失败 {type(exc).__name__}: {exc}", flush=True)

    @staticmethod
    def enable_streams(bridge, loop, ports):
        """开启三路相机流。"""
        try:
            names = loop.run_until_complete(bridge.get_camera_names())
        except Exception as exc:
            print(f"[frames] get_camera_names 失败: {exc}", flush=True)
            return
        for name, port in ports:
            if name not in list(names):
                print(f"[frames] 场景里没有相机 {name}（可用: {list(names)}）", flush=True)
                continue
            try:
                loop.run_until_complete(bridge.set_streaming_enabled(name, False))
                loop.run_until_complete(
                    bridge.set_camera_properties(
                        name, capture_rgb=True, capture_depth=False,
                        capture_normal=False, capture_object_color=False, color_port=port,
                    )
                )
                loop.run_until_complete(bridge.set_streaming_enabled(name, True))
                print(f"[frames] {name} 流已开启 (port {port})", flush=True)
            except Exception as exc:
                print(f"[frames] {name} 开流失败: {type(exc).__name__}: {exc}", flush=True)

    def grab(self, max_wait=6.0, *, required_simulate_index=None) -> dict[str, Any]:
        """读取最新帧；指定 required_simulate_index 时只接受该索引的帧。

        超时还没出的相机会被跳过（调用方自行判断缺帧）。
        """
        out: dict[str, Any] = {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait and len(out) < len(self.receivers):
            for name, r in self.receivers.items():
                if name in out:
                    continue
                try:
                    idx = r.get_latest_simulate_index()
                    if idx is None:
                        continue
                    if required_simulate_index is not None and idx != required_simulate_index:
                        continue
                    img = r.decode_frame_at(idx)
                    if img is not None:
                        out[name] = img
                except Exception:
                    continue
            if len(out) < len(self.receivers):
                time.sleep(0.02)
        return out

    def close(self):
        for r in self.receivers.values():
            try:
                r.stop(save_pending=False)
            except Exception:
                pass


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def _load_scene(endpoint):
    import asyncio

    import grpc
    from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
    from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)   # 必须先设置，否则 grpc aio 的 Future 会挂到别的 loop
    try:
        channel = grpc.aio.insecure_channel(endpoint)
        bridge = OrcaStudioBridge(GrpcServiceStub(channel))
        xml = loop.run_until_complete(asyncio.wait_for(bridge.load_model_xml(), 30))
        loop.run_until_complete(channel.close())
        return xml
    finally:
        loop.close()


def _short_path(path: str) -> str:
    """把 OrcaLab 的临时路径缩成便于阅读的形式。"""
    return f"{Path(path).name}  —  MuJoCo 场景描述文件"


def _probe_model():
    """确认模型后端可连通：只发一句话，不涉及仿真。"""
    backend = _make_backend("anthropic")
    reply = backend.ask([], "只回复两个字：可用")
    return f"{backend.model} -> {reply.strip()[:40] or '（空回复）'}"


def selfcheck(endpoint="127.0.0.1:50051") -> int:
    """逐项检查运行环境，任何一项失败都能立刻定位。"""
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            print(f"  ✅ {name}: {fn()}")
        except Exception as exc:
            ok = False
            print(f"  ❌ {name}: {type(exc).__name__}: {exc}")

    print("== 1. Python 依赖 ==")
    check("mujoco", lambda: __import__("mujoco").__version__)
    check("grpc", lambda: __import__("grpc").__version__)
    check("PIL", lambda: __import__("PIL").__version__)
    check("orca_gym", lambda: str(Path(__import__("orca_gym").__file__).parent))
    check("mink", lambda: __import__("mink").__version__ if hasattr(__import__("mink"), "__version__") else "ok")

    print("== 2. 仓库模块 ==")

    def imp():
        from examples.euler.medication_sorting import execution, geometry, scene, tools  # noqa: F401

        return "scene / geometry / tools / execution 可导入"

    check("examples.euler.medication_sorting", imp)

    print("== 3. 仿真端口 ==")

    def ports():
        import socket

        res = []
        for name, port in [("gRPC", 50051), *CAMERA_PORTS]:
            s = socket.socket()
            s.settimeout(2)
            try:
                s.connect(("127.0.0.1", port))
                res.append(f"{name}:{port}✓")
            except Exception:
                raise RuntimeError(f"{name}:{port} 未监听（OrcaLab 运行时没起？）") from None
            finally:
                s.close()
        return " ".join(res)

    check("OrcaLab 与相机", ports)

    print("== 4. 场景加载 ==")
    check("场景模型（MJCF）", lambda: _short_path(_load_scene(endpoint)))

    print("== 5. 相机取帧 ==")

    def grab():
        import asyncio

        import grpc
        from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
        from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

        from examples.euler.medication_sorting.execution import SortingEnv

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        channel = grpc.aio.insecure_channel(endpoint)
        bridge = OrcaStudioBridge(GrpcServiceStub(channel))
        env = None
        try:
            xml_path = loop.run_until_complete(asyncio.wait_for(bridge.load_model_xml(), 30))
            # 必须 step 一次触发渲染，相机才会出帧
            env = SortingEnv(xml_path, endpoint=endpoint)
            env.step(env.command)
            loop.run_until_complete(bridge.pause_simulation())
            g = FrameGrabber(bridge=bridge, loop=loop)
            try:
                for i in range(4):
                    loop.run_until_complete(
                        bridge.render(env.data.qpos.ravel(), float(env.data.time),
                                      simulate_index=i, request_idr=True)
                    )
                    time.sleep(0.15)
                frames = g.grab(max_wait=20.0)
                if not frames:
                    raise RuntimeError("没抓到任何帧（开流后仍在等帧）")
                return ", ".join(f"{k}{v.shape}" for k, v in frames.items())
            finally:
                g.close()
        finally:
            if env is not None:
                env.close()
            loop.run_until_complete(channel.close())
            loop.close()

    check("三路相机", grab)

    print("== 6. 模型调用 ==")
    check("视觉大模型", _probe_model)

    print()
    print("全部通过 ✅ 可以跑：python main.py" if ok else "有项目失败 ❌ 按上面提示修复")
    return 0 if ok else 1


def _load_dotenv():
    """自动加载仓库根的 .env（FABLE_API_KEY 等），已存在的环境变量不覆盖。"""
    p = Path(__file__).resolve()
    for _ in range(6):
        env_file = p.parent / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return True
        p = p.parent
    return False


class DeepSeekBackend(AnthropicBackend):
    """DeepSeek 的 Anthropic 兼容端点（视觉模型用 deepseek-flash）。

    环境变量：DEEPSEEK_API_KEY / DEEPSEEK_MODEL（默认 deepseek-flash）/
              DEEPSEEK_BASE_URL（默认 https://api.deepseek.com/anthropic）。
    """

    name = "deepseek"

    def __init__(self, **kw):
        kw.setdefault("api_key", os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("FABLE_API_KEY", ""))
        kw.setdefault("model", os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
        kw.setdefault("base_url", os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"))
        super().__init__(**kw)


def _make_backend(name: str):
    if name == "anthropic":
        return AnthropicBackend()
    if name == "deepseek":
        return DeepSeekBackend()
    if name == "openai":
        return OpenAICompatBackend()
    raise SystemExit(f"未知后端：{name}")


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Agent-as-Policy（单机版）")
    ap.add_argument("--backend", choices=["anthropic", "deepseek", "openai"], default="anthropic",
                    help="模型后端（默认 anthropic）")
    ap.add_argument("--homography", type=Path, default=Path("layouts/homography.json"),
                    help="单应矩阵文件（每个场景一份；新场景先用 --calibrate 生成）")
    ap.add_argument("--task", type=Path, default=Path("tasks/medication_sorting.yaml"),
                    help="任务 YAML（默认 tasks/medication_sorting.yaml；换任务改一行配置即可）")
    ap.add_argument("--list-bodies", action="store_true",
                    help="列出场景里可移动的 body（名字/位置/尺寸），用于写 YAML 与标定")
    ap.add_argument("--calibrate", action="store_true",
                    help="为当前场景标定单应矩阵，写到 --homography")
    ap.add_argument("--ref-body", default=None,
                    help="标定参考物体的 body 名（配合 --calibrate）")
    ap.add_argument("--ref-desc", default=None,
                    help="标定参考物体的视觉描述，模型看图用（如「黄色收纳筐」）")
    ap.add_argument("--ref-arm", default="left", choices=["left", "right"],
                    help="标定时用哪只手（默认 left）")
    ap.add_argument("--selfcheck", action="store_true", help="环境自检")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--output", type=Path, default=Path("./agent_out"))
    ap.add_argument("--capture-images", action="store_true", default=True,
                    help="开相机（agentic 视觉必须）")
    ap.add_argument("--no-capture", dest="capture_images", action="store_false",
                    help="关闭相机（调试用，视觉工具会全部失效）")
    ap.add_argument("--arm-only", action="store_true", default=True,
                    help="把模型裁成只剩双臂（默认开启）")
    ap.add_argument("--full-body", dest="arm_only", action="store_false",
                    help="使用完整机器人模型（调试用）")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck(args.endpoint)

    from examples.euler.medication_sorting.agent import (
        print_movable_bodies, run_agentic_episode, run_calibration,
    )

    backend = _make_backend(args.backend)

    if args.list_bodies:
        return print_movable_bodies(args)
    if args.calibrate:
        return run_calibration(args, backend)
    return 0 if run_agentic_episode(args, backend) else 1


if __name__ == "__main__":
    sys.exit(main())
