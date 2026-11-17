#!/usr/bin/env python3
"""Agent-as-Policy：视觉大模型看图做高层决策，驱动已有运动原语完成药品分拣。

单机运行（推荐）
----------------
模型、仿真、执行全在同一台机器上：

    export FABLE_API_KEY=sk-ant-...
    python agent_policy.py --selfcheck                       # 环境自检
    python agent_policy.py --backend anthropic --capture-images

设计原则
--------
1. 观测只有图像：三路相机 + 极简文本，不含任何物体坐标。
2. 动作空间离散，不含 arm（机械臂由「物体 → 目标筐」决定）：
       {"action":"pick","object":"calcium","yaw":45}
       {"action":"skip","object":"herbbox"}
       {"action":"finish"}
3. 失败重试由模型主导：执行器回报失败原因，模型决定换角度 / 换物体 / 跳过。
4. 成功判定用真值校验，模型判断只用于决定下一步。

依赖
----
- PYTHONPATH 含仓库根目录（能 import examples.euler.medication_sorting.*）
- OrcaLab 运行时在线（50051 + 相机 7070/7072/7074）
- orca_gym / mujoco / PIL / grpc
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

# 自动定位仓库根目录并加入 sys.path，这样直接 `python examples/.../agent_policy.py` 也能跑
def _bootstrap_path():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "examples" / "euler" / "medication_sorting").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None


_REPO_ROOT = _bootstrap_path()

# 右臂先抓最靠近的盒子（herbbox 的 X 更小、离右臂基座更近），再抓远的
OBJECT_KEYS = ("calcium", "melatonin", "herbbox", "pharyngitis")
# 固定抓取顺序（阶段1）：先左臂两瓶（近的钙片先，挡住褪黑素），再右臂两盒
FIXED_ORDER = ("calcium", "melatonin", "pharyngitis", "herbbox")
ARM_FOR = {"calcium": "left", "melatonin": "left", "pharyngitis": "right", "herbbox": "right"}
BIN_FOR = {"calcium": "left_bin", "melatonin": "left_bin", "pharyngitis": "right_bin", "herbbox": "right_bin"}
KIND_FOR = {"calcium": "bottle", "melatonin": "bottle", "pharyngitis": "box", "herbbox": "box"}
YAW_CANDIDATES = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)
CAMERA_PORTS = (("head_cam", 7070), ("camera_left", 7074), ("camera_right", 7072))

TASK_TEXT = (
    "任务：桌面有 4 件药品——两只白色药瓶（钙片、褪黑素）和两个蓝色药盒（咽炎片、她秘方）。"
    "需要把药瓶放进左侧紫色筐，把药盒放进右侧蓝色筐。四件全部入筐才算完成。"
)


# --------------------------------------------------------------------------
# 观测 / 决策
# --------------------------------------------------------------------------


@dataclass
class Observation:
    step: int = 0
    remaining: list[str] = field(default_factory=list)
    placed: dict[str, bool] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    last_error: str | None = None
    attempt_counts: dict[str, int] = field(default_factory=dict)
    tried_yaws: dict[str, list[float]] = field(default_factory=dict)

    def to_prompt(self) -> str:
        done = [f"{k}({KIND_FOR[k]})" for k, v in self.placed.items() if v]
        todo = [f"{k}({KIND_FOR[k]})" for k in self.remaining]
        lines = [
            TASK_TEXT,
            "",
            f"[第 {self.step} 步]",
            f"已入筐：{done if done else '无'}",
            f"待处理：{todo if todo else '无'}",
        ]
        lines.append(
            "提示：同一只手上的物体可能互相遮挡——先抓离机械臂近的，"
            "把挡路的拿走后，后面的往往才抓得到。抓不到不代表它永远抓不到。"
        )
        tried = ", ".join(f"{k} 试了 {n} 次" for k, n in self.attempt_counts.items() if n)
        if tried:
            lines.append(f"尝试次数：{tried}")
        if self.tried_yaws:
            hist = ", ".join(f"{k}: {v}" for k, v in self.tried_yaws.items() if v)
            if hist:
                lines.append(f"已试过的角度：{hist}  ← 不要重复，优先试还没试过的角度")
            exhausted = [k for k, v in self.tried_yaws.items() if len(v) >= len(YAW_CANDIDATES)]
            if exhausted:
                lines.append(f"⚠ 已试尽全部角度仍失败：{exhausted}  ← 对它们直接输出 skip")
        if self.last_error:
            lines.append(f"上一次尝试失败原因：{self.last_error}")
        if self.images:
            names = ["头部相机（俯视全场景）", "左腕相机", "右腕相机"]
            lines.append("")
            lines.append(f"共 {len(self.images)} 张图像：")
            for i in range(len(self.images)):
                lines.append(f"  [图{i + 1}] {names[i] if i < len(names) else '相机'}")
        lines += [
            "",
            "你能做的动作（只输出一个 JSON，不要解释）：",
            '  {"action":"pick","object":"<物体名>","yaw":<角度>}   ← 抓取并放入对应筐',
            '  {"action":"skip","object":"<物体名>"}                ← 这件做不到，跳过',
            '  {"action":"finish"}                                  ← 四件都完成了',
            "",
            "角度候选（度，绕竖直轴）：0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90",
            "同一物体用某角度失败后，换一个明显不同的角度再试。",
            "同一物体连续失败 3 次以上，考虑 skip 它先做其他物体。",
            "只输出 JSON。",
        ]
        return "\n".join(lines)


@dataclass
class Decision:
    action: str
    object: str | None = None
    yaw: float | None = None
    reason: str | None = None

    @classmethod
    def parse(cls, raw: str) -> "Decision":
        text = raw.strip()
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.lower().startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    text = block
                    break
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"输出里没有 JSON: {raw[:200]}")
        data = json.loads(text[start : end + 1])
        obj = data.get("object")
        if isinstance(obj, str):
            obj = obj.split("(")[0].split("（")[0].strip().lower()
            if obj not in OBJECT_KEYS:
                for k in OBJECT_KEYS:
                    if k in obj:
                        obj = k
                        break
        return cls(
            action=str(data.get("action", "")).lower(),
            object=obj,
            yaw=float(data["yaw"]) if data.get("yaw") is not None else None,
            reason=data.get("reason"),
        )


# --------------------------------------------------------------------------
# 后端
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# 图像预处理：缩小尺寸 + 压缩，显著降低模型延迟
# --------------------------------------------------------------------------

MAX_IMAGE_SIDE = 512
JPEG_QUALITY = 75


def prepare_image(path: str) -> tuple[str, str] | None:
    """把图片缩到 <=512px 并压成 JPEG，返回 (base64, media_type)。

    实测：640x480 PNG/高质量 JPEG 单张约 40-90 KB，
    缩到 512px + q75 后约 15-25 KB，三张图请求体小 3 倍以上，延迟下降明显。
    """
    fp = Path(path)
    if not fp.exists():
        return None
    try:
        from io import BytesIO

        from PIL import Image

        im = Image.open(fp).convert("RGB")
        w, h = im.size
        scale = min(1.0, MAX_IMAGE_SIDE / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    except Exception:
        raw = base64.b64encode(fp.read_bytes()).decode()
        return raw, "image/jpeg"


class Backend(Protocol):
    name: str

    def decide(self, obs: Observation) -> Decision: ...


class DryRunBackend:
    """离线状态机：0° 必失败，用于验证决策闭环。"""

    name = "dryrun"

    def __init__(self):
        self.tried: dict[tuple[str, float], int] = {}

    def decide(self, obs: Observation) -> Decision:
        if not obs.remaining:
            return Decision(action="finish")
        key = obs.remaining[0]
        for yaw in YAW_CANDIDATES:
            if not self.tried.get((key, float(yaw))):
                self.tried[(key, float(yaw))] = 1
                return Decision(action="pick", object=key, yaw=float(yaw))
        return Decision(action="skip", object=key)


class HeuristicBackend:
    name = "heuristic"

    def __init__(self):
        self.count: dict[str, int] = {}

    def decide(self, obs: Observation) -> Decision:
        if not obs.remaining:
            return Decision(action="finish")
        key = obs.remaining[0]
        n = self.count.get(key, 0)
        self.count[key] = n + 1
        if n >= len(YAW_CANDIDATES):
            return Decision(action="skip", object=key)
        return Decision(action="pick", object=key, yaw=float(YAW_CANDIDATES[n]))


class AnthropicBackend:
    """Anthropic Messages API（Claude Fable 5.2），原生多图。

    注意：该模型不接受 temperature 参数（传了会 400）。
    """

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

    def decide(self, obs: Observation) -> Decision:
        import urllib.request

        content: list[dict[str, Any]] = []
        for p in obs.images:
            blk = self._image_block(p)
            if blk:
                content.append(blk)
        content.append({"type": "text", "text": obs.to_prompt()})

        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}]}
        headers = {"content-type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        last_err: Exception | None = None
        for _ in range(self.retries):
            req = urllib.request.Request(f"{self.base_url}/v1/messages",
                                         data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            try:
                return Decision.parse(text)
            except Exception as exc:
                last_err = exc
                body = {**body, "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": text[:200]},
                    {"role": "user", "content": "上一轮没有输出可解析的 JSON。只输出一个 JSON 对象，不要任何其他文字。"},
                ]}
        raise RuntimeError(f"模型输出无法解析：{last_err}")


def _verify_prompt(obj: str, bin_name: str) -> str:
    return (
        f"看图回答：{obj}（{KIND_FOR.get(obj, '物体')}）现在是否已经放进"
        f"{'左侧紫色筐' if 'left' in bin_name else '右侧蓝色筐'}里？\n"
        "只看图像判断，不要猜测。如果物体还在桌面上或还在机械手上，就算没有放进。\n"
        '只输出 JSON：{"in_bin": true} 或 {"in_bin": false}'
    )


class Verifier:
    """让模型看图判断「是否已入筐」。

    同时也把真值记下来，用于事后评估模型的判断准确率。
    """

    def __init__(self, backend):
        self.backend = backend

    def check(self, images: list[str], obj: str, bin_name: str) -> tuple[bool | None, str]:
        if not images:
            return None, "无图像"
        try:
            raw = self.backend.ask(images, _verify_prompt(obj, bin_name))
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1:
                return None, raw[:80]
            data = json.loads(text[start : end + 1])
            return bool(data.get("in_bin")), raw[:80]
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"


class OpenAICompatBackend:
    """备用：任意 OpenAI 兼容端点。"""

    name = "openai"

    def __init__(self, *, base_url=None, api_key=None, model=None, timeout=90.0):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise SystemExit("缺少 OPENAI_BASE_URL / OPENAI_MODEL")

    def decide(self, obs: Observation) -> Decision:
        import urllib.request

        content: list[dict[str, Any]] = []
        for p in obs.images:
            fp = Path(p)
            if fp.exists():
                b64 = base64.b64encode(fp.read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": obs.to_prompt()})
        body = {"model": self.model, "max_tokens": 300,
                "messages": [{"role": "user", "content": content}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return Decision.parse(data["choices"][0]["message"]["content"])


# --------------------------------------------------------------------------
# 相机取帧（自包含，不依赖仓库补丁）
# --------------------------------------------------------------------------


class FrameGrabber:
    """读三路相机的最新一帧。

    本地 orca_gym 需要先通过 bridge 打开相机流（否则 WebSocket 端口不监听）：
        set_streaming_enabled(False) -> set_camera_properties(color_port=...) -> set_streaming_enabled(True)
    之后用 VideoRecorder 订阅并 decode。
    """

    def __init__(self, bridge=None, loop=None, ports=CAMERA_PORTS, warmup=5.0):
        from orca_gym.recorder.video_recorder import VideoRecorder

        if bridge is not None and loop is not None:
            self._enable_streams(bridge, loop, ports)

        self.receivers = {}
        for name, port in ports:
            try:
                r = VideoRecorder(name, port, 64)
                r.start()
                self.receivers[name] = r
            except Exception as exc:
                print(f"[frames] {name}:{port} 启动失败 {type(exc).__name__}: {exc}", flush=True)

    @staticmethod
    def _enable_streams(bridge, loop, ports):
        """按仓库 recording.py 的方式开启三路相机流。"""
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
        """Read frames; supervised actions require the exact requested render index.

        A camera that has not produced that frame is omitted on timeout. Legacy
        callers without an index retain their existing latest-frame behavior.
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
# 主策略
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# 关节锁：只允许 7×2 臂关节 + 夹爪运动，其余全部锁死
# --------------------------------------------------------------------------


class JointLock:
    """把除「14 个臂关节 + 16 个夹爪连杆」之外的所有执行器锁死。

    - 指令层：每次 step 前，把锁定执行器的目标值强制写回初始值
    - 物理层：SortingEnv.fix_base() 已用等式约束焊死骨盆，这里再记录漂移用于验证
    """

    def __init__(self, scene, env):
        import numpy as np

        self.env = env
        self.np = np
        controllable = set()
        for arm in ("left", "right"):
            for n in scene.arms[arm].joint_names:
                if n in env.joint_names:
                    controllable.add(env.joint_names.index(n))
        grippers = [i for i, n in enumerate(env.joint_names) if "_gripper_" in n]
        controllable.update(grippers)

        self.arm_idx = sorted(i for i in controllable if i not in grippers)
        self.grip_idx = grippers
        self.locked_idx = [i for i in range(len(env.joint_names)) if i not in controllable]
        self.locked_names = [env.joint_names[i] for i in self.locked_idx]

        self.reference = np.asarray(env.command, dtype=float).copy()
        self.initial_qpos = env.data.qpos.ravel()[env.qids][self.locked_idx].copy()
        self.max_drift = 0.0

        original_step = env.step

        def step_with_lock(command):
            cmd = np.asarray(command, dtype=float).copy()
            cmd[self.locked_idx] = self.reference[self.locked_idx]
            out = original_step(cmd)
            cur = env.data.qpos.ravel()[env.qids][self.locked_idx]
            drift = float(np.max(np.abs(cur - self.initial_qpos))) if len(self.locked_idx) else 0.0
            self.max_drift = max(self.max_drift, drift)
            return out

        env.step = step_with_lock

    def summary(self) -> str:
        return (f"可动 {len(self.arm_idx) + len(self.grip_idx)} 个执行器"
                f"（臂 {len(self.arm_idx)} + 夹爪 {len(self.grip_idx)}），"
                f"锁定 {len(self.locked_idx)} 个：{', '.join(n.replace('unitree_g1_1_', '') for n in self.locked_names)}")


def _reset_to_demo_default(scene, env, settle=40):
    """把两只手臂复位到示范默认姿态（关节全 0 = 双臂水平前伸）。

    为什么需要：OrcaLab 没有复位接口，场景状态会跨运行保留——
    上一轮结束时手臂停在哪里，这一轮就从哪里开始，导致每轮结果不一致。
    显式复位后，每次运行都从同一个初始状态出发。
    """
    import numpy as np

    # 平滑复位：从当前指令线性插值到 0，避免瞬时跳变造成"突然抖动"
    ids_all = {}
    start_cmd = {}
    for arm in ("left", "right"):
        ids = [env.joint_names.index(n) for n in scene.arms[arm].joint_names]
        ids_all[arm] = ids
        start_cmd[arm] = env.command[ids].copy()

    ramp = max(settle, 40)
    for i in range(1, ramp + 1):
        a = i / ramp
        a = a * a * (3 - 2 * a)          # 缓入缓出
        for arm in ("left", "right"):
            env.command[ids_all[arm]] = (1 - a) * start_cmd[arm]
        env.step(env.command)
    for _ in range(settle):
        env.step(env.command)
    scene.set_qpos(env.data.qpos.ravel().copy())
    max_jump = max(float(np.abs(start_cmd[a]).max()) for a in start_cmd)
    print(f"[start] 已平滑复位到示范默认姿态（{ramp} 步过渡，最大初始偏移 {max_jump:.2f} rad）", flush=True)


def _prepare_ready_pose(scene, env, loop, bridge, height=5.25, ramp_steps=60):
    """启动姿态处理：默认什么都不做。

    场景默认的手臂姿态（关节全 0 = 双臂水平前伸）就是示范数据的起始姿态，
    从它出发已经能抓 3/4 的物体，所以不需要额外的抬臂动作。
    只有在该姿态与物体/筐碰撞时，才平滑抬到预备姿态作为安全兜底。
    """
    import numpy as np

    from examples.euler.medication_sorting.planning import IKError, solve_ik

    scene.set_qpos(env.data.qpos.ravel().copy())
    blocked = [a for a in ("left", "right") if not scene.collision_free(scene.d.qpos.copy(), a)]
    if not blocked:
        print("[start] 保持示范默认姿态（双臂水平），不做额外动作", flush=True)
        return False

    print(f"[start] 默认姿态碰撞（{blocked}），平滑抬到预备姿态 z≈{height}", flush=True)
    targets = {}
    for arm in ("left", "right"):
        p, r = scene.end_pose(arm)
        try:
            sol = solve_ik(scene, arm, np.array([p[0], p[1], height]), r,
                           scene.d.qpos.copy(), attempts=30, random_seed=0)
            targets[arm] = sol[scene.arms[arm].qpos_ids]
        except IKError as exc:
            print(f"[start] {arm} 预备姿态求解失败: {str(exc)[-50:]}", flush=True)
    if not targets:
        return False

    ids = {arm: [env.joint_names.index(n) for n in scene.arms[arm].joint_names] for arm in targets}
    start_cmd = {arm: env.command[ids[arm]].copy() for arm in targets}
    for i in range(1, ramp_steps + 1):
        # smoothstep 缓入缓出，比线性更柔和
        a = i / ramp_steps
        a = a * a * (3 - 2 * a)
        for arm, goal in targets.items():
            env.command[ids[arm]] = (1 - a) * start_cmd[arm] + a * goal
        env.step(env.command)
    print(f"[start] 已平滑抬到预备姿态（{ramp_steps} 步，≈{ramp_steps * 0.02:.1f}s）", flush=True)
    return True


class Reporter:
    """把「现在正在做什么」清楚打到终端：阶段 + 时间戳 + 已用时长。"""

    def __init__(self):
        self.t0 = time.monotonic()

    def _stamp(self):
        el = time.monotonic() - self.t0
        return f"{time.strftime('%H:%M:%S')} +{int(el // 60):02d}:{int(el % 60):02d}"

    def stage(self, msg: str):
        print(f"[{self._stamp()}] ▸ {msg}", flush=True)

    def done(self, msg: str):
        print(f"[{self._stamp()}] ✅ {msg}", flush=True)

    def fail(self, msg: str):
        print(f"[{self._stamp()}] ✗ {msg}", flush=True)

    def banner(self, msg: str):
        line = "─" * 62
        print(f"\n{line}\n{msg}\n{line}", flush=True)


@dataclass
class Metrics:
    """运行指标：调用次数、延迟、失败、兜底次数。"""
    decisions: int = 0
    verifications: int = 0
    llm_seconds: float = 0.0
    llm_failures: int = 0
    fallbacks: int = 0
    fast_paths: int = 0
    skipped_verifications: int = 0

    def summary(self) -> str:
        n = self.decisions + self.verifications
        avg = self.llm_seconds / n if n else 0.0
        return (f"LLM 调用 {n} 次（决策 {self.decisions} / 校验 {self.verifications}），"
                f"累计 {self.llm_seconds:.1f}s，平均 {avg:.2f}s；"
                f"失败 {self.llm_failures}，兜底 {self.fallbacks}，"
                f"快路径 {self.fast_paths}，跳过校验 {self.skipped_verifications}")


class AgentPolicy:
    def __init__(self, backend: Backend, *, log_path: Path | None = None,
                 frames_dir: Path | None = None):
        self.rep = Reporter()
        self.metrics = Metrics()
        self.backend = backend
        self.log_path = log_path
        self.frames_dir = frames_dir
        self.records: list[dict[str, Any]] = []

    def run_dry(self, steps: int = 20) -> bool:
        pending = list(OBJECT_KEYS)
        deferred: list[str] = []          # 暂缓的物体（等挡路的被移走后可重试）
        deferred_rounds = 0
        placed = {k: False for k in OBJECT_KEYS}
        counts = {k: 0 for k in OBJECT_KEYS}
        last_error = None
        for step in range(steps):
            obs = Observation(step=step, remaining=list(pending), placed=dict(placed),
                              last_error=last_error, attempt_counts=dict(counts))
            d = self.backend.decide(obs)
            if d.action == "finish":
                self._record(obs, d, last_error, 0.0)
                break
            if d.action == "skip" and d.object in pending:
                pending.remove(d.object)
                self._record(obs, d, last_error, 0.0)
                continue
            if d.action == "pick" and d.object in pending:
                counts[d.object] += 1
                if float(d.yaw or 0) == 0.0:
                    last_error = f"{d.object}@0deg: 起点碰撞"
                else:
                    placed[d.object] = True
                    pending.remove(d.object)
                    last_error = None
            self._record(obs, d, last_error, 0.0)
        ok = all(placed.values())
        print(f"[dry-run] {'成功' if ok else '失败'}  已放入={[k for k, v in placed.items() if v]}")
        return ok

    def run_sim_episode(self, args) -> bool:
        import asyncio

        import grpc
        import numpy as np

        from examples.euler.medication_sorting.execution import SortingEnv
        from examples.euler.medication_sorting.skills import CODE_HINT, classify, get_skill
        from examples.euler.medication_sorting.workflow import (
            DESTINATIONS, Trial, placement_satisfied, prepare_model, Scene,
        )
        from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
        from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        channel = grpc.aio.insecure_channel(
            args.endpoint, options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)]
        )
        bridge = OrcaStudioBridge(GrpcServiceStub(channel))
        source = loop.run_until_complete(asyncio.wait_for(bridge.load_model_xml(), 30))

        pad = None
        if getattr(args, "arm_only", True):
            # 裁成「只剩两条手臂」：腿/腰/头/浮动底座从模型里物理删除
            from examples.euler.medication_sorting import arm_only_model

            # 先应用 condim=6 / impratio=2 等物理参数，再裁剪
            prepared = prepare_model(source, out / "prepared.xml")
            ao = arm_only_model.build(prepared)
            scene_xml = str(ao.out_path)
            env = arm_only_model.make_arm_only_env(ao, args.endpoint)
            pad = ao.pad

            # OrcaStudio 需要完整维度 qpos，统一在 render 出口补齐
            _render = bridge.render

            def render_padded(qpos, *a, **kw):
                return _render(pad(qpos), *a, **kw)

            bridge.render = render_padded
        else:
            scene_xml = prepare_model(source, out / "scene.xml")
            env = SortingEnv(scene_xml, endpoint=args.endpoint)

        scene = Scene(scene_xml)
        loop.run_until_complete(bridge.pause_simulation())

        # 直接用 OrcaLab 当前打开的布局，不再套 --reference-layout 覆盖物体位置
        env.set_joint_qvel(np.zeros(env.model.nv))
        env.mj_forward()
        env.step(env.command)
        scene.set_qpos(env.data.qpos.ravel().copy())
        print("[layout] 使用 OrcaLab 当前布局（不覆盖物体位置）:", flush=True)
        for k in OBJECT_KEYS:
            p, _ = scene.body_pose(k)
            print(f"    {k:<12} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})", flush=True)

        if pad is None:
            lock = JointLock(scene, env)
            print(f"[lock] {lock.summary()}", flush=True)
        else:
            lock = None
            print("[lock] 关节已在模型层删除（只剩双臂 + 夹爪），无需指令层锁定", flush=True)

        _reset_to_demo_default(scene, env)

        # 不做启动抬臂：prepare_pick 里每条手臂自带的 +8cm clearance 已经
        # 既能解除初始接触、又能提供好的 IK 种子。预处理把手臂抬到 z≈5.25
        # 反而会让部分物体超出可达范围（实测可抓数从 3/4 掉到 1/4）。
        _ = _prepare_ready_pose  # 保留函数作为排障工具，但不在主流程调用

        trial = Trial(env, scene, bridge, loop, out, args.record, args.task_index, "all")
        trial.hold(75, "initial_settle")
        trial.align_initial_command()
        trial.home = env.command.copy()
        trial.record_start()

        # 记录「抬臂后」的安全指令，每次抓取失败后复位，避免手臂越爬越高
        home_command = env.command.copy()

        def restore_home():
            # 平滑复位：smoothstep 过渡手臂关节到 home，避免瞬时跳变
            ramp = 40
            start_cmd = env.command.copy()
            arm_ids = []
            for a in ("left", "right"):
                arm_ids += [env.joint_names.index(n) for n in scene.arms[a].joint_names]
            for i in range(1, ramp + 1):
                t = i / ramp
                t = t * t * (3 - 2 * t)          # 缓入缓出
                env.command[arm_ids] = (1 - t) * start_cmd[arm_ids] + t * home_command[arm_ids]
                env.step(env.command)
            env.command[:] = home_command        # 夹爪等其余部分直接到位（手臂已在 home）
            trial.hold(30, "home_reset")

        grabber = FrameGrabber(bridge=bridge, loop=loop) if args.capture_images else None
        verifier = Verifier(self.backend) if hasattr(self.backend, "ask") else None

        # 视觉接地（L1）：加载单应矩阵，存在则启用「模型报像素 → 粗世界坐标」
        grounder = None
        if getattr(args, "ground", True) and Path("layouts/homography.json").exists():
            from examples.euler.medication_sorting.visual_grounding import load_homography

            try:
                grounder = load_homography("layouts/homography.json")
                print("[ground] 已加载 homography.json，启用视觉接地（L1）", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ground] 加载失败，退回真值（L0）：{exc}", flush=True)

        # Playbook：跨 episode 的持久经验（物体→最佳抓取角度）
        from examples.euler.medication_sorting.playbook import Playbook

        playbook = Playbook("layouts/playbook.json")
        if playbook.hint():
            print(f"[playbook] {playbook.hint()}", flush=True)

        def ground_object(images, key):
            """让模型报物体像素 → 单应矩阵 → 粗世界坐标。失败返回 None（退回真值）。"""
            if grounder is None or not images or not hasattr(self.backend, "ask"):
                return None
            import re

            from examples.euler.medication_sorting.visual_grounding import pixel_to_world

            prompt = (f"这是桌面俯视图（640x480）。找出目标物体（{key}）的中心。"
                      "只输出一行 JSON：{\"u\":整数,\"v\":整数}")
            try:
                raw = self.backend.ask(images, prompt)
            except Exception:  # noqa: BLE001
                return None
            u = re.search(r'"u"\s*:\s*(\d+)', raw)
            v = re.search(r'"v"\s*:\s*(\d+)', raw)
            if not u or not v:
                return None
            x, y = pixel_to_world(grounder, int(u.group(1)), int(v.group(1)))
            return (float(x), float(y))

        pending = list(OBJECT_KEYS)
        deferred: list[str] = []          # 暂缓的物体（等挡路的被移走后可重试）
        deferred_rounds = 0
        placed = {k: False for k in OBJECT_KEYS}
        counts = {k: 0 for k in OBJECT_KEYS}
        tried: dict[str, set] = {k: set() for k in OBJECT_KEYS}
        last_error: str | None = None
        ok = False
        verify_stats = {"agree": 0, "disagree": 0, "unknown": 0}

        def capture(tag: str):
            if not grabber:
                return []
            for i in range(3):
                try:
                    loop.run_until_complete(
                        bridge.render(env.data.qpos.ravel(), float(env.data.time),
                                      simulate_index=900000 + len(self.records) * 10 + i,
                                      request_idr=True)
                    )
                except Exception:
                    break
                time.sleep(0.08)
            frames = grabber.grab(max_wait=8.0)
            paths = []
            if self.frames_dir and frames:
                d = Path(self.frames_dir) / tag
                d.mkdir(parents=True, exist_ok=True)
                from PIL import Image as PILImage

                for name, arr in frames.items():
                    pth = d / f"{name}.jpg"
                    PILImage.fromarray(arr).save(pth, "JPEG", quality=90)
                    paths.append(str(pth))
            return paths

        try:
            for step in range(args.max_steps):
                total_n = len(OBJECT_KEYS)
                done_n = sum(1 for v in placed.values() if v)
                self.rep.banner(f"第 {step + 1} 步 · 进度 {done_n}/{total_n} 已入筐"
                                f" {[k for k, v in placed.items() if v]}")
                self.rep.stage(f"① 观测：拍照（待处理 {len(pending)} 件：{pending}）")
                images = capture(f"step{step:02d}_before")
                self.rep.stage(f"   看到 {len(images)} 张图（head / left_wrist / right_wrist）")
                if not images and step == 0:
                    print("[warn] 没抓到相机帧，模型只能靠文字决策", flush=True)

                obs = Observation(step=step, remaining=list(pending), placed=dict(placed),
                                  images=images, last_error=last_error,
                                  attempt_counts=dict(counts),
                                  tried_yaws={k: sorted(v) for k, v in tried.items() if v})
                self.rep.stage("② 决策：分析图像，选择动作…")
                t0 = time.monotonic()
                if getattr(args, "fixed_order", False):
                    # 固定顺序：不调模型，取序列里下一个「待处理且角度未试尽」的物体
                    def _exhausted(k):
                        return len(tried.get(k, ())) >= len(YAW_CANDIDATES)
                    nxt = next((k for k in FIXED_ORDER if k in pending and not _exhausted(k)), None)
                    decision = Decision(action="pick", object=nxt, yaw=0.0,
                                        reason="fixed_order") if nxt else Decision(action="finish")
                    llm_s = 0.0
                    self.rep.stage(f"   决策 → 调用技能 pick({nxt})" if nxt else "   决策 → finish()")
                else:
                    decision = self._decide(obs)
                    llm_s = time.monotonic() - t0
                    self.rep.stage(f"   决策 → {decision.action}({decision.object or ''})"
                                   f"{' yaw=' + str(decision.yaw) if decision.yaw is not None else ''}"
                                   f"（{llm_s:.1f}s）")

                if decision.action == "finish":
                    self._record(obs, decision, last_error, llm_s)
                    break
                if decision.action == "skip":
                    key = decision.object
                    if key in pending:
                        pending.remove(key)
                        if not placed.get(key):
                            # 暂缓而非永久放弃：挡路的物体被移走后可能就能抓了
                            if key not in deferred:
                                deferred.append(key)
                            print(f"[defer] {key} 暂缓，等其它物体移走后重试", flush=True)
                    self._record(obs, decision, last_error, llm_s)
                    if not pending and deferred and deferred_rounds < 1:
                        deferred_rounds += 1
                        pending = list(deferred)
                        deferred = []
                        for k in pending:
                            tried[k] = set()      # 角度历史清空，允许重新尝试
                        print(f"[retry] 第一轮结束，重试暂缓的物体：{pending}", flush=True)
                    continue
                if decision.action not in ("pick", "skip", "finish"):
                    last_error = f"非法动作：{asdict(decision)}，只能 pick/skip/finish"
                    self._record(obs, decision, last_error, llm_s)
                    self._invalid_actions = getattr(self, "_invalid_actions", 0) + 1
                    if self._invalid_actions >= 3:
                        print("[warn] 模型连续输出非法动作，自动结束", flush=True)
                        break
                    continue
                if decision.action == "pick" and (decision.object is None or decision.object not in pending):
                    # 模型没给出有效物体：退回启发式，抓第一个待处理
                    key_fallback = pending[0] if pending else None
                    print(f"[warn] 模型给的物体无效（{decision.object}），改抓 {key_fallback}", flush=True)
                    decision = Decision(action="pick", object=key_fallback,
                                        yaw=decision.yaw, reason="invalid_object_fallback")
                    if key_fallback is None:
                        break

                key = decision.object
                arm = ARM_FOR[key]
                hint = float(decision.yaw if decision.yaw is not None else 0.0)
                # 剧本优先：有已知最佳角度时，用它作为首选提示（仍允许执行层兜底搜索）
                if decision.yaw is None and playbook.best_yaw(key) is not None:
                    hint = playbook.best_yaw(key)
                    self.rep.stage(f"   剧本提示 {key} 用 {hint}°")
                counts[key] += 1

                # 视觉接地：模型报像素 → 粗世界坐标（L1）；失败退回真值（L0）
                coarse_xy = ground_object(images, key)

                # 技能库执行：grasp 内部做角度 + 局部搜索，返回结构化结果
                self.rep.stage(f"③ 技能：grasp({key})  ← {ARM_FOR[key]}臂 → {DESTINATIONS[key]}")
                ctx = {"trial": trial, "scene": scene, "restore_home": restore_home}
                result = get_skill("grasp").run(
                    ctx, key=key, arm=arm, yaw=hint,
                    yaw_candidates=YAW_CANDIDATES, tried=tried[key],
                    coarse_xy=coarse_xy, search_radius=0.08,
                )
                used_yaw = result.detail.get("used_yaw")
                exec_err = None if result.ok else f"[{result.code}] {result.message}"
                exec_s = time.monotonic() - t0 - llm_s
                if used_yaw is not None:
                    self.rep.stage(f"   抓取完成，实际角度 {used_yaw}°（执行 {exec_s:.0f}s）")
                else:
                    self.rep.fail(f"   抓取失败 [{result.code}]（执行 {exec_s:.0f}s）")

                # ---- RobotUse：place ≠ release，释放前确认 ----
                # grasp 只把物体降到筐内并保留夹持；释放前先让模型看一眼。
                pending_release = result.detail.get("pending_release")
                if result.ok and pending_release:
                    self.rep.stage(f"④ 校验：释放前视觉确认「{key} 可以松手了吗」")
                    pre_images = capture(f"step{step:02d}_prerelease")
                    ready, raw_ready = self._verify_ready(pre_images, key, DESTINATIONS[key])
                    if ready is None:
                        self.rep.stage("   模型无法判断，退回几何规则")
                    elif ready:
                        self.rep.stage("   模型判断：可以松手")
                    else:
                        self.rep.fail("   模型判断：尚未到位（仍释放，记录分歧）")
                    # 记录释放前判断（用于评测模型可靠性）
                    self._release_checks = getattr(self, "_release_checks", [])
                    self._release_checks.append({"object": key, "ready": ready, "raw": raw_ready})
                    try:
                        trial.release_and_return(*pending_release)
                        release_err = None
                    except Exception as exc:  # noqa: BLE001
                        release_err = f"[{classify(exc)}] {type(exc).__name__}: {exc}"
                        if exec_err is None:
                            exec_err = release_err
                    if release_err:
                        self.rep.fail(f"   释放失败：{release_err[:60]}")
                    else:
                        self.rep.done(f"   已释放 → {key} 落入筐内")

                # ---- 成功判定：模型看图 + 真值对照 ----
                truth = bool(placement_satisfied(scene, key, DESTINATIONS[key]))
                after_images = capture(f"step{step:02d}_after")
                # 只要有图就校验：抓取过程可能中途抛异常但物体已落筐
                if after_images:
                    model_verdict, raw = self._verify(verifier, after_images, key, DESTINATIONS[key])
                else:
                    model_verdict, raw = None, "无图像"
                    self.metrics.skipped_verifications += 1
                if model_verdict is None:
                    verify_stats["unknown"] += 1
                elif model_verdict == truth:
                    verify_stats["agree"] += 1
                else:
                    verify_stats["disagree"] += 1
                if model_verdict is None:
                    self.rep.stage(f"⑤ 校验跳过，用真值：{'入筐' if truth else '未入筐'}")
                elif model_verdict == truth:
                    self.rep.stage(f"⑤ 模型判断={model_verdict}，与真值一致")
                else:
                    self.rep.fail(f"⑤ 模型判断={model_verdict}，真值={truth} ⚠ 不一致")

                # 流程用模型判断（贴近真机），但真值永远记录
                if model_verdict is True or (model_verdict is None and truth):
                    placed[key] = True
                    if key in pending:
                        pending.remove(key)
                    last_error = None
                    if used_yaw is not None:
                        playbook.record_success(key, used_yaw)  # 持久化最佳角度
                    self.rep.done(f"{key} 已入筐（累计 {sum(1 for v in placed.values() if v)}/{total_n}）")
                    if not pending and deferred and deferred_rounds < 1:
                        deferred_rounds += 1
                        pending = list(deferred)
                        deferred = []
                        for k in pending:
                            tried[k] = set()
                        print(f"[retry] 第一轮结束，重试暂缓的物体：{pending}", flush=True)
                else:
                    last_error = exec_err or f"{key} 未入筐（真值={truth}）"
                    # 附上结构化失败码的中文提示，供模型下一轮选择恢复动作
                    if not result.ok and result.code in CODE_HINT:
                        last_error += f"（{CODE_HINT[result.code]}）"
                    self.rep.fail(f"{key} 未入筐 → {str(last_error)[:80]}")
                self._record(obs, decision,
                             {"error": last_error, "used_yaw": used_yaw,
                              "code": getattr(result, "code", None),
                              "model_verdict": model_verdict, "truth": truth}, llm_s)
        finally:
            ok = all(placed.values())
            total = verify_stats["agree"] + verify_stats["disagree"]
            acc = (verify_stats["agree"] / total * 100) if total else float("nan")
            print(f"\n结果：{'成功' if ok else '失败'}  已入筐={[k for k, v in placed.items() if v]}")
            print(f"视觉判定与真值一致率：{acc:.0f}%  (一致 {verify_stats['agree']} / 不一致 "
                  f"{verify_stats['disagree']} / 无法判断 {verify_stats['unknown']})")
            print(f"运行指标：{self.metrics.summary()}")
            if lock is not None:
                print(f"关节锁：锁定 {len(lock.locked_idx)} 个执行器，"
                      f"全程最大漂移 {lock.max_drift * 1000:.3f} mm")
            else:
                print("关节锁：非手臂关节已从模型中删除，漂移恒为 0")
            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.log_path).write_text(json.dumps(
                    {"records": self.records, "verify_stats": verify_stats, "success": ok,
                 "metrics": asdict(self.metrics)},
                    ensure_ascii=False, indent=2))
            if grabber:
                grabber.close()
            env.close()
            loop.run_until_complete(channel.close())
            loop.close()
        return ok


    def _decide(self, obs: Observation) -> Decision:
        """调模型决策；连续失败时退回启发式，保证循环不会卡死。"""
        t0 = time.monotonic()
        for attempt in range(2):
            try:
                d = self.backend.decide(obs)
                self.metrics.decisions += 1
                self.metrics.llm_seconds += time.monotonic() - t0
                return d
            except Exception as exc:  # noqa: BLE001
                self.metrics.llm_failures += 1
                print(f"[warn] 模型决策失败({attempt + 1}/2): {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.8)
        # 兜底：抓第一个待处理物体，让执行层去搜角度
        self.metrics.fallbacks += 1
        key = obs.remaining[0] if obs.remaining else None
        print(f"[warn] 退回启发式决策：pick {key}", flush=True)
        return Decision(action="pick", object=key, yaw=0.0, reason="heuristic_fallback")

    def _verify(self, verifier, images, key, bin_name):
        """视觉校验；失败不抛异常，返回 None 让调用方走真值兜底。"""
        if verifier is None or not images:
            self.metrics.skipped_verifications += 1
            return None, "跳过"
        t0 = time.monotonic()
        try:
            verdict, raw = verifier.check(images, key, bin_name)
            self.metrics.verifications += 1
            self.metrics.llm_seconds += time.monotonic() - t0
            return verdict, raw
        except Exception as exc:  # noqa: BLE001
            self.metrics.llm_failures += 1
            return None, f"{type(exc).__name__}: {exc}"

    def _verify_ready(self, images, key, bin_name):
        """释放前视觉确认（RobotUse：成功≠夹住了，释放前必须看一眼）。

        返回 (ready: bool|None, raw)。None 表示无法判断，调用方退回几何规则。
        """
        if not images or not hasattr(self.backend, "ask"):
            return None, "无图像或后端不支持"
        prompt = (f"看图回答：目标物体（{key}）现在是否已经降到"
                  f"{'左侧紫色筐' if 'left' in bin_name else '右侧蓝色筐'}内、"
                  "可以被松手放下了？只看图像判断，不要猜测。"
                  '只输出 JSON：{"ready": true} 或 {"ready": false}')
        t0 = time.monotonic()
        try:
            raw = self.backend.ask(images, prompt)
            import re
            m = re.search(r'"ready"\s*:\s*(true|false)', raw, re.IGNORECASE)
            self.metrics.verifications += 1
            self.metrics.llm_seconds += time.monotonic() - t0
            if not m:
                return None, raw[:60]
            return m.group(1).lower() == "true", raw[:60]
        except Exception as exc:  # noqa: BLE001
            self.metrics.llm_failures += 1
            return None, f"{type(exc).__name__}: {exc}"

    def _record(self, obs, decision, error, llm_s):
        self.records.append({
            "step": obs.step, "remaining": obs.remaining, "decision": asdict(decision),
            "error": error, "llm_s": round(llm_s, 2), "images": len(obs.images),
        })
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.log_path).write_text(json.dumps(self.records, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------
# 自检
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
    name = Path(path).name
    return f"{name}  (OrcaLab 生成的世界模型，含机器人/桌子/筐/物体)"


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

    print("== 2. 仓库模块 ==")

    def imp():
        from examples.euler.medication_sorting import workflow, planning, execution  # noqa: F401
        return "workflow / planning / execution 可导入"

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
                raise RuntimeError(f"{name}:{port} 未监听（OrcaLab 运行时没起？）")
            finally:
                s.close()
        return " ".join(res)

    check("OrcaLab 与相机", ports)

    print("== 4. 场景加载 ==")
    check("OrcaLab 世界模型(XML)", lambda: _short_path(_load_scene(endpoint)))

    print("== 5. 场景物体 ==")

    def layout_check():
        # 直接用 OrcaLab 当前布局，不再要求参考布局文件
        import mujoco

        p = _load_scene(endpoint)
        m = mujoco.MjModel.from_xml_path(p)
        return f"OrcaLab 场景已加载 nq={m.nq} nbody={m.nbody}（物体位置以当前场景为准，不覆盖）"

    check("当前场景四件物体", layout_check)

    print("== 6. 相机取帧 ==")

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
                # 相机按 simulate_index 打标签，必须带 index 渲染几次才会出帧
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

    print("== 7. 模型调用 ==")
    check("claude-fable-5-2", lambda: _probe_model())

    print()
    print("全部通过 ✅ 可以跑：python agent_policy.py --backend anthropic --capture-images"
          if ok else "有项目失败 ❌ 按上面提示修复")
    return 0 if ok else 1


def _probe_model():
    b = AnthropicBackend()
    obs = Observation(step=0, remaining=list(OBJECT_KEYS),
                      placed={k: False for k in OBJECT_KEYS})
    d = b.decide(obs)
    return f"{b.model} -> {d.action} {d.object} yaw={d.yaw}"


def _load_dotenv():
    """自动加载仓库根的 .env（FABLE_API_KEY 等），已存在的环境变量不覆盖。"""
    import os
    from pathlib import Path

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


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Agent-as-Policy（单机版）")
    ap.add_argument("--backend", choices=["dryrun", "heuristic", "anthropic", "openai"],
                    default="anthropic", help="默认 anthropic（真大模型）；测试可用 dryrun")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--reference-layout", type=Path, default=Path("layouts/final_layout.json"))
    ap.add_argument("--output", type=Path, default=Path("./agent_out"))
    ap.add_argument("--log", type=Path, default=Path("./agent_out/decisions.json"))
    ap.add_argument("--frames", type=Path, default=Path("./agent_out/frames"))
    ap.add_argument("--capture-images", action="store_true", default=True,
                    help="默认开相机（agentic 视觉必须）")
    ap.add_argument("--no-capture", dest="capture_images", action="store_false",
                    help="关闭相机")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--arm-only", action="store_true", default=True,
                    help="把模型裁成只剩双臂（默认开启）")
    ap.add_argument("--full-body", dest="arm_only", action="store_false",
                    help="使用完整机器人模型（调试用）")
    ap.add_argument("--no-ground", dest="ground", action="store_false", default=True,
                    help="关闭视觉接地，退回真值坐标（L0）")
    ap.add_argument("--fixed-order", action="store_true",
                    help="固定抓取顺序 calcium→melatonin→pharyngitis→herbbox，不调模型选物体")
    ap.add_argument("--agentic", action="store_true", default=True,
                    help="默认细粒度 agentic 循环")
    ap.add_argument("--legacy", dest="agentic", action="store_false",
                    help="改用旧的粗粒度 pick_place 流程")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--objects", nargs="+", choices=OBJECT_KEYS,
                    help="agentic 模式只运行指定物体，例如 --objects pharyngitis")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck(args.endpoint)

    backend: Backend = (
        AnthropicBackend() if args.backend == "anthropic"
        else OpenAICompatBackend() if args.backend == "openai"
        else HeuristicBackend() if args.backend == "heuristic"
        else DryRunBackend()
    )
    policy = AgentPolicy(backend, log_path=args.log, frames_dir=args.frames)

    if args.dry_run:
        ok = policy.run_dry()
        print("\n--- 决策日志 ---")
        for r in policy.records:
            print(f"  step {r['step']:>2}: {r['decision']}  err={r['error']}")
        return 0 if ok else 1

    if args.agentic:
        # 懒导入，避免与 agent_policy 的循环依赖
        from examples.euler.medication_sorting.agentic_loop import run_agentic_episode
        return 0 if run_agentic_episode(args, policy) else 1

    return 0 if policy.run_sim_episode(args) else 1


if __name__ == "__main__":
    sys.exit(main())
