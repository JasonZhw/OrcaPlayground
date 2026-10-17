#!/usr/bin/env python3
"""通过 OrcaLab 官方 MCP 加载药品场景，然后运行物理结构探针。"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from scene_utils import analyze_layout

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = PROJECT / "data/medication_sorting/scenes/drag_classfication.json"


def save_report(output, report):
    (output / "preparation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def decode_result(result):
    """官方服务返回 text 内嵌 JSON；工具错误和业务失败均须停止流程。"""
    raw = result.model_dump(by_alias=True, exclude_none=True)
    if raw.get("isError"):
        raise RuntimeError(f"OrcaLab 工具报错：{raw.get('content')}")
    texts = [block["text"] for block in raw.get("content", []) if block.get("type") == "text"]
    if len(texts) != 1:
        raise RuntimeError("OrcaLab 未返回唯一的 JSON 文本")
    try:
        payload = json.loads(texts[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("OrcaLab 返回的内容不是 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OrcaLab 返回的 JSON 不是对象")
    if payload.get("success") is False:
        raise RuntimeError(payload.get("message", "OrcaLab 操作失败"))
    return payload


async def prepare_scene(client, layout_path, output, *, timeout=120):
    """先检查资产并备份，再切换场景；返回后仍须验证物理模型。"""
    contents = layout_path.read_bytes()
    layout = json.loads(contents)
    expected = analyze_layout(layout)
    if expected["missing_task_actors"]:
        raise ValueError(f"输入 Layout 缺少任务对象：{expected['missing_task_actors']}")
    report = {
        "status": "preparing", "layout_path": str(layout_path.resolve()),
        "layout_sha256": hashlib.sha256(contents).hexdigest(), "calls": [],
        "required_packages": [reference["name"] for reference in layout.get("package_references", [])],
        "trainable_for_pi05": False, "grasp_verified": False,
    }

    async def call(name, arguments=None):
        entry = {"tool": name, "arguments": arguments or {}}
        report["calls"].append(entry)
        save_report(output, report)
        print(f"OrcaLab: {name}", flush=True)
        result = await asyncio.wait_for(
            client.call_tool_mcp(name=name, arguments=arguments or {}, timeout=timeout), timeout=timeout,
        )
        entry["result"] = decode_result(result)
        save_report(output, report)
        return entry["result"]

    async def wait_state(target):
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            state = await call("get_simulation_state")
            if state.get("state") == target:
                return
            if state.get("state") == "Failed" or asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"仿真没有进入 {target} 状态：{state}")
            await asyncio.sleep(0.5)

    try:
        state = await call("get_simulation_state")
        if state.get("state") not in ("Stopped", "Running"):
            raise RuntimeError(f"当前仿真尚未就绪：{state}")
        asset_map = await call("get_asset_map")
        available = {key.removesuffix(".spawnable") for key in asset_map}
        report["missing_assets"] = []
        for asset in expected["assets"]:
            if asset["asset_path"] in available:
                continue
            # 26.9.1 的资产表可能只返回部分已同步资产，逐项查询作为权威回退。
            info = await call("get_asset_info", {"asset_path": asset["asset_path"]})
            actual = str(info.get("asset_path", "")).removesuffix(".spawnable")
            if info.get("code") == 404 or actual != asset["asset_path"]:
                report["missing_assets"].append(asset)
        if report["missing_assets"]:
            missing = ", ".join(asset["name"] for asset in report["missing_assets"])
            packages = ", ".join(report["required_packages"])
            raise RuntimeError(f"当前账号/资产面板缺少场景所需资产：{missing}；场景引用资产包：{packages}。请订阅并同步后重试")

        # 场景替换前先保存已有编辑内容；备份失败时不停止或替换已有场景。
        backup = output.resolve() / "before_layout.json"
        await call("save_layout", {"layout_path": str(backup)})
        saved_layout = json.loads(backup.read_text()) if backup.is_file() else None
        valid_actors = (
            isinstance(saved_layout, dict)
            and ("actors" not in saved_layout or isinstance(saved_layout["actors"], list))
        )
        if not valid_actors:
            raise RuntimeError("当前场景备份未落盘或不是有效 Layout，已停止加载")
        report["backup_path"] = str(backup)
        if state["state"] == "Running":
            await call("stop_simulation")
            await wait_state("Stopped")
        await call("load_layout", {"layout_path": str(layout_path.resolve())})
        actors = await call("get_all_actors")
        report["missing_actor_paths"] = [asset["path"] for asset in expected["assets"] if asset["path"] not in actors]
        if report["missing_actor_paths"]:
            raise RuntimeError(f"Layout 加载后仍缺少对象：{report['missing_actor_paths']}")
        for asset in expected["assets"]:
            actual = np.asarray(actors[asset["path"]].get("world_transform.position"), dtype=float)
            if actual.shape != (3,) or not np.allclose(actual, asset["world_position"], atol=1e-4, rtol=0):
                raise RuntimeError(f"Layout 加载后的对象位置不匹配：{asset['path']}")
        report["loaded_actor_count"] = len(expected["assets"])
        await call("start_simulation", {"program_name": "external"})
        await wait_state("Running")
        report["status"] = "scene_ready_for_inspection"
        return report
    except BaseException as exc:
        report.update(status="preparation_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(output, report)


def create_orcalab_client(port, *, timeout):
    """本机 MCP 直接连接回环地址，不读取 Shell 中的 HTTP/SOCKS 代理。"""
    import httpx2
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    transport = StreamableHttpTransport(
        f"http://127.0.0.1:{port}/mcp",
        httpx_client_factory=partial(httpx2.AsyncClient, trust_env=False),
    )
    return Client(transport, timeout=timeout, init_timeout=10)


async def connect_and_prepare(args, output):

    config = tomllib.loads((PROJECT / "software/OrcaPlayground/.orcalab/config.toml").read_text())
    port = config.get("mcp", {}).get("port", 12345)
    url = f"http://127.0.0.1:{port}/mcp"
    print(f"连接现有 OrcaLab：{url}", flush=True)
    async with create_orcalab_client(port, timeout=args.timeout) as client:
        return await prepare_scene(client, args.layout, output, timeout=args.timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--timeout", type=float, default=120, help="单次 MCP 操作最长等待秒数")
    parser.add_argument("--record-seconds", type=float, default=3, help="只记录诊断状态，不含 RGB 或动作标签")
    args = parser.parse_args()
    if not 0 < args.timeout <= 180 or not 0 <= args.record_seconds <= 60:
        parser.error("超时范围 0–180 秒，诊断录制范围 0–60 秒")
    output = PROJECT / "data/medication_sorting/reports" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-prepare")
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = asyncio.run(connect_and_prepare(args, output))
        probe = subprocess.run([
            sys.executable, str(Path(__file__).with_name("inspect_scene.py")),
            "--layout", str(args.layout.resolve()), "--record-seconds", str(args.record_seconds),
            "--output", str(output / "live"),
        ], check=False)
        report["diagnostic_probe_exit_code"] = probe.returncode
        report["status"] = "diagnostic_probe_passed" if probe.returncode == 0 else "diagnostic_probe_failed"
        save_report(output, report)
        print("诊断报告：", output, flush=True)
        print("当前入口只加载场景和记录状态；尚未执行抓取、采集视觉示范或启动 π0.5 训练。", flush=True)
        return 0 if probe.returncode == 0 else 1
    except KeyboardInterrupt:
        print("已中断；请查看 OrcaLab 当前状态。", file=sys.stderr)
        return 130
    except Exception as exc:
        if not (output / "preparation.json").exists():
            save_report(output, {"status": "mcp_connection_failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"尚未完成加载：{exc}\n报告：{output / 'preparation.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
