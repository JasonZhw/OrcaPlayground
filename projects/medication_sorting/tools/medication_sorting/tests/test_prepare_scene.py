"""验证加载场景时的备份、资产检查和失败处理，不连接桌面。"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from prepare_scene import decode_result, prepare_scene  # noqa: E402
from scene_utils import analyze_layout  # noqa: E402

LAYOUT = TASK_DIR.parents[1] / "data/medication_sorting/scenes/drag_classfication.json"


class DesktopFixture:
    """官方 MCP 响应的离线替身，记录操作并写入实际备份文件。"""

    def __init__(self, *, running=False, missing_asset=False, fail_backup=False, fail_load=False, omit_robot=False):
        self.state = "Running" if running else "Stopped"
        self.calls = []
        self.missing_asset = missing_asset
        self.fail_backup = fail_backup
        self.fail_load = fail_load
        self.omit_robot = omit_robot
        self.omit_asset_map = False
        self.save_without_actors = False
        self.expected = analyze_layout(json.loads(LAYOUT.read_text()))["assets"]

    async def call_tool_mcp(self, name, arguments, **kwargs):
        self.calls.append(name)
        if name == "get_simulation_state":
            value = {"state": self.state, "running": self.state == "Running"}
        elif name == "get_asset_map":
            value = {asset["asset_path"]: {} for asset in self.expected}
            if self.omit_asset_map:
                value = {}
            if self.missing_asset:
                value.pop(self.expected[0]["asset_path"])
        elif name == "get_asset_info":
            asset_path = arguments["asset_path"]
            if self.missing_asset and asset_path == self.expected[0]["asset_path"]:
                value = {"code": 404, "message": "资产路径不存在"}
            else:
                value = {"asset_path": asset_path + ".spawnable"}
        elif name == "save_layout":
            value = {"success": not self.fail_backup}
            if not self.fail_backup:
                saved = {} if self.save_without_actors else {"actors": []}
                Path(arguments["layout_path"]).write_text(json.dumps(saved))
        elif name == "stop_simulation":
            self.state = "Stopped"
            value = {"success": True}
        elif name == "load_layout":
            value = {"success": not self.fail_load}
        elif name == "get_all_actors":
            value = {asset["path"]: {"name": asset["name"], "world_transform.position": asset["world_position"]}
                     for asset in self.expected
                     if not (self.omit_robot and asset["name"] == "unitree_g1_1")}
        elif name == "start_simulation":
            assert arguments == {"program_name": "external"}
            self.state = "Running"
            value = {"success": True}
        else:
            raise AssertionError(name)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(value))])


def test_decodes_official_text_result_and_rejects_application_failure():
    result = CallToolResult(content=[TextContent(type="text", text='{"state":"Stopped"}')])
    assert decode_result(result) == {"state": "Stopped"}
    for value in ('{"success":false,"message":"缺少资产"}', 'not json'):
        with pytest.raises(RuntimeError):
            decode_result(CallToolResult(content=[TextContent(type="text", text=value)]))
    with pytest.raises(RuntimeError):
        decode_result(CallToolResult(is_error=True, content=[TextContent(type="text", text="拒绝加载")]))


def test_missing_assets_do_not_stop_or_replace_current_scene(tmp_path):
    client = DesktopFixture(running=True, missing_asset=True)
    with pytest.raises(RuntimeError, match="资产"):
        asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert "stop_simulation" not in client.calls
    assert "load_layout" not in client.calls
    assert json.loads((tmp_path / "preparation.json").read_text())["missing_assets"]


def test_backup_failure_does_not_replace_or_stop_scene(tmp_path):
    client = DesktopFixture(running=True, fail_backup=True)
    with pytest.raises(RuntimeError):
        asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert "load_layout" not in client.calls
    assert "stop_simulation" not in client.calls


def test_asset_info_precheck_accepts_assets_not_returned_by_asset_map(tmp_path):
    client = DesktopFixture()
    client.omit_asset_map = True
    asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert "get_asset_info" in client.calls
    assert "start_simulation" in client.calls


def test_empty_v4_layout_without_actors_is_a_valid_backup(tmp_path):
    client = DesktopFixture()
    client.save_without_actors = True
    asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert "load_layout" in client.calls


def test_backup_precedes_stop_load_and_manual_simulation(tmp_path):
    client = DesktopFixture(running=True)
    asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert client.calls.index("save_layout") < client.calls.index("stop_simulation") < client.calls.index("load_layout")
    assert client.calls.index("get_all_actors") < client.calls.index("start_simulation")
    report = json.loads((tmp_path / "preparation.json").read_text())
    assert report["status"] == "scene_ready_for_inspection"
    assert report["trainable_for_pi05"] is False
    assert (tmp_path / "before_layout.json").is_file()


@pytest.mark.parametrize("failure", ["fail_load", "omit_robot"])
def test_load_failure_or_missing_robot_does_not_start_simulation(tmp_path, failure):
    client = DesktopFixture(**{failure: True})
    with pytest.raises(RuntimeError):
        asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert "start_simulation" not in client.calls


def test_wrong_actor_position_is_rejected_before_simulation(tmp_path):
    client = DesktopFixture()
    client.expected[0]["world_position"][0] += 10
    with pytest.raises(RuntimeError, match="位置"):
        asyncio.run(prepare_scene(client, LAYOUT, tmp_path, timeout=1))
    assert "start_simulation" not in client.calls


def test_local_mcp_client_ignores_invalid_proxy_without_changing_environment(monkeypatch):
    """重现用户的 socks:// 配置：真实 HTTP 客户端构造仍须成功。"""
    import httpx2
    from prepare_scene import create_orcalab_client

    keys = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
    for key in keys:
        monkeypatch.setenv(key, "socks://127.0.0.1:7897/")
    before = {key: os.environ[key] for key in keys}
    client = create_orcalab_client(12345, timeout=7)
    assert client.transport.url == "http://127.0.0.1:12345/mcp"
    http = client.transport.httpx_client_factory(
        headers={"x-test": "local"}, auth=None, follow_redirects=True,
        timeout=httpx2.Timeout(3, read=7),
    )
    try:
        assert http.trust_env is False
        assert http.headers["x-test"] == "local"
        assert http.timeout.read == 7
        assert {key: os.environ[key] for key in keys} == before
    finally:
        asyncio.run(http.aclose())
