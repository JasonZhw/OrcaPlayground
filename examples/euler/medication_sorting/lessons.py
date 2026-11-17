"""经验库：记录「失败码 → 成功恢复动作」，供后续决策时提示大模型。

设计很轻：一个 JSON 文件累加统计，不训练模型、不改权重。
  失败发生 → 记 pending
  下一个动作成功 → 记 (失败码, 该动作) worked+1
  下一个动作也失败 → 记 (失败码, 该动作) failed+1
下次遇到同一失败码时，把「历史上哪些动作成功过」塞进 prompt。
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path("agent_out/lessons.json")


def _load() -> dict:
    if not _PATH.exists():
        return {}
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def record(failure_code: str, action: str, *, worked: bool) -> None:
    """记录一次「遇到 failure_code 后采取 action」的结果。"""
    if not failure_code or not action:
        return
    data = _load()
    entry = data.setdefault(str(failure_code), {})
    stat = entry.setdefault(str(action), {"worked": 0, "failed": 0})
    stat["worked" if worked else "failed"] = int(stat.get("worked" if worked else "failed", 0)) + 1
    _save(data)


def suggest(failure_code: str, limit: int = 3) -> str:
    """返回该失败码下历史成功恢复动作的一句话提示；没有经验返回空串。"""
    if not failure_code:
        return ""
    entry = _load().get(str(failure_code), {})
    if not isinstance(entry, dict) or not entry:
        return ""
    ranked = sorted(
        ((action, s) for action, s in entry.items() if isinstance(s, dict) and s.get("worked", 0) > 0),
        key=lambda kv: -int(kv[1].get("worked", 0)),
    )[:limit]
    if not ranked:
        return ""
    parts = [f"改做「{action}」成功过 {s.get('worked', 0)} 次" for action, s in ranked]
    return "；".join(parts)
