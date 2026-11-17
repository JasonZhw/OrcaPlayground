"""Playbook：跨 episode 的持久经验（对齐 RobotUse 的 playbook）。

记录「某物体用哪个角度最好」，下一轮直接优先用，不再从 13 个角度里盲搜。
"""

from __future__ import annotations

import json
from pathlib import Path


class Playbook:
    def __init__(self, path="layouts/playbook.json"):
        self.path = Path(path)
        self.data = {"best_yaw": {}, "success": {}, "attempts": {}}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text()))
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))

    def best_yaw(self, key) -> float | None:
        v = self.data["best_yaw"].get(key)
        return float(v) if v is not None else None

    def record_success(self, key, yaw):
        self.data["best_yaw"][key] = float(yaw)
        self.data["success"][key] = self.data["success"].get(key, 0) + 1
        self.save()

    def record_attempt(self, key):
        self.data["attempts"][key] = self.data["attempts"].get(key, 0) + 1

    def hint(self) -> str:
        """给 agent 的剧本提示：哪些物体有已知的最佳角度。"""
        known = {k: v for k, v in self.data["best_yaw"].items() if self.data["success"].get(k, 0) > 0}
        if not known:
            return ""
        return "（剧本：已知最佳角度 " + ", ".join(f"{k}={v}°" for k, v in sorted(known.items())) + "）"
