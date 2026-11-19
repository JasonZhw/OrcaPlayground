#!/usr/bin/env python3
"""汇总 agent_out/ 下的运行结果，输出评测表。

用法：
    python tools/eval_runs.py                 # 汇总全部运行
    python tools/eval_runs.py --since 20261119  # 只看指定日期之后的
    python tools/eval_runs.py --csv out.csv     # 额外导出 CSV

每行一次运行：后端 / 模型 / 状态 / 完成件数 / 耗时 / 首个失败码。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def _duration(events) -> str:
    """从事件时间戳算本轮耗时（秒）。"""
    times = []
    for e in events:
        t = e.get("time")
        if not t:
            continue
        try:
            times.append(datetime.fromisoformat(t))
        except ValueError:
            continue
    if len(times) < 2:
        return "-"
    return f"{(max(times) - min(times)).total_seconds():.0f}s"


def _first_failure(events) -> str:
    for e in events:
        code = e.get("code")
        if code and code not in ("ok",):
            return f"{e.get('stage', '?')}:{code}"
    return "-"


def collect(root: Path, since: str | None):
    rows = []
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not run.name.startswith("2"):
            continue
        if since and not run.name.startswith(since):
            continue
        result = run / "episode_result.json"
        if not result.is_file():
            continue
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        events = data.get("events", [])
        placed = data.get("placed", {}) or {}
        done = sum(1 for v in placed.values() if v)
        codes = Counter(
            e.get("code") for e in events
            if e.get("code") and e.get("code") not in ("ok",)
        )
        rows.append({
            "run": run.name,
            "backend": data.get("backend", "?"),
            "model": data.get("model", "")[:22],
            "status": data.get("status", "?"),
            "done": f"{done}/{len(placed)}",
            "time": _duration(events),
            "first_fail": _first_failure(events),
            "codes": ", ".join(f"{c}×{n}" for c, n in codes.most_common(3)) or "-",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("agent_out"))
    ap.add_argument("--since", default=None, help="按目录名前缀过滤，如 20261119")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = collect(args.root, args.since)
    if not rows:
        print(f"{args.root} 下没有可汇总的运行（目录名需以 2 开头，且含 episode_result.json）")
        return 1

    head = ("运行", "后端", "模型", "状态", "完成", "耗时", "首个失败", "失败码")
    widths = [len(h) for h in head]
    body = [[r["run"], r["backend"], r["model"], r["status"], r["done"],
             r["time"], r["first_fail"], r["codes"]] for r in rows]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(head))
    print("  ".join("-" * w for w in widths))
    for row in body:
        print(fmt(row))

    # 按后端汇总成功率
    print()
    by_backend: dict[str, list] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)
    for backend, items in sorted(by_backend.items()):
        ok = sum(1 for r in items if r["status"] == "complete")
        print(f"{backend:10s} 成功 {ok}/{len(items)}")

    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(head)
            writer.writerows(body)
        print(f"\n已导出 {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
