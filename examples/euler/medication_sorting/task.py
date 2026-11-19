"""任务定义：把「哪件物体、长什么样、进哪个筐、用哪只手」从代码里抽出来。

harness（agent / tools / scene / geometry）只读这份配置，不硬编码任何物体名。
换任务 = 换一份 YAML，代码不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ObjectSpec:
    """一件物体的任务属性。"""

    key: str
    desc: str
    dest: str
    arm: str


@dataclass(frozen=True)
class TaskSpec:
    """一次任务的完整定义：每件物体的 key / 描述 / 目标筐 / 使用哪只手。"""

    name: str
    objects: dict[str, ObjectSpec] = field(default_factory=dict)
    bins: dict[str, str] = field(default_factory=dict)

    @property
    def keys(self) -> tuple[str, ...]:
        """按配置顺序返回物体 key（即抓取顺序）。"""
        return tuple(self.objects)

    def object(self, key: str) -> ObjectSpec:
        return self.objects[key]

    def validate(self) -> "TaskSpec":
        if not self.objects:
            raise ValueError("任务至少需要一件物体")
        for obj in self.objects.values():
            if obj.arm not in ("left", "right"):
                raise ValueError(f"{obj.key}: arm 必须是 left/right，得到 {obj.arm!r}")
            if obj.dest not in self.bins:
                raise ValueError(f"{obj.key}: 目标筐 {obj.dest!r} 不在 bins 里")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        """从 YAML 读取。"""
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} 必须是 YAML 对象")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        bins = {str(k): str(v) for k, v in data.get("bins", {}).items()}
        objects: dict[str, ObjectSpec] = {}
        for item in data.get("objects", []):
            if not isinstance(item, dict) or "key" not in item:
                raise ValueError("objects 每项都需要 key 字段")
            key = str(item["key"])
            objects[key] = ObjectSpec(
                key=key,
                desc=str(item.get("desc", key)),
                dest=str(item.get("dest", "")),
                arm=str(item.get("arm", "left")),
            )
        return cls(name=str(data.get("name", "task")), objects=objects, bins=bins).validate()
