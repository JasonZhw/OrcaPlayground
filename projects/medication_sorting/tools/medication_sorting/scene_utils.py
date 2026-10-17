"""药品分类场景与观测校验；离线解析不会证明物理抓取可用。"""

from itertools import combinations
import re

import numpy as np

ROBOT = "unitree_g1_1"
BOTTLES = ("calcium_gluconate_tablets_bottle_1", "melatonin_tablet_bottle_1")
BOXES = ("fu_shi_gao_herb_drink_packaging_box_1", "pharyngitis_tablets_box_1")
BLUE_BIN = "blue_plastic_crates_1"
PURPLE_BIN = "purple_plastic_crates_1"
REQUIRED_ACTORS = (ROBOT, BLUE_BIN, PURPLE_BIN, *BOTTLES, *BOXES)


def analyze_layout(layout):
    """展开层级，组合父节点变换并输出四个分类任务。四元数顺序为 wxyz。"""
    assets, paths = [], set()

    def walk(nodes, parent):
        for actor in nodes:
            path = actor["path"]
            if path in paths:
                raise ValueError(f"重复 Actor 路径：{path}")
            paths.add(path)
            transform = actor.get("transform", {})
            pos = np.asarray(transform.get("position", [0, 0, 0]), dtype=float)
            quat = np.asarray(transform.get("rotation", [1, 0, 0, 0]), dtype=float)
            scale = float(transform.get("scale", 1))
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError(f"位置无效：{path}")
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
                raise ValueError(f"四元数无效：{path}")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"缩放无效：{path}")
            w, x, y, z = quat / np.linalg.norm(quat)
            rotation = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ])
            local = np.eye(4)
            local[:3, :3] = rotation * scale
            local[:3, 3] = pos
            world = parent @ local
            if actor.get("type") == "AssetActor":
                assets.append({
                    "name": actor["name"], "path": path,
                    "asset_path": actor.get("asset_path"),
                    "local_position": pos.tolist(),
                    "world_position": world[:3, 3].tolist(),
                })
            walk(actor.get("children", []), world)

    walk(layout["actors"], np.eye(4))
    names = [asset["name"] for asset in assets]
    if any(names.count(name) > 1 for name in REQUIRED_ACTORS):
        raise ValueError("任务 Actor 名称重复，无法唯一绑定")
    task_assets = {asset["name"]: asset for asset in assets if asset["name"] in REQUIRED_ACTORS}
    tasks = [{
        "object_name": name,
        "category": "bottle" if name in BOTTLES else "box",
        "target_bin": BLUE_BIN if name in BOTTLES else PURPLE_BIN,
        "language": "把药瓶放入蓝色筐" if name in BOTTLES else "把药盒放入紫色筐",
    } for name in (*BOTTLES, *BOXES) if name in task_assets]
    pairs = []
    for first, second in combinations(tasks, 2):
        a, b = first["object_name"], second["object_name"]
        distance = np.linalg.norm(np.array(task_assets[a]["world_position"]) - task_assets[b]["world_position"])
        pairs.append({"first": a, "second": b, "origin_distance_m": float(distance)})
    return {
        "layout_version": layout.get("version"), "level": layout.get("level"),
        "asset_count": len(assets), "assets": assets, "tasks": tasks,
        "missing_task_actors": [name for name in REQUIRED_ACTORS if name not in task_assets],
        "medicine_origin_distances": sorted(pairs, key=lambda pair: pair["origin_distance_m"]),
        "physics_verified": False,
    }


def match_body_names(actor_name, available, *, asset_path=None):
    """按 Actor 名称匹配；必要时按资产路径末段匹配 OrcaLab 生成的 body 名称。

    部分可交互资产加载后会被 OrcaLab 重命名为 ``[数字]_AssetName`` 或
    ``ParentAsset_AssetName``，这类名称不会保留 Layout 中的 Actor 名称。
    资产路径末段是稳定的逻辑标识，因此只在 Actor 名称没有命中时使用它，
    并要求 token 前后是名称边界，避免短编号误匹配更长编号。
    """
    direct = sorted(
        name for name in available
        if name == actor_name or name.startswith((actor_name + "_", actor_name + "/"))
    )
    if direct or not asset_path:
        return direct

    token = asset_path.rstrip("/").rsplit("/", 1)[-1]
    if not token:
        return []
    boundary = re.compile(rf"(?:^|_){re.escape(token)}(?:$|_)", re.IGNORECASE)
    return sorted(name for name in available if boundary.search(name))


def validate_observation(qpos, qvel, qacc, *, nq, nv):
    for name, values, length in (("qpos", qpos, nq), ("qvel", qvel, nv), ("qacc", qacc, nv)):
        array = np.asarray(values, dtype=float)
        if array.shape != (length,):
            raise ValueError(f"{name} 维度异常：{array.shape}，预期 {(length,)}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} 包含非有限数值")
