"""Load an OrcaLab layout JSON and publish it together with the demo G1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene


@dataclass(frozen=True)
class LayoutActorSpec:
    """Validated subset of an OrcaLab layout actor required by ``Actor``."""

    name: str
    asset_path: str
    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    scale: float


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must be a list of {size} numbers")
    vector = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{label} must contain only finite values")
    return vector


def load_layout_actor_specs(layout_path: str | Path) -> list[LayoutActorSpec]:
    """Read and validate the AssetActor entries in an OrcaLab v3 layout."""
    path = Path(layout_path).expanduser().resolve()
    with path.open(encoding="utf-8") as layout_file:
        layout = json.load(layout_file)

    if str(layout.get("version")) != "3.0":
        raise ValueError(f"unsupported layout version: {layout.get('version')!r}; expected '3.0'")
    raw_actors = layout.get("actors")
    if not isinstance(raw_actors, list) or not raw_actors:
        raise ValueError("layout must contain at least one actor")

    specs: list[LayoutActorSpec] = []
    names: set[str] = set()
    for index, raw_actor in enumerate(raw_actors):
        label = f"actors[{index}]"
        if not isinstance(raw_actor, dict):
            raise ValueError(f"{label} must be an object")
        if raw_actor.get("type") != "AssetActor":
            raise ValueError(f"{label} has unsupported type {raw_actor.get('type')!r}")

        name = raw_actor.get("name")
        asset_path = raw_actor.get("asset_path")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate actor name in layout: {name!r}")
        if not isinstance(asset_path, str) or not asset_path:
            raise ValueError(f"{label}.asset_path must be a non-empty string")

        transform = raw_actor.get("transform")
        if not isinstance(transform, dict):
            raise ValueError(f"{label}.transform must be an object")
        position = _finite_vector(transform.get("position"), 3, f"{label}.transform.position")
        rotation = _finite_vector(transform.get("rotation"), 4, f"{label}.transform.rotation")
        scale = float(transform.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{label}.transform.scale must be a positive finite number")

        names.add(name)
        specs.append(
            LayoutActorSpec(
                name=name,
                asset_path=asset_path,
                position=(position[0], position[1], position[2]),
                rotation_wxyz=(rotation[0], rotation[1], rotation[2], rotation[3]),
                scale=scale,
            )
        )
    return specs


def publish_layout_with_g1(
    *,
    grpc_addr: str,
    layout_path: str | Path,
    g1_actor_name: str,
    g1_asset_path: str,
    g1_position_xyz: tuple[float, float, float],
    g1_rotation_wxyz: tuple[float, float, float, float],
    include_layout_actors: bool = False,
) -> int:
    """Keep a running manual Layout and publish one code-generated G1.

    ``include_layout_actors`` exists only for simple, flat layouts.  OrcaLab
    v3 editor layouts commonly contain nested GroupActor entries and should be
    opened manually before this function publishes G1.
    """
    path = Path(layout_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    layout_specs = load_layout_actor_specs(path) if include_layout_actors else []
    if g1_actor_name in {spec.name for spec in layout_specs}:
        raise ValueError(f"G1 actor name collides with a layout actor: {g1_actor_name!r}")

    scene = OrcaGymScene(grpc_addr=grpc_addr)
    try:
        scene.publish_scene()
        for spec in layout_specs:
            scene.add_actor(
                Actor(
                    name=spec.name,
                    asset_path=spec.asset_path,
                    position=np.asarray(spec.position, dtype=np.float64),
                    rotation=np.asarray(spec.rotation_wxyz, dtype=np.float64),
                    scale=spec.scale,
                )
            )
        scene.add_actor(
            Actor(
                name=g1_actor_name,
                asset_path=g1_asset_path,
                position=np.asarray(g1_position_xyz, dtype=np.float64),
                rotation=np.asarray(g1_rotation_wxyz, dtype=np.float64),
                scale=1.0,
            )
        )
        scene.publish_scene()
    finally:
        scene.close()
    return len(layout_specs)
