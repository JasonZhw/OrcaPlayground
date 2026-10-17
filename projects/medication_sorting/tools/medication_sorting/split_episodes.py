"""将已验收的完整药品回合实际分成训练/验证输入目录。

项目唯一解释器；不连接仿真、MCP/gRPC，不加载模型或使用 GPU：
    /home/pc/ZXJ/SY/.venv/bin/python /home/pc/ZXJ/SY/tools/medication_sorting/split_episodes.py --input /absolute/final_raw_session --output /absolute/new_split

默认至少 20 个独立内容训练组和 1 个独立验证组。先逐回合执行
verify_episodes，失败/中断/仍写入的回合仅进 excluded；不足样本时返回
非零且不创建输出。按 content_sha256 稳定排序并整组分配，重复副本
不能分散到两边，也不能充当 20 个独立训练样本。优先按 20% 留验证，
但始终保留至少 20 个独立训练组。组哈希证明精确内容分离，不能证明
任务、初始位置或场景分布具有额外泛化能力。

可选同时传入 --per-object-train 32 --per-object-validation 8，启用严格
四物品分层：每个 task_specs.TARGETS 目标恰好 40 个合格完整回合，
全局内容和非负整数 seed 均不得重复，randomization_profile 必须显式
提供且全局一致。每层按 content_sha256 排序，前 32 个训练、后 8 个
验证；不截取过量回合，也不以未通过验收的回合补足数量。

输出 train/episode_* 和 validation/episode_* 为指向原始证据目录的
符号链接，原始 metadata/NPZ/PNG/report/trace 不修改、不复制。转换器
和验收器显式扫描这些直接子回合链接。split_manifest.json 记录组哈希、
源目录、链接、排除项、验收警告、频率和两套独立 LeRobot repo_id。
输出须为新目录且位于输入树之外；不会覆盖既有划分。源目录需保留。

随后分别转换 train 与 validation，不能把 split 根目录作为一个训练集：
    /home/pc/ZXJ/SY/.venv/bin/python -m tools.medication_sorting.convert_episodes --input /absolute/new_split/train --output /absolute/cache/local/g1_medication_train --repo-id local/g1_medication_train --fps 5 --terminal-phase settle_final
    /home/pc/ZXJ/SY/.venv/bin/python -m tools.medication_sorting.convert_episodes --input /absolute/new_split/validation --output /absolute/cache/local/g1_medication_validation --repo-id local/g1_medication_validation --fps 5 --terminal-phase settle_final

fps 以输出清单为准。训练配置及 norm stats 只能使用 train repo_id；
validation 保持独立，仅用于验证。本脚本实际建立输入目录，但不会
安装依赖、转换、训练，或宣称现有 OpenPI loader 已改用训练集。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.medication_sorting.convert_episodes import scan_episodes
from tools.medication_sorting.pi05_adapter import validate_repo_id
from tools.medication_sorting.task_specs import TARGETS
from tools.medication_sorting.verify_episodes import verify_dataset, verify_episode


def _stratification_metadata(path):
    """只读取显式分层标签，不从路径、任务文本或默认值猜测。"""
    metadata = json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json 必须为 JSON 对象：{path}")
    object_name = metadata.get("object_name")
    if not isinstance(object_name, str) or object_name not in TARGETS:
        raise ValueError(f"metadata.object_name 必须为四物品任务目标：{path}")
    seed = metadata.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"metadata.seed 必须显式提供非负整数：{path}")
    profile = metadata.get("randomization_profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(f"metadata.randomization_profile 必须显式提供非空文本：{path}")
    return {"object_name": object_name, "seed": seed, "randomization_profile": profile}


def _stratified_partitions(episodes, train_count, validation_count):
    """严格拒绝重复和配额偏差；seed 在所有物品、所有分区间独立。"""
    strata = {name: [] for name in TARGETS}
    contents, seeds = {}, {}
    profile = None
    experiment = None
    for episode in episodes.values():
        path = episode["path"]
        episode.update(_stratification_metadata(path))
        metadata = json.loads((Path(path)/'metadata.json').read_text())
        contract = {key: metadata.get(key) for key in ('expert_profile', 'experiment_layout_sha256',
                                                      'randomization_config', 'randomized_reset', 'fps')}
        if experiment is None:
            experiment = contract
        elif contract != experiment:
            raise ValueError(f'跨回合实验配置、场景或随机范围不一致：{path}')
        digest, seed = episode["content_sha256"], episode["seed"]
        if digest in contents:
            raise ValueError(f"重复内容 content_sha256：{contents[digest]} 与 {path}")
        if seed in seeds:
            raise ValueError(f"全局 seed 冲突 {seed}：{seeds[seed]} 与 {path}")
        if profile is None:
            profile = episode["randomization_profile"]
        elif episode["randomization_profile"] != profile:
            raise ValueError(f"metadata.randomization_profile 跨回合不一致：{path}")
        contents[digest], seeds[seed] = path, path
        strata[episode["object_name"]].append(episode)
    partitions, counts = {"train": [], "validation": []}, {}
    for name in sorted(strata):
        selected = sorted(strata[name], key=lambda item: item["content_sha256"])
        if len(selected) != train_count + validation_count:
            raise ValueError(f"物品 {name} 必须恰好 40 个唯一内容、seed 独立的合格回合；"
                             f"当前 {len(selected)} 个，拒绝创建划分")
        partitions["train"].extend(selected[:train_count])
        partitions["validation"].extend(selected[train_count:])
        counts[name] = {"accepted": len(selected), "unique_groups": len(selected),
                        "train": train_count, "validation": validation_count}
    return partitions, counts


def materialize_split(source, output, *, min_train=20, min_validation=1,
                      validation_fraction=0.2, train_repo_id="local/g1_medication_train",
                      validation_repo_id="local/g1_medication_validation",
                      per_object_train=None, per_object_validation=None):
    """只分配验收通过的已有数据；不满足数量或内容发生变化时不落盘。"""
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("划分输出必须位于输入目录之外，避免回扫和链接环")
    if type(min_train) is not int or min_train < 20 or type(min_validation) is not int or min_validation < 1:
        raise ValueError("至少需要 20 个独立训练组和 1 个独立验证组，不能降低门槛")
    stratified = per_object_train is not None or per_object_validation is not None
    if stratified:
        if per_object_train is None or per_object_validation is None:
            raise ValueError("per_object_train 和 per_object_validation 必须同时提供")
        if (type(per_object_train) is not int or type(per_object_validation) is not int
                or (per_object_train, per_object_validation) != (32, 8)):
            raise ValueError("四物品分层固定为每类 32 个训练、8 个验证，共 40 个合格回合")
    validate_repo_id(train_repo_id)
    validate_repo_id(validation_repo_id)
    if train_repo_id == validation_repo_id:
        raise ValueError("训练和验证必须使用不同 repo_id")
    verified = verify_dataset(source, validation_fraction=validation_fraction)
    groups = {}
    # 同一物理源目录被多个别名扫描时，不重复计入回合数。
    episodes = {item["path"]: item for item in verified["accepted"]}
    for episode in episodes.values():
        groups.setdefault(episode["content_sha256"], []).append(episode)
    if not stratified and len(groups) < min_train + min_validation:
        raise ValueError(f"至少需要 {min_train} 个独立训练组和 {min_validation} 个独立验证组；"
                         f"当前仅 {len(groups)} 组、{len(episodes)} 个合格回合，拒绝创建划分")
    hashes = sorted(groups)
    if stratified:
        partitions, per_object = _stratified_partitions(episodes, per_object_train, per_object_validation)
        validation_count = len(partitions["validation"])
    else:
        validation_count = min(len(hashes) - min_train,
                               max(min_validation, round(len(hashes) * validation_fraction)))
        val_hashes = set(hashes[:validation_count])
        partitions = {"train": [], "validation": []}
        for digest in hashes:
            partition = "validation" if digest in val_hashes else "train"
            partitions[partition].extend(sorted(groups[digest], key=lambda item: item["path"]))
    # 链接将保留源证据，故发布前再验证源内容未在首次扫描之后改变。
    for episode in episodes.values():
        current = verify_episode(episode["path"])
        if not current["accepted"] or current.get("content_sha256") != episode["content_sha256"]:
            raise ValueError(f"源回合变化或不再合格，须重新验收：{episode['path']}")
        if stratified:
            metadata = _stratification_metadata(episode["path"])
            if metadata != {key: episode[key] for key in metadata}:
                raise ValueError(f"源回合分层元数据变化，须重新验收：{episode['path']}")
    first = json.loads((Path(next(iter(episodes))) / "metadata.json").read_text(encoding="utf-8"))
    fps = first["fps"]
    manifest = {"status": "materialized", "source": str(source), "output": str(output),
                "created_at": datetime.now(timezone.utc).isoformat(), "fps": fps,
                "train_unique_groups": len(hashes) - validation_count,
                "validation_unique_groups": validation_count, "train": [], "validation": [],
                "excluded": verified["rejected"], "training_loader_split_applied": False,
                "conversion": {"train_input": str(output / "train"), "validation_input": str(output / "validation"),
                               "train_repo_id": train_repo_id, "validation_repo_id": validation_repo_id,
                               "norm_stats_repo_id": train_repo_id},
                "evidence_policy": "只包含 verify 接受的完整回合；相同内容哈希的所有副本属于同一分区。"}
    if stratified:
        manifest["per_object"] = per_object
        manifest["evidence_policy"] = "只包含 verify 接受的完整回合；每物品按内容哈希排序前 32 个训练、后 8 个验证，全局内容及 seed 唯一。"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # 独占创建；已存在时绝不删除或覆盖。
    try:
        for partition, selected in partitions.items():
            folder = output / partition
            folder.mkdir()
            for index, episode in enumerate(selected):
                link = folder / f"episode_{index:06d}"
                link.symlink_to(episode["path"], target_is_directory=True)
                manifest[partition].append({"source": episode["path"], "link": str(link),
                                            "content_sha256": episode["content_sha256"],
                                            "frames": episode["stats"]["frames"], "warnings": episode["warnings"]})
                if stratified:
                    manifest[partition][-1].update({key: episode[key] for key in (
                        "object_name", "seed", "randomization_profile")})
            accepted, rejected = scan_episodes(folder, fps=fps, terminal_phase="settle_final")
            if len(accepted) != len(selected) or rejected:
                raise ValueError(f"{partition} 链接转换预检不一致：{rejected}")
        (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except Exception:
        # 只清理本调用独占创建的输出；rmtree 不跟随回合目录链接。
        shutil.rmtree(output)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="按验收回合内容哈希实际建立独立训练/验证链接目录；至少 20 个训练组及 1 个验证组。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-validation", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--per-object-train", type=int, help="与 --per-object-validation 同时提供，固定每物品 32 个训练回合")
    parser.add_argument("--per-object-validation", type=int, help="与 --per-object-train 同时提供，固定每物品 8 个验证回合")
    parser.add_argument("--train-repo-id", default="local/g1_medication_train")
    parser.add_argument("--validation-repo-id", default="local/g1_medication_validation")
    args = parser.parse_args(argv)
    try:
        result = materialize_split(args.input, args.output, min_train=args.min_train, min_validation=args.min_validation,
                                   validation_fraction=args.validation_fraction, train_repo_id=args.train_repo_id,
                                   validation_repo_id=args.validation_repo_id, per_object_train=args.per_object_train,
                                   per_object_validation=args.per_object_validation)
    except (ValueError, OSError) as error:
        result = {"status": "blocked", "output_created": False, "error": str(error)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
