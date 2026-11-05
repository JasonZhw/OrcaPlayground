"""Read-only data checks, norms, bounded training and offline prediction; no actuation."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

from .adapter import CAMERAS, MedicationInputs, validate_info
from .guards import verify_norm, verify_source, write_launch_audit


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-dir", type=Path, required=True)
    p.add_argument("--data-home", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--base-params", type=Path, required=True)
    p.add_argument("--batch-size", type=positive_int, default=2)
    p.add_argument("--num-workers", type=int, choices=range(9), default=0)
    commands = p.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("data-smoke")
    smoke.add_argument("--skip-norm-stats", action="store_true")
    smoke.add_argument("--output", type=Path, required=True)
    commands.add_parser("norm")
    train = commands.add_parser("train")
    train.add_argument("--exp-name", required=True)
    train.add_argument("--steps", type=positive_int, default=10)
    train.add_argument("--resume", action="store_true")
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--index", type=int, default=0)
    infer.add_argument("--output", type=Path, required=True)
    return p


def prepare(args):
    # LeRobot resolves this at import time, before any config/reader import.
    os.environ["HF_LEROBOT_HOME"] = str(args.data_home.resolve())
    root = args.openpi_dir.resolve()
    verify_source(root)
    provenance = json.loads((root.parent / "source_provenance.json").read_text())
    if provenance.get("revision") != "215abfb217dbac7d5f1273282331b9b1866c0479":
        raise ValueError("Wrong OpenPI revision")
    if hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest() != provenance["lock_sha256"]:
        raise ValueError("OpenPI lock file changed")
    sys.path[:0] = [str(root / "src"), str(root), str(root / "packages/openpi-client/src")]
    from .config import LEROBOT_REVISION, REPO_ID, make_config

    direct = json.loads(importlib.metadata.distribution("lerobot").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != LEROBOT_REVISION:
        raise ValueError("Installed LeRobot must come from the pinned git revision")
    data_root = args.data_home / REPO_ID
    validate_info(
        json.loads((data_root / "meta/info.json").read_text()), json.loads((data_root / "validation.json").read_text())
    )
    name = getattr(args, "exp_name", "inspect")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Experiment must be a single directory name")
    config = make_config(
        assets_dir=args.assets_dir.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        base_params=args.base_params.resolve(),
        exp_name=name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=getattr(args, "steps", 10),
        resume=getattr(args, "resume", False),
    )
    if args.command in ("train", "infer") or (args.command == "data-smoke" and not args.skip_norm_stats):
        verify_norm(
            config.assets_dirs / config.data.repo_id, data_root, json.loads((data_root / "meta/info.json").read_text())
        )
    return config, data_root


def smoke(config, args):
    import jax
    from openpi.models import model
    from openpi.training import data_loader
    from PIL import Image

    if args.output.exists():
        raise FileExistsError(args.output)
    loader = data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=args.skip_norm_stats)
    observation, actions = next(iter(loader))
    state, actions = np.asarray(observation.state), np.asarray(actions)
    if state.shape != (config.batch_size, 32) or actions.shape != (config.batch_size, 50, 32):
        raise ValueError(f"Wrong model shapes: {state.shape}, {actions.shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite batch")
    if state[:, 30:].any() or actions[:, :, 16:].any():
        raise ValueError("Nonzero padding")
    for slot in CAMERAS.values():
        image = np.asarray(observation.images[slot])
        if image.shape != (config.batch_size, 224, 224, 3) or not np.isfinite(image).all():
            raise ValueError(f"Wrong image shape/range: {slot}")
        if image.min() < -1 or image.max() > 1 or not np.asarray(observation.image_masks[slot]).all():
            raise ValueError(f"Wrong image range/mask: {slot}")
    args.output.mkdir(parents=True)
    for label, obs in [
        ("input", observation),
        ("train_augmented", model.preprocess_observation(jax.random.key(0), observation, train=True)),
    ]:
        row = np.concatenate([np.asarray(obs.images[slot])[0] for slot in CAMERAS.values()], axis=1)
        Image.fromarray(np.clip((row + 1) * 127.5, 0, 255).astype(np.uint8)).save(args.output / f"{label}.png")
    report = dict(
        state=list(state.shape),
        actions=list(actions.shape),
        images={k: list(v.shape) for k, v in observation.images.items()},
        prompt_tokens=list(observation.tokenized_prompt.shape),
        normalization="skipped" if args.skip_norm_stats else "own quantiles",
        weights_loaded=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def train(config, args):
    import jax
    from flax import nnx, traverse_util

    if config.checkpoint_dir.exists() and not config.resume:
        raise FileExistsError(f"Experiment already exists: {config.checkpoint_dir}")
    if config.resume and not config.checkpoint_dir.exists():
        raise FileNotFoundError("Cannot resume a missing experiment")
    if not (config.assets_dirs / config.data.repo_id / "norm_stats.json").is_file():
        raise FileNotFoundError("Run norm first")
    if not args.base_params.is_dir() or not (args.base_params / "_METADATA").is_file():
        raise FileNotFoundError("Verified local pi05_base parameters are required")
    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("One visible JAX CUDA GPU is required")
    abstract = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable = nnx.state(abstract).filter(config.trainable_filter).to_pure_dict()
    flat = traverse_util.flatten_dict(trainable, sep="/")
    shapes = {k: list(v.shape) for k, v in flat.items()}
    if not any("lora" in k for k in shapes):
        raise ValueError("No trainable LoRA parameters found")
    audit = dict(
        config=repr(config),
        trainable_shapes=shapes,
        trainable_parameters=sum(int(np.prod(v.shape)) for v in flat.values()),
        devices=[str(x) for x in jax.devices()],
        policy_metadata=config.policy_metadata,
    )
    write_launch_audit(args.checkpoint_dir, config.exp_name, config.resume, audit)
    importlib.import_module("scripts.train").main(config)


def infer(config, args):
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training import data_loader

    if args.output.exists() or args.output.suffix != ".npy":
        raise ValueError("Choose a new .npy output file")
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / "assets").is_dir():
        raise FileNotFoundError("Checkpoint must contain params and its own assets")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, 50, config.model)
    if not 0 <= args.index < len(dataset):
        raise ValueError("Observation index out of range")
    raw = dict(dataset[args.index])
    raw.pop("action", None)
    MedicationInputs()(raw)
    policy = create_trained_policy(config, args.checkpoint)
    actions = np.asarray(policy.infer(raw)["actions"])
    if actions.shape != (50, 16) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid prediction: {actions.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        np.save(f, actions, allow_pickle=False)
    report = dict(
        shape=list(actions.shape),
        finite=True,
        checkpoint=str(args.checkpoint),
        index=args.index,
        closure_min=float(actions[:, [7, 15]].min()),
        closure_max=float(actions[:, [7, 15]].max()),
        simulator_actuated=False,
        task_success_verified=False,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    args = parser().parse_args()
    config, data_root = prepare(args)
    if args.command == "data-smoke":
        smoke(config, args)
    elif args.command == "norm":
        from .statistics import compute_stats

        compute_stats(config, data_root)
    elif args.command == "train":
        train(config, args)
    else:
        infer(config, args)


if __name__ == "__main__":
    main()
