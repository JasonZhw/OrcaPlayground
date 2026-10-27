"""Cloud entry points using the pinned OpenPI checkout's public functions."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--openpi-dir", type=Path, required=True)
    result.add_argument("--data-home", type=Path, required=True, help="Parent of local/g1_lightblue_box_tray_v1")
    result.add_argument("--assets-dir", type=Path, required=True)
    result.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    result.add_argument("--batch-size", type=positive_int, default=4)
    result.add_argument("--num-workers", type=int, choices=range(0, 9), default=0)
    commands = result.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("data-smoke", help="One real OpenPI loader batch; no model weights")
    smoke.add_argument("--skip-norm-stats", action="store_true", help="Only for the initial pre-statistics check")
    commands.add_parser("norm", help="Compute state15/action9 statistics over every frame")
    train = commands.add_parser("train", help="Run the upstream JAX LoRA trainer")
    train.add_argument("--exp-name", required=True, help="New experiment directory name")
    train.add_argument("--steps", type=positive_int, default=10)
    train.add_argument("--resume", action="store_true")
    infer = commands.add_parser("infer", help="Predict one dataset observation; never actuate the simulator")
    infer.add_argument("--checkpoint", type=Path, required=True,
                       help="Checkpoint step directory containing params/assets")
    infer.add_argument("--index", type=int, default=0, help="Global dataset frame index")
    infer.add_argument("--output", type=Path, required=True, help="New .npy file for the (50,9) absolute action chunk")
    return result


def prepare(args):
    # LeRobot reads this variable at import time. Set it before importing OpenPI.
    os.environ["HF_LEROBOT_HOME"] = str(args.data_home.resolve())
    root = args.openpi_dir.resolve()
    for required in ["src/openpi/training/config.py", "scripts/train.py", "scripts/compute_norm_stats.py"]:
        if not (root / required).is_file():
            raise FileNotFoundError(f"Incomplete OpenPI checkout: {root / required}")
    sys.path[:0] = [str(root / "src"), str(root), str(Path(__file__).resolve().parent.parent)]
    from pi05_training.config import REPO_ID, make_config

    info_path = args.data_home / REPO_ID / "meta/info.json"
    info = json.loads(info_path.read_text())
    expected = {"observation.state": [15], "action": [9],
                "observation.images.head": [224, 224, 3], "observation.images.right_wrist": [224, 224, 3]}
    if info["codebase_version"] != "v2.1" or info["fps"] != 50:
        raise ValueError(f"Expected a local LeRobot v2.1 dataset at 50 fps: {info_path}")
    for key, shape in expected.items():
        if info["features"][key]["shape"] != shape:
            raise ValueError(f"Unexpected dataset feature shape for {key}: {info['features'][key]}")
    return make_config(
        assets_dir=args.assets_dir.resolve(), checkpoint_dir=args.checkpoint_dir.resolve(),
        exp_name=getattr(args, "exp_name", "inspect"), batch_size=args.batch_size,
        num_workers=args.num_workers, num_train_steps=getattr(args, "steps", 10),
        resume=getattr(args, "resume", False),
    )


def smoke(config, skip_norm_stats):
    from openpi.training import data_loader

    loader = data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=skip_norm_stats)
    observation, actions = next(iter(loader))
    state, actions = np.asarray(observation.state), np.asarray(actions)
    if state.shape != (config.batch_size, 32) or actions.shape != (config.batch_size, 50, 32):
        raise ValueError(f"Unexpected model shapes: state={state.shape}, actions={actions.shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Model inputs contain nonfinite values")
    if state[:, 15:].any() or actions[:, :, 9:].any():
        raise ValueError("Padding changed the raw state15/action9 contract")
    for name, present in [("base_0_rgb", True), ("left_wrist_0_rgb", False), ("right_wrist_0_rgb", True)]:
        image = np.asarray(observation.images[name])
        if image.shape != (config.batch_size, 224, 224, 3) or not np.isfinite(image).all():
            raise ValueError(f"Unexpected model image: {name}, {image.shape}")
        if not np.all(np.asarray(observation.image_masks[name]) == present):
            raise ValueError(f"Unexpected image mask: {name}")
    print(json.dumps({"state": list(state.shape), "actions": list(actions.shape),
                      "images": {k: list(v.shape) for k, v in observation.images.items()},
                      "prompt_tokens": list(observation.tokenized_prompt.shape),
                      "normalization": "skipped" if skip_norm_stats else "dataset quantiles",
                      "weights_loaded": False}, indent=2))


def compute_stats(config):
    from openpi.shared import normalize
    from tqdm import tqdm

    output_dir = config.assets_dirs / config.data.repo_id
    if (output_dir / "norm_stats.json").exists():
        raise FileExistsError(f"Stats already exist; select a new --assets-dir to recompute: {output_dir}")
    script = importlib.import_module("scripts.compute_norm_stats")
    data_config = config.data.create(config.assets_dirs, config.model)
    # The upstream loader drops incomplete batches. Batch one includes every frame.
    loader, count = script.create_torch_dataloader(
        data_config, config.model.action_horizon, 1, config.model, config.num_workers,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm(loader, total=count, desc="Dataset norm stats"):
        for key, accumulator in stats.items():
            accumulator.update(np.asarray(batch[key]))
    normalize.save(output_dir, {key: accumulator.get_statistics() for key, accumulator in stats.items()})
    print(f"Saved state15/action9 quantiles from {count} frames to {output_dir / 'norm_stats.json'}")


def train(config):
    import jax

    if Path(config.exp_name).name != config.exp_name or config.exp_name in (".", ".."):
        raise ValueError("--exp-name must be a single directory name")
    if config.checkpoint_dir.exists() and not config.resume:
        raise FileExistsError(f"Use a new --exp-name or explicit --resume: {config.checkpoint_dir}")
    if config.resume and not config.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"No experiment to resume: {config.checkpoint_dir}")
    if not (config.assets_dirs / config.data.repo_id / "norm_stats.json").is_file():
        raise FileNotFoundError("Run the norm command before training")
    if jax.device_count() != 1 or jax.default_backend() != "gpu":
        raise RuntimeError("This pilot is configured for one visible CUDA GPU; set CUDA_VISIBLE_DEVICES=0")
    print(f"Checkpoint directory: {config.checkpoint_dir}", flush=True)
    importlib.import_module("scripts.train").main(config)


def infer(config, args):
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training import data_loader

    if args.output.suffix != ".npy":
        raise ValueError("--output must end with .npy")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / "assets").is_dir():
        raise FileNotFoundError("--checkpoint must be a trained step directory with params/ and assets/")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if not 0 <= args.index < len(dataset):
        raise ValueError(f"--index must be in [0,{len(dataset)})")
    raw = dict(dataset[args.index])
    raw.pop("action", None)
    policy = create_trained_policy(config, args.checkpoint)
    actions = np.asarray(policy.infer(raw)["actions"])
    if actions.shape != (50, 9) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid inference actions: {actions.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, actions, allow_pickle=False)
    print(f"Saved absolute arm7 + gripper(a,d) targets {actions.shape} to {args.output}")


def main():
    args = parser().parse_args()
    config = prepare(args)
    if args.command == "data-smoke":
        smoke(config, args.skip_norm_stats)
    elif args.command == "norm":
        compute_stats(config)
    elif args.command == "train":
        train(config)
    else:
        infer(config, args)


if __name__ == "__main__":
    main()
