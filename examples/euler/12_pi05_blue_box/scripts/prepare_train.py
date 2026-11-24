"""Prepare the saved full-finetuning recipe; launch only with --launch."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
SAVED_CONFIG = REPO / "configs/train_config.saved.json"
SOURCES = REPO / "configs/sources.json"
EXPECTED_FEATURES = {
    "observation.images.image": [3, 224, 224],
    "observation.images.wrist_image": [3, 224, 224],
    "observation.state": [9],
    "action": [9],
}


def validate_dataset(root, strict=True):
    metadata = root / "meta/info.json"
    with metadata.open(encoding="utf-8") as stream:
        info = json.load(stream)
    if info.get("total_episodes") != 200 or info.get("total_frames", 0) < 1:
        raise ValueError("Expected 200 nonempty episodes")
    if strict and info["total_frames"] != 277084:
        raise ValueError("Expected exactly 277084 original frames; use --new-collection for fresh data")
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 50:
        raise ValueError("Expected the reference LeRobot v3.0 dataset at 50 Hz")
    if info.get("video_path") is not None:
        raise ValueError("Expected embedded images, not the earlier video dataset")
    features = info.get("features", {})
    for name, shape in EXPECTED_FEATURES.items():
        feature = features.get(name, {})
        actual = feature.get("shape")
        if name.startswith("observation.images."):
            if feature.get("dtype") != "image":
                raise ValueError(f"Expected an embedded image feature: {name}")
            names = feature.get("names", [])
            if len(names) != 3 or actual is None or len(actual) != 3:
                raise ValueError(f"Expected named image dimensions: {name}")
            # Match LeRobot 0.6.1 dataset_to_policy_features channel ordering.
            if names[2] in {"channel", "channels"}:
                actual = [actual[2], actual[0], actual[1]]
        if actual != shape:
            raise ValueError(f"Unexpected feature shape for {name}: {feature}")
    # Counts and shapes are a preflight, not proof of dataset identity.
    return info


def make_config(saved, dataset_root, dataset_id, base_model, output_dir):
    config = json.loads(json.dumps(saved))
    policy = config["policy"]
    if (
        policy["type"] != "pi05"
        or policy["use_peft"]
        or policy["freeze_vision_encoder"]
        or policy["train_expert_only"]
        or config["peft"] is not None
    ):
        raise ValueError("Source configuration is not the expected full-finetuning recipe")
    if config["steps"] != 30000 or policy["scheduler_decay_steps"] != 30000:
        raise ValueError("Preserve the original 30000-step plan and scheduler")
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["repo_id"] = dataset_id
    config["output_dir"] = str(output_dir)
    config["job_name"] = "pi05-blue-box-full-finetune"
    config["resume"] = False
    config.pop("checkpoint_path", None)
    policy["pretrained_path"] = base_model
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    policy["pretrained_revision"] = (
        sources["base"]["revision"] if base_model == sources["base"]["repo_id"] else None
    )
    policy["push_to_hub"] = False
    policy["repo_id"] = None
    config["wandb"]["enable"] = False
    config["wandb"]["run_id"] = None
    config["save_checkpoint_to_hub"] = False
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Local snapshot downloaded at the recorded tokenizer revision")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--new-collection", action="store_true",
                        help="Accept a fresh 200-episode dataset with different frame count")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    destination = args.config_out.resolve()
    if output.exists():
        parser.error("Training output already exists; inspect it before resuming or overwriting")
    if destination.exists():
        parser.error("Configuration destination already exists")
    if destination == output or output in destination.parents:
        parser.error("Save the generated config outside the new training output directory")
    if args.launch:
        version = importlib.metadata.version("lerobot")
        if version != "0.6.1":
            parser.error(f"Expected lerobot 0.6.1, found {version}")
    validate_dataset(root, strict=not args.new_collection)
    with SAVED_CONFIG.open(encoding="utf-8") as stream:
        saved = json.load(stream)
    config = make_config(saved, root, args.dataset_id, args.base_model, output)
    command = [
        sys.executable, str(REPO / "scripts/train.py"),
        "--config-path", str(destination), "--tokenizer", str(args.tokenizer.resolve()),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(f"CONFIG: {destination}")
    print(f"COMMAND: {shlex.join(command)}")
    print("Plan: 30000 updates; evaluate checkpoint 025000 with unchanged schedule.")
    print("Preflight validates metadata only; data identity and GPU capacity need separate checks.")
    if args.launch:
        import os
        environment = dict(os.environ)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
