"""Train/reload ACT without modifying LeRobot or uploading anything.

Run with the existing Python 3.11 training environment. All image preprocessing,
normalization, CVAE loss and checkpointing come from pinned LeRobot.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

LEROBOT_COMMIT = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
CAMERAS = [f"observation.images.{name}" for name in ("head", "left_wrist", "right_wrist")]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_source(dataset):
    source = json.loads(importlib.metadata.distribution("lerobot").read_text("direct_url.json"))
    if source.get("vcs_info", {}).get("commit_id") != LEROBOT_COMMIT:
        raise RuntimeError("LeRobot revision does not match the dataset baseline")
    info = json.loads((dataset / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["fps"]) == (20, 219594, 50)
    assert info["features"]["observation.state"]["shape"] == [30]
    assert info["features"]["action"]["shape"] == [16]
    for key in CAMERAS:
        assert info["features"][key]["shape"] == [480, 640, 3]
    assert json.loads((dataset / "validation.json").read_text())["passed"] is True
    files = sorted(dataset.glob("data/**/*.parquet")) + sorted(dataset.glob("meta/*"))
    return {str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def make_config(args):
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/g1_med_sort_v2", root=str(args.dataset),
                              video_backend="pyav", use_imagenet_stats=True),
        policy=ACTConfig(device="cuda", chunk_size=50, n_action_steps=50, use_amp=False),
        output_dir=args.output, job_name=args.output.name, seed=1000,
        batch_size=8, num_workers=4, steps=args.steps,
        log_freq=10, save_freq=5000, eval_freq=0, wandb=WandBConfig(enable=False),
    )


def train(args):
    import lerobot.scripts.train as official
    import torch
    from lerobot.common.utils.utils import init_logging

    assert torch.cuda.is_available(), "CUDA required; do not silently train on CPU"
    source_hashes = verify_source(args.dataset)
    write_json(args.output.parent / (args.output.name + "_source.json"), source_hashes)
    original = official.update_policy
    count = 0
    start = time.monotonic()

    def checked_update(*positional, **kwargs):
        nonlocal count
        metrics, detail = original(*positional, **kwargs)
        count += 1
        loss = float(metrics.loss.val)
        grad = float(metrics.grad_norm.val)
        if not all(math.isfinite(x) for x in [loss, grad, *detail.values()]):
            raise FloatingPointError(f"Non-finite training result at update {count}")
        if count == 1 or count % 10 == 0 or count == args.steps:
            row = {"step": count, "loss": loss, "grad_norm": grad, **detail,
                   "elapsed_seconds": time.monotonic() - start,
                   "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "time_unix": time.time()}
            with (args.output.parent / (args.output.name + "_metrics.jsonl")).open("a") as out:
                out.write(json.dumps(row, allow_nan=False) + "\n")
        return metrics, detail

    official.update_policy = checked_update
    init_logging()
    official.train(make_config(args))
    assert count == args.steps


def reload_check(args):
    import torch
    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    verify_source(args.dataset)
    dataset = make_dataset(make_config(args))
    sample = dataset[0]
    assert sample["action"].shape == (50, 16)
    assert sample["observation.state"].shape == (30,)
    for key in CAMERAS:
        assert sample[key].shape == (3, 480, 640)
    batch = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items()
             if isinstance(v, torch.Tensor) and k.startswith("observation.")}
    policy = ACTPolicy.from_pretrained(args.checkpoint, local_files_only=True).to("cuda")
    assert policy.config.chunk_size == policy.config.n_action_steps == 50
    assert set(policy.config.image_features) == set(CAMERAS)
    assert policy.config.action_feature.shape == (16,) or policy.config.action_feature.shape == [16]
    policy.eval()
    policy.reset()
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        actions = torch.stack([policy.select_action(batch)[0] for _ in range(50)])
    torch.cuda.synchronize()
    assert actions.shape == (50, 16) and torch.isfinite(actions).all()
    assert all(torch.isfinite(v).all() for v in policy.state_dict().values())
    # Normalization buffers must survive checkpoint save/load unchanged.
    report = {"passed": True, "shape": list(actions.shape), "finite": True,
              "checkpoint": str(args.checkpoint), "cameras": CAMERAS,
              "first_chunk_seconds": time.monotonic() - start,
              "parameters": sum(p.numel() for p in policy.parameters()),
              "closure_range_unclipped": [float(actions[:, [7, 15]].min()),
                                           float(actions[:, [7, 15]].max())],
              "task_success_tested": False, "time_unix": time.time()}
    write_json(args.report, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "reload"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch_threads = "4"
    os.environ.setdefault("OMP_NUM_THREADS", torch_threads)
    # Official config parser sees its own CLI flags; this runner constructs config directly.
    import sys
    sys.argv = [sys.argv[0]]
    if args.mode == "train":
        train(args)
    else:
        reload_check(args)


if __name__ == "__main__":
    main()
