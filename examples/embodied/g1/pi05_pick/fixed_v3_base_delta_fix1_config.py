"""π0.5 base-model training config with the corrected action transform."""

import dataclasses
from pathlib import Path

from openpi.training import config, optimizer, weight_loaders

from serve_pi05_g1_delta_fixed import G1DataConfigFixed


ROOT = Path("/root/gpufree-data/SY")
BASE_PARAMS = ROOT / "cache/openpi-assets/checkpoints/pi05_base/params"
ASSET_ROOT = ROOT / "openpi/assets/pi05_fixed_v3_base"
CHECKPOINT_ROOT = ROOT / "checkpoints"
EXP_NAME = "pi05_fixed_v3_base_delta_fix1_20000"


def get_config():
    previous = config.get_config("pi05_centered_lora")
    data = G1DataConfigFixed(
        repo_id="blue_intel_fixed_v3",
        assets=config.AssetsConfig(
            assets_dir=str(ASSET_ROOT),
            asset_id="blue_intel_fixed_v3",
        ),
        base_config=config.DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    )
    return dataclasses.replace(
        previous,
        name="pi05_fixed_v3_base_delta_fix1",
        exp_name=EXP_NAME,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
        assets_base_dir=str(ASSET_ROOT),
        checkpoint_base_dir=str(CHECKPOINT_ROOT),
        num_train_steps=20_000,
        save_interval=1_000,
        keep_period=20_000,
        overwrite=False,
        resume=False,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-5,
            decay_steps=20_000,
            decay_lr=1e-6,
        ),
    )
