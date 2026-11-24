#!/usr/bin/env python
"""pi0.5 全量微调，配方取自 piper_train_vla 的 exp-018（lr、预热、余弦衰减；数据增强按仿真数据改为只留轻微几何扰动）。
环境变量（GRAD_CKPT、WEIGHT_DECAY、DECAY_STEPS、NORM 可选，缺省同第一版；PyPI 版 lerobot 0.6.1 没有梯度累积）：PRETRAINED DATA_ROOT OUT_DIR STEPS BATCH LR WARMUP SAVE_FREQ WANDB_RUN
"""
import os
import sys

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.scripts.lerobot_train import train


def _itc(ttype, kwargs):
    return ImageTransformConfig(weight=1.0, type=ttype, kwargs=kwargs)


# 仿真数据：相机固定、光照一致，不做光照类增强；全程只用一个轻微的几何扰动
TRANSFORMS = {"affine": _itc("RandomAffine", {"degrees": (-3.0, 3.0), "translate": (0.03, 0.03), "scale": (0.97, 1.03)})}


def main():
    e = os.environ
    if e.get("RESUME_CONFIG"):  # 续训：配置（含增强、wandb run）全部来自保存的 train_config.json
        sys.argv = [sys.argv[0], f"--config_path={e['RESUME_CONFIG']}", "--resume=true"]
        train()
        return
    steps = e["STEPS"]
    cli = [
        f"--policy.pretrained_path={e['PRETRAINED']}",
        "--policy.type=pi05",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        f"--policy.gradient_checkpointing={e.get('GRAD_CKPT', 'false')}",
        "--policy.push_to_hub=false",
        # 归一化：官方在 LIBERO 上复现 openpi 成绩用的是均值/标准差（不是 pi0.5 默认的分位数）；
        # 本数据大量帧集中在起始零位与夹爪全开/全合这些极端值上，分位数归一化把它们压在 ±1 边界上或边界外
        f"--policy.normalization_mapping={e.get('NORM', '{\"VISUAL\": \"IDENTITY\", \"STATE\": \"QUANTILES\", \"ACTION\": \"QUANTILES\"}')}",
        "--dataset.repo_id=g1-blue-intel-61",
        f"--dataset.root={e['DATA_ROOT']}",
        f"--batch_size={e['BATCH']}",
        f"--num_workers={e.get('NUM_WORKERS', '16')}",
        f"--steps={steps}",
        f"--policy.optimizer_lr={e['LR']}",
        f"--policy.optimizer_weight_decay={e.get('WEIGHT_DECAY', '0.01')}",
        f"--policy.scheduler_decay_steps={e.get('DECAY_STEPS', steps)}",
        f"--policy.scheduler_warmup_steps={e['WARMUP']}",
        f"--save_freq={e['SAVE_FREQ']}",
        "--log_freq=50",
        f"--output_dir={e['OUT_DIR']}",
        f"--job_name={e['WANDB_RUN']}",
        "--wandb.enable=true",
        "--wandb.project=g1-pi05",
    ]
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=cli)
    cfg.dataset.image_transforms = ImageTransformsConfig(
        enable=True, max_num_transforms=1, random_order=True, tfs=dict(TRANSFORMS)
    )
    train(cfg)


if __name__ == "__main__":
    sys.exit(main())
