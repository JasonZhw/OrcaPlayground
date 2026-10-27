"""Public OpenPI config factory; no changes to the upstream config registry."""

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as training_config
from openpi.training import optimizer, weight_loaders

from .adapter import ACTION_HORIZON, MODEL_DIM, TrayInputs, TrayOutputs

CONFIG_NAME = "pi05_g1_tray_lora"
REPO_ID = "local/g1_lightblue_box_tray_v1"
OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"


def tray_transforms(model_config):
    if model_config.model_type != training_config.ModelType.PI05:
        raise ValueError("The tray configuration requires pi0.5")
    return transforms.Group(inputs=[TrayInputs()], outputs=[TrayOutputs()])


def make_config(*, assets_dir, checkpoint_dir, exp_name="inspect", batch_size=4, num_workers=0,
                num_train_steps=10, resume=False):
    model = Pi0Config(
        pi05=True,
        action_dim=MODEL_DIM,
        action_horizon=ACTION_HORIZON,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return training_config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        data=training_config.SimpleDataConfig(
            repo_id=REPO_ID,
            base_config=training_config.DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),
            data_transforms=tray_transforms,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=5e-5, decay_steps=10_000, decay_lr=5e-6,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        log_interval=1 if num_train_steps <= 10 else 10,
        save_interval=100,
        keep_period=1000,
        assets_base_dir=str(assets_dir),
        checkpoint_base_dir=str(checkpoint_dir),
        overwrite=False,
        resume=resume,
        wandb_enabled=False,
        fsdp_devices=1,
        policy_metadata={
            "state_dim": 15,
            "state_description": "measured right arm7 + measured gripper8",
            "action_dim": 9,
            "action_description": "absolute right arm7 targets + gripper_a,gripper_d",
            "gripper_expansion": "[0,0,a,d,0,0,-a,d]",
            "fps": 50,
            "action_horizon": ACTION_HORIZON,
            "openpi_revision": OPENPI_REVISION,
            "lerobot_revision": LEROBOT_REVISION,
        },
    )
