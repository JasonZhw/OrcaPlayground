"""External configuration for the fixed JAX OpenPI revision."""

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as training_config
from openpi.training import optimizer, weight_loaders

from .adapter import ACTION_HORIZON, MODEL_DIM, MedicationInputs, MedicationOutputs

CONFIG_NAME = "pi05_g1_med20_lora"
REPO_ID = "local/g1_med_sort_v2"
OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"


def medication_transforms(model_config):
    if model_config.model_type != training_config.ModelType.PI05:
        raise ValueError("Medication configuration requires pi0.5")
    return transforms.Group(inputs=[MedicationInputs()], outputs=[MedicationOutputs()])


def make_config(
    *,
    assets_dir,
    checkpoint_dir,
    base_params,
    exp_name="inspect",
    batch_size=2,
    num_workers=0,
    num_train_steps=10,
    resume=False,
):
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
            data_transforms=medication_transforms,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(base_params)),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-5, decay_steps=10000, decay_lr=5e-6),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        log_interval=1 if num_train_steps <= 10 else 10,
        save_interval=1000,
        keep_period=10000,
        assets_base_dir=str(assets_dir),
        checkpoint_base_dir=str(checkpoint_dir),
        overwrite=False,
        resume=resume,
        wandb_enabled=False,
        fsdp_devices=1,
        policy_metadata={
            "state_dim": 30,
            "action_dim": 16,
            "fps": 50,
            "action_horizon": 50,
            "state": "left arm7 + left measured gripper8 + right arm7 + right measured gripper8",
            "action": "left absolute arm7 + closure1 + right absolute arm7 + closure1",
            "gripper_mapping": "medication_closure_v1",
            "closure": "0=open, 1=expert closed command",
            "openpi_revision": OPENPI_REVISION,
            "lerobot_revision": LEROBOT_REVISION,
            "independent_validation": False,
        },
    )
