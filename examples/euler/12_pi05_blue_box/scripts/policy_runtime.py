"""Load a full pi05 checkpoint once and use its saved normalization pipelines."""

import importlib.metadata
from pathlib import Path

import numpy as np
from PIL import Image

from image_preprocessing import prepare_camera


TASK = "Pick up the blue Intel processor box and place it in the wooden basket."


class PolicyRuntime:
    def __init__(self, checkpoint, tokenizer=None, image_mode="pad"):
        import torch
        from safetensors.torch import load_file
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import lerobot.policies.pi05.processor_pi05  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition, transition_to_policy_action,
        )

        if importlib.metadata.version("lerobot") != "0.6.1":
            raise RuntimeError("Expected LeRobot 0.6.1")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if image_mode not in {"pad", "legacy-stretch"}:
            raise ValueError("Unknown camera resize mode")
        self.image_mode = image_mode
        self.torch = torch
        checkpoint = Path(checkpoint).resolve()
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        if (config.type != "pi05" or config.use_peft or config.freeze_vision_encoder
                or config.train_expert_only or config.use_relative_actions
                or config.chunk_size != 50 or config.n_action_steps != 50
                or config.num_inference_steps != 10
                or any(config.normalization_mapping[k].value != "MEAN_STD"
                       for k in ("STATE", "ACTION"))):
            raise ValueError("Checkpoint differs from the full-finetuning protocol")
        config.device = "cuda"
        torch.set_num_threads(4)
        torch.manual_seed(1000)
        self.policy = PI05Policy(config)
        weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        self.policy.load_state_dict(weights, strict=True)
        del weights
        self.policy.eval()
        overrides = {}
        if tokenizer:
            overrides["tokenizer_processor"] = {"tokenizer_name": str(Path(tokenizer).resolve())}
        self.pre = PolicyProcessorPipeline.from_pretrained(
            checkpoint, config_filename="policy_preprocessor.json",
            local_files_only=True, overrides=overrides,
        )
        self.post = PolicyProcessorPipeline.from_pretrained(
            checkpoint, config_filename="policy_postprocessor.json", local_files_only=True,
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
        )

    def reset(self, seed):
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.policy.reset()

    def predict(self, state, head, wrist, task=TASK):
        torch = self.torch
        state = np.asarray(state)
        if state.shape != (9,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError("Expected finite float32[9] measured state")
        batch = {"observation.state": torch.from_numpy(state.copy()), "task": task}
        for image, key in ((head, "observation.images.image"),
                           (wrist, "observation.images.wrist_image")):
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError("Expected uint8[480,640,3] RGB")
            resized = (prepare_camera(image) if self.image_mode == "pad" else
                       np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR)))
            batch[key] = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(self.pre(batch))
            actions = self.post(normalized).squeeze(0).float().cpu().numpy()
        if actions.shape != (50, 9) or not np.isfinite(actions).all():
            raise ValueError("Invalid predicted action chunk")
        return actions, normalized.squeeze(0).float().cpu().numpy()
