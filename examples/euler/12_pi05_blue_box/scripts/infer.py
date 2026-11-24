"""Predict one action chunk without controlling a robot."""

import argparse
import importlib.metadata
from pathlib import Path

from image_preprocessing import prepare_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=2026110700)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Prediction destination already exists")
    if importlib.metadata.version("lerobot") != "0.6.1":
        parser.error("This entry point requires the verified LeRobot 0.6.1 API")

    import numpy as np
    import torch
    from safetensors.torch import load_file
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401 - register processors
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition, transition_to_policy_action,
    )

    if not torch.cuda.is_available():
        parser.error("CUDA is required for the recorded inference setup")
    checkpoint = args.checkpoint.resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if (
        config.type != "pi05"
        or config.use_peft
        or config.freeze_vision_encoder
        or config.train_expert_only
        or config.use_relative_actions
        or config.chunk_size != 50
        or config.n_action_steps != 50
        or config.num_inference_steps != 10
        or any(config.normalization_mapping[key].value != "MEAN_STD"
               for key in ("STATE", "ACTION"))
    ):
        raise ValueError("Checkpoint does not match the recorded full-finetuning protocol")
    config.device = "cuda"
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    policy = PI05Policy(config)
    state_dict = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    policy.load_state_dict(state_dict, strict=True)
    del state_dict
    policy.eval()
    overrides = {}
    if args.tokenizer:
        overrides["tokenizer_processor"] = {"tokenizer_name": str(args.tokenizer.resolve())}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_preprocessor.json",
        local_files_only=True, overrides=overrides,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_postprocessor.json", local_files_only=True,
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    with np.load(args.observation, allow_pickle=False) as observation:
        state = observation["state"]
        if state.shape != (9,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError("state must be finite float32[9]")
        batch = {
            "observation.state": torch.from_numpy(state.copy()),
            "task": args.task,
        }
        for source, target in (
            ("image", "observation.images.image"),
            ("wrist_image", "observation.images.wrist_image"),
        ):
            image = observation[source]
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"{source} must be uint8[480,640,3] RGB")
            resized = prepare_camera(image)
            batch[target] = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy.reset()
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(preprocessor(batch))
        actions = postprocessor(normalized).squeeze(0).float().cpu().numpy()
        normalized_array = normalized.squeeze(0).float().cpu().numpy()
    if actions.shape != (50, 9) or not np.isfinite(actions).all():
        raise ValueError("Invalid action chunk")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream, actions=actions, normalized_actions=normalized_array,
            seed=np.asarray(args.seed), task=np.asarray(args.task),
        )
    print(f"Saved {actions.shape} absolute actions: {args.output.resolve()}")


if __name__ == "__main__":
    main()
