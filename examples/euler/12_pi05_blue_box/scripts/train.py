"""Use the original LeRobot trainer with an explicitly pinned local tokenizer."""

import argparse
import importlib.metadata
import os
from pathlib import Path


def pinned_processor_factory(factory, tokenizer):
    def create(*args, **kwargs):
        overrides = dict(kwargs.get("preprocessor_overrides") or {})
        overrides["tokenizer_processor"] = {
            **overrides.get("tokenizer_processor", {}),
            "tokenizer_name": str(tokenizer),
        }
        kwargs["preprocessor_overrides"] = overrides
        return factory(*args, **kwargs)
    return create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = args.tokenizer.resolve()
    if not (tokenizer / "tokenizer_config.json").is_file():
        parser.error("Expected a local tokenizer snapshot with tokenizer_config.json")
    if importlib.metadata.version("lerobot") != "0.6.1":
        parser.error("Expected LeRobot 0.6.1")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train

    cfg = draccus.parse(
        config_class=TrainPipelineConfig,
        args=[f"--config_path={args.config_path.resolve()}"],
    )
    # LeRobot 0.6.1 has no tokenizer revision setting. Override only its source.
    lerobot_train.make_pre_post_processors = pinned_processor_factory(
        lerobot_train.make_pre_post_processors, tokenizer,
    )
    lerobot_train.train(cfg)


if __name__ == "__main__":
    main()
