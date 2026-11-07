"""Serve the corrected fixed-v3 pi0.5 checkpoint for evaluation."""

import dataclasses
import logging

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from fixed_v3_base_delta_fix1_config import get_config


CHECKPOINT = (
    "/root/gpufree-data/SY/checkpoints/"
    "pi05_fixed_v3_base_delta_fix1/"
    "pi05_fixed_v3_base_delta_fix1_20000/19999"
)


class SeededPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.rng = np.random.default_rng(0)

    def infer(self, observation):
        observation = dict(observation)
        seed = observation.pop("episode_seed", None)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        noise = self.rng.standard_normal((10, 32)).astype(np.float32)
        output = self.policy.infer(observation, noise=noise)
        if output["actions"].shape != (10, 9):
            raise ValueError(f"Wrong action shape: {output['actions'].shape}")
        if not np.isfinite(output["actions"]).all():
            raise ValueError("Non-finite policy actions")
        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    print("GPU_DEVICES", jax.devices(), flush=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("Policy inference requires the GPU backend")
    train_config = get_config()
    policy = policy_config.create_trained_policy(
        train_config, CHECKPOINT, sample_kwargs={"num_steps": 10}
    )
    print("POLICY_READY", CHECKPOINT, flush=True)
    WebsocketPolicyServer(
        policy=SeededPolicy(policy),
        host="127.0.0.1",
        port=8011,
        metadata={
            "checkpoint": CHECKPOINT,
            "action_dim": 9,
            "action_horizon": 10,
            "policy": "pi05_fixed_v3_base_delta_fix1",
        },
    ).serve_forever()
