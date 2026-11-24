"""Check the portable fixed-base scene through Euler's public API, without RGB."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from environment import DualArmEnv
from placement import Placement


def main():
    env = DualArmEnv(offline=True)
    try:
        env.restore()
        placement = Placement(env)
        before = env.observe()
        for _ in range(50):
            env.step(env.setpoint)
        after = env.observe()
        if not np.array_equal(before["pelvis"], after["pelvis"]):
            raise RuntimeError("Fixed base moved")
        if len(env.action_ids) != 9:
            raise RuntimeError("Expected nine policy action dimensions")
        print({"frames": 50, "simulated_seconds": float(env.data.time),
               "action_dim": 9, "basket_interior_min": placement.inner_min.tolist(),
               "basket_interior_max": placement.inner_max.tolist(),
               "placement": placement.measure(env), "offline_rgb": False})
    finally:
        env.close()


if __name__ == "__main__":
    main()
