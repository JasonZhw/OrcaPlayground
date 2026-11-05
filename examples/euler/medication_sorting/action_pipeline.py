"""Controlled oracle-action experiment using the production online primitives.

No images, model inference or network: each predicted chunk is the correctly
indexed future demonstration. Arrival delay is explicit in simulation ticks.
"""

import numpy as np

from .pi05.online import ActionConditioner, ActionQueue, bounded_closures


def process_recorded_actions(actions, initial, limits, mode, latency_steps=11):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 16 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected nonempty finite (N,16) actions")
    if mode not in ("queue", "inference", "arm_only", "gripper_only"):
        raise ValueError("Unknown processing mode")
    if not isinstance(latency_steps, int) or not 1 <= latency_steps <= 28:
        raise ValueError("Latency must allow at least two overlap steps (1..28)")
    conditioner = ActionConditioner(
        initial,
        limits,
        max_arm_delta=0.08,
        dt=0.02,
        arm_tau=0.06 if mode in ("inference", "arm_only") else 0,
        close_speed=0.625 if mode in ("inference", "gripper_only") else 50,
        open_speed=1.6 if mode in ("inference", "gripper_only") else 50,
    )

    def chunk(observed):
        # Terminal padding never extends the physical trial beyond its last action.
        rows = np.minimum(np.arange(observed, observed + 50), len(actions) - 1)
        return bounded_closures(actions[rows])[0]

    queue = ActionQueue()
    queue.replace(chunk(0), 0, 0)
    events = [dict(observed_step=0, received_step=0, discarded=0)]
    pending = None
    requested, executed = [], []
    for step in range(len(actions)):
        if pending is not None and step == pending + latency_steps:
            age = queue.replace(chunk(pending), pending, step, blend_steps=10)
            events.append(dict(observed_step=pending, received_step=step, discarded=age))
            pending = None
        value = queue.pop(step)
        requested.append(value)
        executed.append(conditioner.step(value))
        if step + 1 < len(actions) and pending is None and queue.remaining(step + 1) <= 30:
            pending = step + 1
    return np.array(requested), np.array(executed), events
