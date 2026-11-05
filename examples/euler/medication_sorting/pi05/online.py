"""Pure online boundaries: native state order, bounded targets, simulation-time queue."""

import numpy as np


def bounded_closures(actions):
    """Auditably saturate small model extrapolation, never change arm coordinates."""
    actions = np.array(actions, dtype=float, copy=True)
    if actions.shape != (50, 16) or not np.isfinite(actions).all():
        raise ValueError("Invalid prediction")
    closure = actions[:, [7, 15]]
    if np.any(closure < -0.1) or np.any(closure > 1.1):
        raise ValueError("Gripper extrapolation exceeds 0.1 tolerance")
    count = int(np.count_nonzero((closure < 0) | (closure > 1)))
    actions[:, [7, 15]] = np.clip(closure, 0, 1)
    return actions, count


def state_vector(qpos, arms, grippers):
    result = np.concatenate([qpos[ids] for a, g in zip(arms, grippers) for ids in (a, g)])
    if result.shape != (30,) or not np.isfinite(result).all():
        raise ValueError("Expected finite state30")
    return result


def checked_action(action, previous, limits, max_delta, *, max_closure_delta=None):
    action = np.asarray(action, dtype=float)
    previous, limits = np.asarray(previous), np.asarray(limits)
    if (
        action.shape != (16,)
        or previous.shape != (16,)
        or limits.shape != (16, 2)
        or not np.isfinite(action).all()
        or not np.isfinite(previous).all()
        or not np.isfinite(limits).all()
        or not np.isfinite(max_delta)
        or max_delta <= 0
    ):
        raise ValueError("Invalid action, limits or delta bound")
    if np.any(action < limits[:, 0]) or np.any(action > limits[:, 1]):
        raise ValueError("Action outside joint/closure limits")
    delta_bounds = np.full(16, max_delta, dtype=float)
    if max_closure_delta is not None:
        if not np.isfinite(max_closure_delta) or max_closure_delta <= 0:
            raise ValueError("Invalid closure delta bound")
        delta_bounds[[7, 15]] = max_closure_delta
    differences = np.abs(action - previous)
    if np.any(differences > delta_bounds):
        index = int(np.argmax(differences / delta_bounds))
        raise ValueError(f"Action discontinuity at {index}: {differences[index]:.4f} > {delta_bounds[index]}")
    return action.copy()


class ActionConditioner:
    """Separate closure slew limits from arm checks and optional arm low-pass.

    All rates use simulation time. Raw arm jumps are checked before smoothing;
    executed arm jumps are checked again afterwards. No position-limit violation
    is hidden by filtering. This adds lag and must be evaluated in closed loop.
    """

    def __init__(self, initial, limits, *, max_arm_delta, dt, arm_tau, close_speed, open_speed):
        values = np.asarray([dt, arm_tau, close_speed, open_speed], dtype=float)
        if not np.isfinite(values).all() or dt <= 0 or arm_tau < 0 or close_speed <= 0 or open_speed <= 0:
            raise ValueError("Invalid action conditioning parameters")
        self.requested = checked_action(initial, initial, limits, max_arm_delta)
        self.executed = self.requested.copy()
        self.limits, self.max_arm_delta = np.asarray(limits).copy(), max_arm_delta
        self.alpha = 1.0 if arm_tau == 0 else -np.expm1(-dt / arm_tau)
        self.close_step, self.open_step = close_speed * dt, open_speed * dt

    def step(self, requested):
        requested = checked_action(requested, self.requested, self.limits, self.max_arm_delta, max_closure_delta=1.0)
        executed = self.executed + self.alpha * (requested - self.executed)
        for index in (7, 15):
            delta = requested[index] - self.executed[index]
            executed[index] = self.executed[index] + np.clip(delta, -self.open_step, self.close_step)
        executed = checked_action(
            executed,
            self.executed,
            self.limits,
            self.max_arm_delta,
            max_closure_delta=max(self.close_step, self.open_step) + 1e-12,
        )
        self.requested, self.executed = requested, executed
        return executed.copy()


class ActionQueue:
    """Action t belongs to the observation's simulation step plus t, never arrival time."""

    def __init__(self):
        self.actions = {}

    def replace(self, actions, observed_step, current_step, blend_steps=0):
        actions = np.asarray(actions)
        if actions.shape != (50, 16) or not np.isfinite(actions).all() or observed_step > current_step:
            raise ValueError("Invalid prediction or future observation")
        age = current_step - observed_step
        if age >= len(actions):
            raise ValueError("Prediction expired")
        updated = {observed_step + i: a.copy() for i, a in enumerate(actions) if i >= age}
        if blend_steps:
            overlap = []
            for i in range(blend_steps):
                step = current_step + i
                if step not in self.actions or step not in updated:
                    break
                overlap.append(step)
            if len(overlap) < 2:
                raise ValueError("Insufficient queued overlap for transition")
            for i, step in enumerate(overlap):
                t = i / (len(overlap) - 1)
                weight = 10 * t**3 - 15 * t**4 + 6 * t**5
                updated[step] = (1 - weight) * self.actions[step] + weight * updated[step]
        self.actions = updated
        return age

    def pop(self, step):
        if step not in self.actions:
            raise RuntimeError("Action queue exhausted; stop stepping, never replay")
        return self.actions.pop(step)

    def remaining(self, step):
        return sum(i >= step for i in self.actions)
