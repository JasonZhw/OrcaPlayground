"""Solve arm targets, plan collision-checked paths, and sample joint commands."""

from collections.abc import Callable

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class IKError(RuntimeError):
    """No pose-accurate, valid solution was found within the finite budget."""


def solve_ik(
    scene,
    arm,
    position,
    rotation,
    seed_qpos,
    *,
    valid=None,
    attempts=10,
    max_nfev=180,
    position_tolerance=0.003,
    rotation_tolerance=0.035,
    random_seed=0,
    site=False,
):
    """Return a full qpos, changing only the requested seven arm joints.

    ``valid`` accepts a full qpos and can enforce phase-specific collision rules.
    If omitted, the scene's strict arm collision check is used.
    """
    position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
    seed_qpos = np.asarray(seed_qpos, dtype=float)
    if (
        position.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("Target must contain a finite position and proper rotation matrix")
    if seed_qpos.shape != (scene.m.nq,) or not np.isfinite(seed_qpos).all():
        raise ValueError("Seed must be a finite full qpos")
    if (
        not isinstance(attempts, int)
        or attempts < 1
        or not isinstance(max_nfev, int)
        or max_nfev < 1
        or not np.isfinite([position_tolerance, rotation_tolerance]).all()
        or min(position_tolerance, rotation_tolerance) <= 0
    ):
        raise ValueError("IK budgets and tolerances must be positive")
    binding = scene.arms[arm]
    m, d = scene.m, mujoco.MjData(scene.m)
    d.qpos[:] = seed_qpos
    rng = np.random.default_rng(random_seed)
    seeds = [np.clip(seed_qpos[binding.qpos_ids], binding.lower, binding.upper)]
    seeds.extend(rng.uniform(binding.lower, binding.upper, (attempts - 1, 7)))
    check = valid if valid is not None else lambda q: scene.collision_free(q, arm)

    def pose():
        if site:
            return d.site_xpos[binding.end_site_id], d.site_xmat[binding.end_site_id].reshape(3, 3)
        return d.xpos[binding.end_body_id], d.xmat[binding.end_body_id].reshape(3, 3)

    def residual(q):
        d.qpos[binding.qpos_ids] = q
        mujoco.mj_kinematics(m, d)
        p, r = pose()
        return np.r_[p - position, 0.15 * Rotation.from_matrix(rotation @ r.T).as_rotvec()]

    best_error = float("inf")
    for seed in seeds:
        result = least_squares(
            residual, seed, bounds=(binding.lower, binding.upper), max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10
        )
        error = residual(result.x)
        position_error, rotation_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:]) / 0.15
        best_error = min(best_error, position_error)
        if position_error <= position_tolerance and rotation_error <= rotation_tolerance:
            qpos = seed_qpos.copy()
            qpos[binding.qpos_ids] = result.x
            if check(qpos):
                return qpos
    raise IKError(f"No valid {arm} IK solution in {attempts} attempts (best position error {best_error:.6f} m)")


class PlanningError(RuntimeError):
    """Invalid planning input or no collision-free path within the budget."""


def _array(value, name):
    try:
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise PlanningError(f"{name} must be a numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise PlanningError(f"{name} must contain only finite values")
    return result


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise PlanningError(f"{name} must be finite and positive")


def plan_joint_path(
    start,
    goal,
    lower,
    upper,
    is_valid: Callable[[np.ndarray], bool],
    seed=0,
    max_iterations=2500,
    step_size=0.15,
    resolution=0.025,
):
    """Return an (N, DOF) path, or raise ``PlanningError``.

    ``max_iterations`` bounds all tree-extension attempts, including the greedy
    connect phase. Fixed joints may have identical lower and upper limits.
    Identical start and goal configurations return a single waypoint.
    """
    start, goal, lower, upper = (
        _array(value, name) for value, name in ((start, "start"), (goal, "goal"), (lower, "lower"), (upper, "upper"))
    )
    if start.ndim != 1 or not start.size or any(q.shape != start.shape for q in (goal, lower, upper)):
        raise PlanningError("Configurations and limits must be matching nonempty vectors")
    if np.any(lower > upper):
        raise PlanningError("Lower joint limits exceed upper limits")
    _positive(step_size, "step_size")
    _positive(resolution, "resolution")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 0
    ):
        raise PlanningError("max_iterations must be a nonnegative integer")

    def valid(q):
        return np.all(np.isfinite(q)) and np.all(q >= lower) and np.all(q <= upper) and bool(is_valid(q))

    def edge_valid(a, b):
        intervals = max(1, int(np.ceil(np.linalg.norm(b - a) / resolution)))
        return all(valid(q) for q in np.linspace(a, b, intervals + 1))

    if not valid(start) or not valid(goal):
        raise PlanningError("Start or goal is in collision or outside joint limits")
    if np.array_equal(start, goal):
        return start[None, :]
    if edge_valid(start, goal):
        return np.stack((start, goal))

    rng = np.random.default_rng(seed)
    nodes_a, parents_a = [start], [-1]
    nodes_b, parents_b = [goal], [-1]
    a_is_start = True

    def extend(nodes, parents, target):
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        origin = nodes[nearest]
        distance = np.linalg.norm(target - origin)
        if distance == 0:
            return nearest, True
        reached = distance <= step_size
        candidate = target.copy() if reached else origin + (target - origin) * (step_size / distance)
        if not edge_valid(origin, candidate):
            return None, False
        nodes.append(candidate)
        parents.append(nearest)
        return len(nodes) - 1, reached

    def trace(nodes, parents, index):
        path = []
        while index != -1:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    def finish(path):
        for _ in range(100):
            if len(path) < 3:
                break
            i, j = sorted(rng.choice(len(path), 2, replace=False))
            if j > i + 1 and edge_valid(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        if not all(edge_valid(a, b) for a, b in zip(path[:-1], path[1:])):
            raise PlanningError("Path failed final collision validation")
        return np.asarray(path)

    attempts = 0
    while attempts < max_iterations:
        target = nodes_b[0] if rng.random() < 0.1 else rng.uniform(lower, upper)
        index_a, _ = extend(nodes_a, parents_a, target)
        attempts += 1
        if index_a is not None:
            while attempts < max_iterations:
                index_b, reached = extend(nodes_b, parents_b, nodes_a[index_a])
                attempts += 1
                if index_b is None:
                    break
                if reached:
                    path_a = trace(nodes_a, parents_a, index_a)
                    path_b = trace(nodes_b, parents_b, index_b)
                    path = path_a + path_b[-2::-1]
                    return finish(path if a_is_start else path[::-1])
        nodes_a, nodes_b = nodes_b, nodes_a
        parents_a, parents_b = parents_b, parents_a
        a_is_start = not a_is_start
    raise PlanningError(f"No collision-free path within {max_iterations} extension attempts")


def smooth_blend(phase):
    """Quintic easing with zero velocity and acceleration at both endpoints."""
    phase = np.asarray(phase)
    return phase**3 * (10 + phase * (-15 + 6 * phase))


def time_samples(path, dt=0.02, max_velocity=0.5, max_acceleration=0.8):
    """Sample a collision-checked polyline with bounded speed and acceleration.

    Returns an (N, DOF) array including the first and final waypoints. Each
    Each segment uses quintic easing and stops at its waypoints without changing
    the geometric path. Peak blend derivatives are 1.875 and 10/sqrt(3).
    Duration rounds up to whole periods; duplicate waypoints add no samples.
    """
    path = _array(path, "path")
    if path.ndim != 2 or not all(path.shape):
        raise PlanningError("path must be a nonempty (N, DOF) array")
    _positive(dt, "dt")
    _positive(max_velocity, "max_velocity")
    _positive(max_acceleration, "max_acceleration")
    samples = [path[:1]]
    for start, goal in zip(path[:-1], path[1:]):
        distance = np.max(np.abs(goal - start))
        if distance == 0:
            continue
        duration = max(1.875 * distance / max_velocity, np.sqrt(10 / np.sqrt(3) * distance / max_acceleration))
        count = max(1, int(np.ceil(duration / dt)))
        phase = np.arange(1, count + 1, dtype=float) / count
        blend = smooth_blend(phase)
        segment = start + blend[:, None] * (goal - start)
        segment[-1] = goal
        samples.append(segment)
    return np.concatenate(samples)


def plan_motion(scene, arm, position, rotation, qpos, *, target=None, carried=None, allow=False, seed=0):
    """Convert one world-space hand target into a collision-checked joint path."""
    binding = scene.arms[arm]

    def valid_full(candidate):
        return scene.collision_free(
            candidate,
            arm,
            target=target,
            allow_target_contact=allow,
            carried=carried,
            penetration_tolerance=0.001,
        )

    goal = solve_ik(scene, arm, position, rotation, qpos, valid=valid_full, attempts=18, random_seed=seed)

    def valid(part):
        full = qpos.copy()
        full[binding.qpos_ids] = part
        return valid_full(full)

    return plan_joint_path(
        qpos[binding.qpos_ids], goal[binding.qpos_ids], binding.lower, binding.upper, valid, seed=seed
    )
