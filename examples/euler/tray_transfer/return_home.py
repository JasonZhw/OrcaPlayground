"""Plan an empty-hand return against the measured, already placed scene."""

import numpy as np

from .placement_result import evaluate_placement
from .plan_tray_transfer import TransportPlanner


def plan_home_return(source_xml, full_qpos, plan):
    planner = TransportPlanner(source_xml, full_qpos, plan)
    home = np.asarray(plan["initial_full_qpos"], dtype=float)[planner.qaddr]
    if not np.isfinite(home).all():
        raise ValueError("Home joints must be finite")
    for name, value in zip(planner.joint_names, home, strict=True):
        joint = planner.model.joint(name).id
        low, high = planner.model.jnt_range[joint]
        if planner.model.jnt_limited[joint] and not low <= value <= high:
            raise ValueError(f"Home exceeds the original joint limits: {name}")
    arm_names = {planner.model.body(i).name for i in planner.arm_bodies}
    distance = float(np.max(np.abs(home - planner.start_q)))
    count = max(2, int(np.ceil(distance / .01)) + 1)
    for fraction in np.linspace(0., 1., count):
        q = (1 - fraction) * planner.start_q + fraction * home
        # The released box stays at its measured pose. Its floor contact is
        # expected; only contacts involving the returning arm obstruct return.
        collisions = [c for c in planner.collisions(q, carried=False)
                      if any(name in arm_names for name in c["bodies"])]
        if collisions:
            raise ValueError(f"Home return intersects the scene: {collisions[0]}")
    duration = max(6., 1.5 * distance / .3)  # smoothstep peak speed <= .3 rad/s
    return {
        "phases": [
            {"name": "return_home", "joints": "arm", "duration_s": duration,
             "q": [planner.start_q.tolist(), home.tolist()]},
            {"name": "home_hold", "joints": "arm", "duration_s": 2.,
             "q": [home.tolist(), home.tolist()]},
        ],
        "home_joint_target": home.tolist(),
        "report": {"planning_only": True, "collision_samples": count,
                   "basis": "Actual post-placement state, including the open fingers and placed box",
                   "home_source": "Layout initial right-arm joint positions",
                   "object_pose_changed": False,
                   "maximum_joint_sample_spacing_rad": distance / (count - 1)},
    }


def evaluate_home_return(trace, target, *, edge_clearance_m=.02, acceptance="stable"):
    """Check the final arm hold and that return did not spoil placement."""
    try:
        target = np.asarray(target, dtype=float)
        groups = []
        for row in trace:
            if not groups or groups[-1] != row["phase"]:
                groups.append(row["phase"])
        if groups != ["return_home", "home_hold"]:
            raise ValueError("Expected a complete return_home then home_hold")
        hold = [r for r in trace if r["phase"] == "home_hold"]
        if hold[-1]["time_s"] - hold[0]["time_s"] < 1. - 1e-9:
            raise ValueError("Home hold must last at least one second")
        window = [r for r in hold if r["time_s"] >= hold[-1]["time_s"] - 1. - 1e-9]
        q = np.asarray([r["right_arm_qpos"] for r in window])
        v = np.asarray([r["right_arm_qvel"] for r in window])
        if (target.shape != (7,) or q.shape != (len(window), 7) or v.shape != q.shape
                or not np.isfinite(np.r_[target, q.ravel(), v.ravel()]).all()):
            raise ValueError("Expected finite seven-joint home evidence")
        placement = evaluate_placement(
            [{**r, "phase": "settle_placed"} for r in trace], edge_clearance_m=edge_clearance_m,
            acceptance=acceptance,
        )
        error, speed = float(np.max(np.abs(q - target))), float(np.max(np.abs(v)))
        checks = {"arm_at_home": error <= .02, "arm_stopped": speed <= .02,
                  "gripper_released": all(not r["gripper"]["hold_active"]
                                          and r["gripper_target_normal_force_n"] < .1 for r in trace),
                  "open_command_maintained": all(
                      r["gripper"]["nominal_command"] == [0., 0., 0., -1., 0., 0., 0., -1.]
                      for r in trace),
                  "no_arm_collision": all(not r["forbidden_contacts"] for r in trace),
                  "object_still_placed": placement["passed"]}
        return {"passed": bool(all(checks.values())), "checks": checks,
                "max_joint_error_rad": error, "max_joint_speed_rad_s": speed,
                "placement_after_return": placement, "training_data": False}
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {"passed": False, "reason": str(error), "training_data": False}
