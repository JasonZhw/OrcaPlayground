"""Locomotion adapter for the mixed actuators in ``g1_pick_usda``."""

from __future__ import annotations

import math

import numpy as np
from g1_locomotion import G1Locomotion
from locomotion_env import LocomotionEnv
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class SimulationGaitClock:
    """Accumulate walking time without depending on renderer wall-clock timing."""

    def __init__(self, gait_period_s: float) -> None:
        if not math.isfinite(gait_period_s) or gait_period_s <= 0.0:
            raise ValueError("gait_period_s must be finite and positive")
        self.gait_period_s = float(gait_period_s)
        self.reset()

    def reset(self) -> None:
        self._last_sim_time_s: float | None = None
        self._walking_elapsed_s = 0.0

    def update(self, sim_time_s: float, *, walking: bool) -> float:
        """Return phase in [0, 1), advancing only with simulated walking time."""
        if not math.isfinite(sim_time_s) or sim_time_s < 0.0:
            raise ValueError("sim_time_s must be finite and non-negative")
        if self._last_sim_time_s is None or sim_time_s < self._last_sim_time_s:
            if self._last_sim_time_s is not None:
                self._walking_elapsed_s = 0.0
            self._last_sim_time_s = float(sim_time_s)
            return 0.0

        elapsed_s = float(sim_time_s - self._last_sim_time_s)
        self._last_sim_time_s = float(sim_time_s)
        if walking:
            self._walking_elapsed_s += elapsed_s
        return (self._walking_elapsed_s % self.gait_period_s) / self.gait_period_s


def count_actor_actuators(actuator_dict: dict[str, dict], actor_name: str) -> int:
    """Count controls owned by one Actor without including other Layout robots."""
    prefix = f"{actor_name}_"
    return sum(
        1
        for actuator_name, actuator_info in actuator_dict.items()
        if actuator_name.startswith(prefix)
        or str(actuator_info.get("JointName", "")).startswith(prefix)
    )


class SimulationClockG1Locomotion(G1Locomotion):
    """Run the frozen policy with a gait phase tied to simulation time."""

    def __init__(self, *args, **kwargs) -> None:
        self._gait_clock: SimulationGaitClock | None = None
        self._simulation_phase = np.zeros((1, 1), dtype=np.float64)
        super().__init__(*args, **kwargs)
        self._gait_clock = SimulationGaitClock(self.gait_period)

    def reset(self) -> None:
        super().reset()
        self._simulation_phase = np.zeros((1, 1), dtype=np.float64)
        if self._gait_clock is not None:
            self._gait_clock.reset()

    def compute_q_target(self, env: OrcaGymEulerEnv) -> np.ndarray:
        if self._gait_clock is None:
            raise RuntimeError("simulation gait clock is not initialized")
        phase = self._gait_clock.update(
            float(env.data.time),
            walking=bool(self.stand_command[0, 0]),
        )
        self._simulation_phase[0, 0] = phase
        return super().compute_q_target(env)

    def _get_phase_time(self) -> np.ndarray:
        return self._simulation_phase.copy()


class G1PickLocomotionEnv(LocomotionEnv):
    """Run the 29-DoF locomotion policy on the 45-actuator picking asset.

    The picking asset exposes 29 body actuators and 16 gripper actuators. Its
    legs and waist use built-in position controllers, while its arms use motor
    actuators. The policy still produces the same 29 joint targets as the base
    G1 locomotion model. For each position actuator, this adapter converts the
    policy PD torque into an equivalent position command using the actuator's
    own gains. Motor actuators receive torque directly and grippers stay at zero.
    """

    NUM_BODY_DOFS = 29

    def initialize_simulation(self):
        """Load the scene without the legacy actuator/sensor suffix scanner."""
        OrcaGymEulerEnv.initialize_simulation(self)
        self.locomotion = SimulationClockG1Locomotion(agent_name=self.agent_name)
        self._torque_clip_count = 0
        self._torque_total_count = 0
        self._policy_action_finite = True
        self._q_target = np.zeros(self.NUM_BODY_DOFS, dtype=np.float64)
        self._build_body_actuator_mapping()

    def _build_body_actuator_mapping(self) -> None:
        actuator_dict = self.model.get_actuator_dict()
        actuators_by_joint: dict[str, list[dict]] = {}
        for actuator_info in actuator_dict.values():
            joint_name = actuator_info["JointName"]
            actuators_by_joint.setdefault(joint_name, []).append(actuator_info)

        actuator_ids: list[int] = []
        position_controlled: list[bool] = []
        position_lower: list[float] = []
        position_upper: list[float] = []
        position_kp: list[float] = []
        position_kd: list[float] = []
        for joint_name in self.locomotion.joint_names:
            matches = actuators_by_joint.get(joint_name, [])
            if len(matches) != 1:
                raise ValueError(f"expected exactly one actuator for {joint_name!r}, found {len(matches)}")
            actuator_info = matches[0]
            actuator_id = int(actuator_info["ActuatorId"])
            bias_type = int(actuator_info["BiasType"])
            if bias_type not in (0, 1):
                raise ValueError(f"unsupported BiasType={bias_type} for joint {joint_name!r}")
            ctrl_range = np.asarray(actuator_info["CtrlRange"], dtype=np.float64)
            gain_prm = np.asarray(actuator_info["GainPrm"], dtype=np.float64)
            bias_prm = np.asarray(actuator_info["BiasPrm"], dtype=np.float64)
            actuator_kp = float(gain_prm[0]) if bias_type == 1 else 0.0
            actuator_kd = float(-bias_prm[2]) if bias_type == 1 else 0.0
            if bias_type == 1 and (actuator_kp <= 0.0 or actuator_kd < 0.0):
                raise ValueError(f"invalid position gains for joint {joint_name!r}: kp={actuator_kp}, kd={actuator_kd}")
            actuator_ids.append(actuator_id)
            position_controlled.append(bias_type == 1)
            position_lower.append(float(ctrl_range[0]))
            position_upper.append(float(ctrl_range[1]))
            position_kp.append(actuator_kp)
            position_kd.append(actuator_kd)

        if len(set(actuator_ids)) != self.NUM_BODY_DOFS:
            raise ValueError("body actuator IDs must be unique")
        if not all(0 <= actuator_id < self.model.nu for actuator_id in actuator_ids):
            raise ValueError("body actuator ID is outside the model control range")

        self._body_actuator_ids = np.asarray(actuator_ids, dtype=np.int64)
        self._position_controlled = np.asarray(position_controlled, dtype=bool)
        self._position_lower = np.asarray(position_lower, dtype=np.float64)
        self._position_upper = np.asarray(position_upper, dtype=np.float64)
        self._position_kp = np.asarray(position_kp, dtype=np.float64)
        self._position_kd = np.asarray(position_kd, dtype=np.float64)

        print("[机器人] 29 个本体关节执行器匹配完成。")

    def step(self, action: np.ndarray) -> tuple:
        """Step with a 29-D policy target while the MuJoCo model has 45 controls."""
        q_target = np.asarray(action, dtype=np.float64).reshape(self.NUM_BODY_DOFS)
        for _ in range(self.frame_skip):
            ctrl = self._pd_controller(q_target)
            self.do_simulation(ctrl, 1)
        obs = self._get_obs()
        reward = self._compute_reward(obs, q_target)
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info: dict[str, float] = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def _pd_controller(self, target: np.ndarray) -> np.ndarray:
        """Route policy PD torque through the asset's mixed actuator types."""
        dof_pos, dof_vel = self.locomotion.read_joint_state(self)
        tau = self.locomotion.compute_tau(target, dof_pos, dof_vel)
        if not np.all(np.isfinite(tau)):
            self._policy_action_finite = False

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        position_mask = self._position_controlled
        motor_mask = ~position_mask

        # A MuJoCo position actuator applies approximately
        #   tau = kp_asset * (ctrl - q) - kd_asset * qd
        # Solve for ctrl so that it reproduces the already clipped policy tau.
        position_targets = (
            dof_pos[position_mask]
            + (tau[position_mask] + self._position_kd[position_mask] * dof_vel[position_mask])
            / self._position_kp[position_mask]
        )
        position_targets = np.clip(
            position_targets,
            self._position_lower[position_mask],
            self._position_upper[position_mask],
        )
        ctrl[self._body_actuator_ids[position_mask]] = position_targets
        ctrl[self._body_actuator_ids[motor_mask]] = tau[motor_mask]

        effort_limits = self.locomotion.motor_effort_limit
        self._torque_clip_count += int(np.sum(np.abs(tau) >= effort_limits - 0.1))
        self._torque_total_count += len(tau)
        return ctrl

    def _draw_debug_viz(self, step: int) -> None:
        """Keep the first compatibility test focused on locomotion only."""
        return None
