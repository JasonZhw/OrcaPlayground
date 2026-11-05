"""Pure NumPy planner tests; no simulator or server is required."""

import unittest
from importlib import import_module

import numpy as np


class TestMedicationPlanner(unittest.TestCase):
    def planner(self):
        try:
            return import_module("examples.euler.medication_sorting.planning")
        except ModuleNotFoundError as exc:
            self.fail(f"Planner implementation is missing: {exc}")

    def test_rrt_detours_with_valid_edges_and_repeatable_seed(self):
        module = self.planner()
        start, goal = np.array([-0.8, 0.0]), np.array([0.8, 0.0])

        def valid(q):
            return not (-0.22 <= q[0] <= 0.22 and -0.55 <= q[1] <= 0.55)

        args = (start, goal, [-1.0, -1.0], [1.0, 1.0], valid)
        path = module.plan_joint_path(*args, seed=13)
        np.testing.assert_array_equal(path, module.plan_joint_path(*args, seed=13))
        np.testing.assert_array_equal(path[0], start)
        np.testing.assert_array_equal(path[-1], goal)
        self.assertGreater(len(path), 2)
        for a, b in zip(path[:-1], path[1:]):
            for q in np.linspace(a, b, 1001):
                self.assertTrue(valid(q), "Returned edge cuts through the obstacle")
                self.assertTrue(np.all(q >= -1.0) and np.all(q <= 1.0))

    def test_impossible_barrier_exhausts_finite_budget(self):
        module = self.planner()
        calls = 0

        def valid(q):
            nonlocal calls
            calls += 1
            return abs(q[0]) > 0.12

        with self.assertRaises(module.PlanningError):
            module.plan_joint_path([-0.8, 0.0], [0.8, 0.0], [-1.0, -1.0], [1.0, 1.0], valid, max_iterations=80)
        self.assertLess(calls, 10000, "Impossible queries must have a finite search budget")

    def test_rejects_colliding_endpoints(self):
        module = self.planner()
        for start, goal in [([0.0], [0.8]), ([-0.8], [0.0])]:
            with self.subTest(start=start, goal=goal), self.assertRaises(module.PlanningError):
                module.plan_joint_path(start, goal, [-1.0], [1.0], lambda q: abs(q[0]) > 0.1)

    def test_direct_connection_checks_interior_of_edge(self):
        module = self.planner()
        with self.assertRaises(module.PlanningError):
            module.plan_joint_path([-0.8], [0.8], [-1.0], [1.0], lambda q: abs(q[0]) > 0.1, max_iterations=30)

    def test_rejects_invalid_limits_and_configurations(self):
        module = self.planner()
        cases = [
            ([np.nan], [0.5], [-1.0], [1.0]),
            ([0.0], [np.inf], [-1.0], [1.0]),
            ([0.0], [0.5], [-np.inf], [1.0]),
            ([0.0], [0.5], [1.0], [-1.0]),
            ([-2.0], [0.5], [-1.0], [1.0]),
            ([0.0], [2.0], [-1.0], [1.0]),
            ([0.0, 0.0], [0.5], [-1.0], [1.0]),
        ]
        for start, goal, lower, upper in cases:
            with self.subTest(start=start, goal=goal), self.assertRaises(module.PlanningError):
                module.plan_joint_path(start, goal, lower, upper, lambda q: True)

    def test_rejects_invalid_search_settings(self):
        module = self.planner()
        for options in [
            {"resolution": 0},
            {"step_size": -1},
            {"max_iterations": -1},
            {"step_size": np.inf},
            {"resolution": np.nan},
            {"max_iterations": 1.2},
        ]:
            with self.subTest(options=options), self.assertRaises(module.PlanningError):
                module.plan_joint_path([0.0], [0.5], [-1.0], [1.0], lambda q: True, **options)

    def test_stationary_path_and_fixed_joint_limits(self):
        module = self.planner()
        path = module.plan_joint_path([0.2, 0.0], [0.2, 0.0], [-1.0, 0.0], [1.0, 0.0], lambda q: True)
        self.assertEqual(path.shape, (1, 2))
        np.testing.assert_array_equal(module.time_samples(path), path)

    def test_samples_preserve_waypoints_and_bound_joint_velocity(self):
        module = self.planner()
        path = np.array([[0.0, 0.2], [1.0, -0.3], [1.0, -0.3], [0.9, 0.2]])
        samples = module.time_samples(path, dt=0.02, max_velocity=0.5)
        np.testing.assert_array_equal(samples[0], path[0])
        np.testing.assert_array_equal(samples[-1], path[-1])
        self.assertTrue(any(np.array_equal(q, path[1]) for q in samples))
        self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0))) / 0.02, 0.5 + 1e-12)
        self.assertTrue(np.all(np.abs(samples[1] - samples[0]) / 0.02 < 0.02))
        self.assertTrue(np.all(np.abs(samples[-1] - samples[-2]) / 0.02 < 0.02))

    def test_time_sampling_rejects_invalid_input(self):
        module = self.planner()
        for path, options in [
            ([], {}),
            ([[np.nan]], {}),
            ([[0.0], [1.0]], {"dt": 0}),
            ([[0.0], [1.0]], {"max_velocity": -1}),
            ([[0.0], [1.0]], {"max_velocity": np.inf}),
            ([[0.0], [1.0]], {"max_acceleration": 0}),
            ([[0.0], [1.0]], {"max_acceleration": np.inf}),
        ]:
            with self.subTest(path=path, options=options), self.assertRaises(module.PlanningError):
                module.time_samples(path, **options)


if __name__ == "__main__":
    unittest.main()


def test_static_gravity_compensation_must_fit_servo_control_range():
    import mujoco
    from types import SimpleNamespace
    from examples.euler.medication_sorting import planning

    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody><body><joint name="arm" axis="0 1 0" range="-180 180"/>
        <geom type="sphere" size=".03" pos=".2 0 0" mass="1"/>
      </body></worldbody>
      <actuator><position joint="arm" kp="10" ctrllimited="true" ctrlrange="-1 1"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    scene = SimpleNamespace(m=model, d=data, arms={'left': SimpleNamespace(qpos_ids=np.array([0]))})
    data.qpos[0] = -.95  # Legal joint/command, but gravity feed-forward clips.
    mujoco.mj_forward(model, data)
    assert not planning.servo_bias_feasible(scene, 'left')
    data.qpos[0] = 0
    mujoco.mj_forward(model, data)
    assert planning.servo_bias_feasible(scene, 'left')
    model.actuator_ctrllimited[0] = False
    data.qpos[0] = -.95
    mujoco.mj_forward(model, data)
    assert planning.servo_bias_feasible(scene, 'left')


def test_motion_planner_rejects_collision_free_but_saturated_ik(monkeypatch):
    from types import SimpleNamespace
    from examples.euler.medication_sorting import planning

    scene = SimpleNamespace(
        arms={'left': SimpleNamespace(qpos_ids=np.arange(7))},
        collision_free=lambda *a, **kw: True,
    )
    monkeypatch.setattr(planning, 'servo_bias_feasible', lambda *a: False, raising=False)
    def solve(scene, arm, position, rotation, qpos, *, valid, **kwargs):
        assert valid(qpos) is False, 'IK acceptance ignored servo saturation'
        raise planning.IKError('compensation clips')
    monkeypatch.setattr(planning, 'solve_ik', solve)
    import pytest
    with pytest.raises(planning.IKError, match='compensation clips'):
        planning.plan_motion(scene, 'left', np.zeros(3), np.eye(3), np.zeros(7))
