import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.euler.medication_sorting.workflow import placement_inside, sample_layout
from examples.euler.medication_sorting import workflow
from examples.euler.medication_sorting.planning import PlanningError


class TargetTests(unittest.TestCase):
    def selection_scene(self):
        positions = {'calcium': np.array([2., 0, 0]), 'melatonin': np.array([1., 0, 0])}
        scene = SimpleNamespace(
            arms={'left': SimpleNamespace(qpos_ids=np.arange(7))},
            end_pose=lambda arm: (np.zeros(3), np.eye(3)),
            body_pose=lambda key: (positions[key], np.eye(3)),
        )
        scene.set_qpos = lambda q: setattr(scene, 'qpos', q.copy())
        return scene, positions

    def choose(self, scene):
        self.assertTrue(hasattr(workflow, 'choose_next_object'), 'Missing layout-dependent target selection')
        return workflow.choose_next_object(scene, 'left', ['calcium', 'melatonin'], np.zeros(7), seed=7)

    def target(self, scene, arm, key):
        return {'approach': np.zeros(3), 'position': np.ones(3), 'rotation': np.eye(3)}

    def test_swapping_positions_changes_first_target_without_changing_arm_assignment(self):
        scene, positions = self.selection_scene()
        def plan(scene, arm, position, rotation, qpos, **kwargs):
            return np.stack([qpos, qpos + 0.1])
        with patch.object(workflow, 'grasp_target', self.target), patch.object(
            workflow, 'plan_motion', plan, create=True
        ):
            self.assertEqual(self.choose(scene)[0], 'melatonin')
            positions['calcium'], positions['melatonin'] = positions['melatonin'], positions['calcium']
            self.assertEqual(self.choose(scene)[0], 'calcium')

    def test_blocked_near_target_does_not_prevent_selecting_other_bottle(self):
        for blocked_phase in (False, True):
            scene, _ = self.selection_scene()
            def plan(scene, arm, position, rotation, qpos, *, target, allow=False, **kwargs):
                scene.set_qpos(np.ones(7) * 9)  # Planning mutates only its private scene.
                if target == 'melatonin' and allow == blocked_phase:
                    raise PlanningError('blocked by object')
                return np.stack([qpos, qpos + 0.1])
            with self.subTest(blocked_phase=blocked_phase), patch.object(workflow, 'grasp_target', self.target), patch.object(
                workflow, 'plan_motion', plan, create=True
            ):
                key, report = self.choose(scene)
                self.assertEqual(key, 'calcium')
                self.assertIn('melatonin', report['rejected'])
                np.testing.assert_array_equal(scene.qpos, np.zeros(7))

    def test_all_blocked_reports_both_candidates_and_restores_scene(self):
        scene, _ = self.selection_scene()
        with patch.object(workflow, 'grasp_target', self.target), patch.object(
            workflow, 'plan_motion', side_effect=PlanningError('blocked'), create=True
        ):
            self.assertTrue(hasattr(workflow, 'choose_next_object'))
            with self.assertRaisesRegex(PlanningError, 'calcium.*melatonin|melatonin.*calcium'):
                self.choose(scene)
            np.testing.assert_array_equal(scene.qpos, np.zeros(7))

    def test_selector_rejects_wrong_arm_objects(self):
        scene, _ = self.selection_scene()
        self.assertTrue(hasattr(workflow, 'choose_next_object'))
        with self.assertRaises(ValueError):
            workflow.choose_next_object(scene, 'left', ['herbbox'], np.zeros(7))

    def test_randomized_layout_reproducible_and_bounded(self):
        base = {"x": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]}}
        a = sample_layout(base, 17, 0.01, 5)
        b = sample_layout(base, 17, 0.01, 5)
        self.assertEqual(a, b)
        self.assertLessEqual(np.max(np.abs(np.array(a["x"]["position"])[:2] - [1, 2])), 0.01)
        self.assertEqual(a["x"]["position"][2], 3)
        self.assertNotEqual(a, sample_layout(base, 18, 0.01, 5))

    def test_accepts_tilted_item_but_not_hovering_or_outside(self):
        self.assertTrue(placement_inside([0, 0, 0.04], [0.06, 0.10], 0.01))
        self.assertFalse(placement_inside([0.12, 0, 0.04], [0.06, 0.10], 0.01))
        self.assertFalse(placement_inside([0, 0, 0.2], [0.06, 0.10], 0.15))


if __name__ == "__main__":
    unittest.main()
