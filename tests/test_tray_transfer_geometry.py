"""Collision-volume geometry tests; no simulator or CUDA required."""

import unittest

import numpy as np

from examples.euler.tray_transfer.geometry import (
    body_bounds,
    candidate_placement,
    primitive_bounds,
    quat_matrix,
)


def geom(kind, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), collision=True):
    return {
        "geom_type": kind,
        "geom_size": list(size),
        "geom_pos": list(pos),
        "geom_quat": list(quat),
        "geom_contype": int(collision),
        "geom_conaffinity": 0,
    }


class TestPrimitiveBounds(unittest.TestCase):
    def test_wxyz_quaternion_rotates_x_to_y_and_normalizes(self):
        rotation = quat_matrix([2, 0, 0, 2])
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_box_composes_geom_and_body_frames(self):
        # Two quarter turns make a half turn, while body translation is applied last.
        quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        lower, upper = primitive_bounds(
            geom(6, [1, 2, 3], pos=[2, 0, 0], quat=[1, 0, 0, 1]),
            frame_rotation=quarter_turn,
            frame_translation=[10, 20, 30],
        )
        np.testing.assert_allclose(lower, [9, 20, 27], atol=1e-12)
        np.testing.assert_allclose(upper, [11, 24, 33], atol=1e-12)

    def test_cylinder_uses_radius_and_half_height_not_third_size(self):
        # Cylinder axis lies halfway between +x and +z.
        q = [np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0]
        lower, upper = primitive_bounds(geom(5, [2, 3, 999], quat=q))
        expected = [3.5355339059327378, 2, 3.5355339059327378]
        np.testing.assert_allclose(upper, expected)
        np.testing.assert_allclose(lower, -np.array(expected))

    def test_capsule_half_length_excludes_hemispheres(self):
        lower, upper = primitive_bounds(geom(3, [2, 3, 999], quat=[1, 0, 1, 0]))
        np.testing.assert_allclose(lower, [-5, -2, -2], atol=1e-12)
        np.testing.assert_allclose(upper, [5, 2, 2], atol=1e-12)

    def test_rotated_ellipsoid_has_exact_support_not_rotated_box(self):
        lower, upper = primitive_bounds(
            geom(4, [1, 3, 2], quat=[np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
        )
        np.testing.assert_allclose(upper, [2.23606797749979, 2.23606797749979, 2])
        np.testing.assert_allclose(lower, -upper)

    def test_sphere_only_uses_radius(self):
        lower, upper = primitive_bounds(geom(2, [2, 999, 999], pos=[1, 2, 3]))
        np.testing.assert_allclose(lower, [-1, 0, 1])
        np.testing.assert_allclose(upper, [3, 4, 5])

    def test_invalid_or_unsupported_geometry_is_rejected(self):
        invalid = [
            geom(7, [1, 1, 1]),
            geom(0, [1, 1, 1]),
            geom(6, [-1, 1, 1]),
            geom(6, [1, 1, 1], quat=[0, 0, 0, 0]),
            geom(6, [1, 1, 1], pos=[np.nan, 0, 0]),
            geom(6, [1, np.inf, 1]),
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                primitive_bounds(item)
        with self.assertRaises(ValueError):
            primitive_bounds(geom(6, [1, 1, 1]), frame_rotation=np.diag([2, 1, 1]))


class TestBodyPlacement(unittest.TestCase):
    def test_body_unions_colliders_and_ignores_visual_mesh(self):
        colliders = [
            geom(6, [1, 1, 1], pos=[-2, 0, 0]),
            geom(2, [1, 0, 0], pos=[2, 0, 0]),
            geom(7, [999, 999, 999], collision=False),
        ]
        # conaffinity alone also marks a collision geom.
        colliders[1]["geom_contype"] = 0
        colliders[1]["geom_conaffinity"] = 2
        lower, upper = body_bounds(colliders, translation=[0, 0, 10])
        np.testing.assert_allclose(lower, [-3, -1, 9])
        np.testing.assert_allclose(upper, [3, 1, 11])

    def test_no_colliders_or_collision_mesh_is_an_error(self):
        for colliders in ([], [geom(7, [1, 1, 1], collision=False)], [geom(7, [1, 1, 1])]):
            with self.subTest(colliders=colliders), self.assertRaises(ValueError):
                body_bounds(colliders)

    def test_placement_corrects_rotated_body_origin_offset(self):
        quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        placement = candidate_placement(
            [geom(6, [0.1, 0.2, 0.3], pos=[1, 2, 3])],
            quarter_turn,
            center_xy=[4, 5],
            tray_floor_z=0.5,
            clearance=0.01,
        )
        np.testing.assert_allclose(placement["body_origin_tray"], [6, 4, -2.19])
        np.testing.assert_allclose(placement["bounds_min"], [3.8, 4.9, 0.51])
        np.testing.assert_allclose(placement["bounds_max"], [4.2, 5.1, 1.11])
        np.testing.assert_allclose(placement["size"], [0.4, 0.2, 0.6])

    def test_placement_rejects_negative_clearance(self):
        with self.assertRaises(ValueError):
            candidate_placement([geom(6, [1, 1, 1])], np.eye(3), [0, 0], 0, -0.01)


if __name__ == "__main__":
    unittest.main()
