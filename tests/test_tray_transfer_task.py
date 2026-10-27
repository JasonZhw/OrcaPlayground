"""Frame and placement regression tests without OrcaLab or GPU access."""

import copy
import unittest

import numpy as np

from examples.euler.tray_transfer.task import build_report, resolve_name


def box(name, body, half_size, pos):
    return {
        "geom_name": name, "body_name": body, "geom_type": 6,
        "geom_contype": 1, "geom_conaffinity": 1,
        "geom_size": half_size, "geom_pos": pos, "geom_quat": [1, 0, 0, 0],
    }


def scene_fixture():
    names = ["robot_pelvis", "[11]_tray", "gray", "[22]_blue"]
    poses = [
        {"body_name": name, "pos": pos, "quat": quat}
        for name, pos, quat in zip(names, [[4, 1, 0], [1, 2, .5], [2, 2, .5], [2, 3, .5]],
                                  [[1, 0, 0, 0], [2**-.5, 0, 0, 2**-.5],
                                   [1, 0, 0, 0], [1, 0, 0, 0]])
    ]
    responses = {
        "QueryAllBodies": {"body_info": [
            {"name": n, "id": i + 1, "parent_id": 0, "mass": 1} for i, n in enumerate(names)
        ]},
        "QueryAllGeoms": {"geom_data_list": [
            box("floor", names[1], [.3, .2, .01], [0, 0, .01]),
            box("gray_shape", "gray", [.02, .03, .04], [.01, 0, .04]),
            box("blue_shape", names[3], [.03, .04, .05], [0, 0, .05]),
        ]},
        "QueryBodyPosMatQuat": {"body_pos_mat_quat_list": poses},
        "QueryAllJoints": {"joint_info": [
            {"name": "robot_floating_base_joint", "type": 0, "body_id": 1}
        ]},
        "QueryAllActuators": {"ActuatorDataList": []},
        "QueryAllSites": {"site_info": []},
        "QueryAllEqualityConstraints": {"equality_constraints": []},
        "GetCameraNames": {"camera_names": []},
        "GetBodyManipulationAnchored": {"body_name": "", "anchor_type": "NONE"},
    }
    config = {
        "task_id": "two_parts_to_tray", "robot": {"base_body": "robot_pelvis", "tcp_sites": []},
        "tray": {"body_suffix": "tray", "floor_size_xyz_m": [.6, .4, .02],
                 "inner_xy_min": [-.28, -.18], "inner_xy_max": [.28, .18],
                 "edge_clearance_m": .02, "release_clearance_m": .005},
        "objects": [
            {"id": "gray", "body_suffix": "gray", "placement_center_xy": [-.12, 0]},
            {"id": "blue", "body_suffix": "blue", "placement_center_xy": [.12, 0]},
        ],
    }
    return {"responses": responses}, config


class TestTrayTransferTask(unittest.TestCase):
    def test_ambiguous_asset_names_do_not_choose_first(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_name(["[1]_blue", "[2]_blue"], "blue")

    def test_floor_is_found_after_runtime_geometry_ids_change(self):
        snapshot, config = scene_fixture()
        snapshot["responses"]["QueryAllGeoms"]["geom_data_list"][0]["geom_name"] = "runtime_NEW_ID"
        # A 180-degree floor rotation has exactly the same box footprint.
        snapshot["responses"]["QueryAllGeoms"]["geom_data_list"][0]["geom_quat"] = [0, 0, 0, 1]
        self.assertTrue(build_report(snapshot, config)["geometry_passed"])

    def test_rotated_tray_places_geometry_center_and_floor_correctly(self):
        snapshot, config = scene_fixture()
        report = build_report(snapshot, config)
        gray = report["objects"][0]
        self.assertTrue(report["geometry_passed"])
        # Tray X maps to world Y; body origin has a 1 cm offset from shape center.
        np.testing.assert_allclose(gray["placement"]["body_origin_world"], [.99, 1.88, .525])
        np.testing.assert_allclose(gray["placement"]["body_origin_base"], [-3.01, .88, .525])
        self.assertAlmostEqual(gray["placement"]["bounds_min"][2], .025)

    def test_two_slots_overlapping_fails_even_if_each_fits_tray(self):
        snapshot, config = scene_fixture()
        config["objects"][1]["placement_center_xy"] = [-.12, 0]
        report = build_report(snapshot, config)
        self.assertFalse(report["geometry_passed"])
        self.assertTrue(any("overlap" in err for err in report["geometry_errors"]))

    def test_rail_in_configured_opening_is_detected(self):
        snapshot, config = scene_fixture()
        snapshot["responses"]["QueryAllGeoms"]["geom_data_list"].append(
            box("new_rail", "[11]_tray", [.02, .1, .08], [-.12, 0, .08])
        )
        report = build_report(snapshot, config)
        self.assertFalse(report["geometry_passed"])
        self.assertTrue(any("new_rail" in err for err in report["geometry_errors"]))

    def test_free_base_is_not_reported_as_fixed(self):
        snapshot, config = scene_fixture()
        report = build_report(snapshot, config)
        self.assertTrue(report["robot"]["base_has_free_joint"])
        self.assertIn("base_stabilization", report["unverified"])

    def test_snapshot_and_configuration_are_not_modified(self):
        snapshot, config = scene_fixture()
        before = copy.deepcopy((snapshot, config))
        build_report(snapshot, config)
        self.assertEqual((snapshot, config), before)


if __name__ == "__main__":
    unittest.main()
