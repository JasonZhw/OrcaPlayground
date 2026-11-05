"""Placement must retain the same bin-local bounds after sharing its logic."""

import mujoco
import numpy as np
import pytest

from examples.euler.medication_sorting.workflow import Scene, placement_satisfied, vertices


def geometry_scene():
    scene = object.__new__(Scene)
    scene.m = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <body name="bin" pos="1 2 0" euler="0 0 90">
            <geom type="box" size=".10 .20 .01"/>
          </body>
          <body name="item" pos="1 2 .07">
            <freejoint/><geom type="box" size=".03 .04 .05"/>
          </body>
        </worldbody></mujoco>""")
    scene.d = mujoco.MjData(scene.m)
    scene.bodies = {name: scene.m.body(name).id for name in ("bin", "item")}
    mujoco.mj_forward(scene.m, scene.d)
    return scene


@pytest.mark.parametrize(
    "position, expected",
    [
        ([1, 2, 0.07], True),
        ([1.15, 2, 0.07], True),  # World X follows the longer bin-local Y axis.
        ([1, 2.15, 0.07], False),
        ([1, 2, 0.20], False),
        ([1, 2, 0.03], False),
    ],
)
def test_placement_uses_bin_frame_and_height(position, expected):
    scene = geometry_scene()
    q = scene.d.qpos.copy()
    q[:3] = position
    scene.set_qpos(q)
    assert placement_satisfied(scene, "item", "bin") is expected


def test_vertices_preserve_world_geometry_and_ignore_noncollision_geoms():
    scene = geometry_scene()
    corners = vertices(scene, "item")
    np.testing.assert_allclose(corners.min(axis=0), [0.97, 1.96, 0.02])
    np.testing.assert_allclose(corners.max(axis=0), [1.03, 2.04, 0.12])
    item_geom = next(iter(scene.body_geoms("item")))
    scene.m.geom_contype[item_geom] = 0
    scene.m.geom_conaffinity[item_geom] = 0
    with pytest.raises(ValueError, match="No collision geometry"):
        vertices(scene, "item")


def test_grasp_target_recomputes_height_after_box_resize():
    from examples.euler.medication_sorting.workflow import grasp_target

    scene = geometry_scene()
    scene.bodies["herbbox"] = scene.bodies["item"]
    before = grasp_target(scene, "right", "herbbox")
    geom = next(iter(scene.body_geoms("herbbox")))
    scene.m.geom_size[geom] *= 0.85
    after = grasp_target(scene, "right", "herbbox")
    np.testing.assert_allclose(before["object_position"], after["object_position"])
    assert before["position"][2] - after["position"][2] == pytest.approx(0.0105)
    np.testing.assert_allclose(after["approach"] - after["position"], [0, 0, 0.065])
