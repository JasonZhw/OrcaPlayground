"""Independent scene checks; cached-model integration tests need no live server."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from examples.euler.medication_sorting.planning import IKError, solve_ik
from examples.euler.medication_sorting.workflow import (
    BODY_SUFFIXES,
    Scene,
    SceneBindingError,
    bind_scene,
    prepare_model,
)

CACHED_XML = Path.home() / ".orcagym/tmp/103966F9_E586_4982_8805_3401293E3780.xml"


def test_strict_dynamic_name_binding():
    names = [f"generated_123_{suffix}" for suffix in BODY_SUFFIXES.values()]
    assert bind_scene(names)["calcium"] == "generated_123_Medical_32"
    with pytest.raises(SceneBindingError, match="Medical_32"):
        bind_scene(names[1:])
    with pytest.raises(SceneBindingError, match="Medical_32"):
        bind_scene(names + ["another_Medical_32"])
    with pytest.raises(SceneBindingError, match="Medical_32"):
        bind_scene([name.replace("Medical_32", "Medical_320") for name in names])


def small_xml(mesh_name="piece.stl"):
    return (
        '<mujoco><option impratio="1"/><compiler/><asset><mesh name="x" file="'
        + mesh_name
        + '"/></asset><worldbody>'
        + "".join(
            f'<body name="robot_gripper_{side}_{finger}_link4"><geom name="{side}{finger}" '
            'type="sphere" size=".01" friction="2 .1 .1" condim="3"/></body>'
            for side in ("l", "r")
            for finger in ("inner", "outer")
        )
        + '<body name="other"><freejoint name="free"/><geom type="sphere" size=".1" condim="3"/></body>'
        + "</worldbody></mujoco>"
    )


def test_prepare_only_expected_contact_changes_and_absolute_asset_path(tmp_path):
    source = tmp_path / "scene.xml"
    source.write_text(small_xml())
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/piece.stl").write_bytes(b"mesh placeholder")
    original = source.read_bytes()
    dest = prepare_model(source, tmp_path / "copy/prepared.xml")
    assert source.read_bytes() == original
    before, after = ET.fromstring(original), ET.parse(dest).getroot()
    assert after.find("option").get("impratio") == "2"
    assert after.find("./asset/mesh").get("file") == str(tmp_path / "assets/piece.stl")
    before_nodes, after_nodes = list(before.iter()), list(after.iter())
    assert len(before_nodes) == len(after_nodes)
    changes = []
    for a, b in zip(before_nodes, after_nodes):
        assert a.tag == b.tag
        for key in a.attrib.keys() | b.attrib.keys():
            if a.get(key) != b.get(key):
                changes.append((a.tag, key))
    assert sorted(changes) == sorted([("option", "impratio"), ("mesh", "file")] + [("geom", "condim")] * 4)
    with pytest.raises(ValueError, match="unchanged"):
        prepare_model(source, source)


def test_prepare_rejects_missing_assets(tmp_path):
    source = tmp_path / "scene.xml"
    source.write_text(small_xml())
    with pytest.raises(FileNotFoundError, match="Missing mesh"):
        prepare_model(source, tmp_path / "prepared.xml")


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    if not CACHED_XML.exists():
        pytest.skip("Cached live scene is not available")
    return Scene(prepare_model(CACHED_XML, tmp_path_factory.mktemp("scene") / "prepared.xml"))


def test_actual_scene_preserves_dimensions_and_friction(scene):
    original = mujoco.MjModel.from_xml_path(str(CACHED_XML))
    assert (scene.m.nq, scene.m.nv, scene.m.nu, scene.m.neq) == (original.nq, original.nv, original.nu, original.neq)
    np.testing.assert_array_equal(scene.m.geom_friction, original.geom_friction)
    tips = set().union(*(binding.finger_geom_ids for binding in scene.arms.values()))
    assert len(tips) == 4
    assert set(np.flatnonzero(scene.m.geom_condim != original.geom_condim)) == tips
    assert all(scene.m.geom_condim[g] == 6 for g in tips)
    assert scene.m.opt.impratio == 2


@pytest.mark.parametrize("arm", ["left", "right"])
def test_actual_ik_roundtrip_and_untouched_other_joints(scene, arm):
    qpos = scene.m.qpos0.copy()
    ids = scene.arms[arm].qpos_ids
    target_qpos = qpos.copy()
    target_qpos[ids] -= np.array([0.06, 0.04, -0.03, 0.04, 0.02, -0.02, 0.03])
    scene.set_qpos(target_qpos)
    position, rotation = scene.end_pose(arm)
    result = solve_ik(scene, arm, position, rotation, qpos, attempts=3)
    scene.set_qpos(result)
    actual_p, actual_r = scene.end_pose(arm)
    np.testing.assert_allclose(actual_p, position, atol=0.003)
    np.testing.assert_allclose(actual_r, rotation, atol=0.035)
    other = np.setdiff1d(np.arange(scene.m.nq), ids)
    np.testing.assert_array_equal(result[other], qpos[other])


def test_ik_rejects_unreachable_or_invalid_target(scene):
    qpos = scene.m.qpos0.copy()
    with pytest.raises(IKError, match="1 attempts"):
        solve_ik(scene, "left", [100.0, 100.0, 100.0], np.eye(3), qpos, attempts=1, max_nfev=15)
    with pytest.raises(ValueError, match="proper rotation"):
        solve_ik(scene, "left", [0.0, 0.0, 0.0], np.zeros((3, 3)), qpos)


def test_collision_query_checks_target_contact_and_preserves_input(scene):
    qpos = scene.m.qpos0.copy()
    scene.set_qpos(qpos)
    tip = next(iter(scene.arms["right"].finger_geom_ids))
    target = next(iter(scene.body_geoms("pharyngitis")))
    address = scene.free_joint_address("pharyngitis")
    qpos[address : address + 3] += scene.d.geom_xpos[tip] - scene.d.geom_xpos[target]
    unchanged = qpos.copy()
    assert not scene.collision_free(qpos, "right", target="pharyngitis")
    strict = list(scene.last_collisions)
    scene.collision_free(qpos, "right", target="pharyngitis", allow_target_contact=True)
    allowed = list(scene.last_collisions)
    tip_names = {scene.m.geom(g).name for g in scene.arms["right"].finger_geom_ids}
    target_names = {scene.m.geom(g).name for g in scene.body_geoms("pharyngitis")}
    assert any((a in tip_names and b in target_names) or (b in tip_names and a in target_names) for a, b, _ in strict)
    assert not any(
        (a in tip_names and b in target_names) or (b in tip_names and a in target_names) for a, b, _ in allowed
    )
    np.testing.assert_array_equal(qpos, unchanged)


def test_carried_transform_is_planning_only_and_has_no_physical_weld(scene):
    qpos = scene.m.qpos0.copy()
    carried = scene.attach_object("left", "calcium", qpos)
    qpos[scene.arms["left"].qpos_ids[0]] += 0.1
    unchanged, neq = qpos.copy(), scene.m.neq
    scene.collision_free(qpos, "left", target="calcium", allow_target_contact=True, carried=carried)
    position, rotation = scene.end_pose("left")
    actual_p, actual_r = scene.body_pose("calcium")
    np.testing.assert_allclose(actual_p, position + rotation @ carried.local_position, atol=1e-9)
    np.testing.assert_allclose(actual_r, rotation @ carried.local_rotation, atol=1e-9)
    np.testing.assert_array_equal(qpos, unchanged)
    assert scene.m.neq == neq


def test_carried_object_collision_with_table_is_not_exempted(scene):
    qpos = scene.m.qpos0.copy()
    qpos[scene.free_joint_address("calcium") + 2] -= 0.03
    carried = scene.attach_object("left", "calcium", qpos)
    assert not scene.collision_free(qpos, "left", target="calcium", allow_target_contact=True, carried=carried)
    assert any("Desk" in a or "Desk" in b for a, b, _ in scene.last_collisions)


def test_prepared_model_repairs_only_connected_four_bar_internal_contacts(tmp_path):
    root = ET.fromstring(small_xml())
    world = root.find("worldbody")
    for side in ["l", "r"]:
        for finger in ["inner", "outer"]:
            for link in [1, 2, 3]:
                ET.SubElement(world, "body", name=f"robot_gripper_{side}_{finger}_link{link}")
    eq = ET.SubElement(root, "equality")
    ET.SubElement(
        eq, "connect", body1="robot_gripper_r_inner_link4", body2="robot_gripper_r_inner_link2", anchor="0 0 0"
    )
    (tmp_path / "piece.stl").write_bytes(b"placeholder")
    source = tmp_path / "in.xml"
    ET.ElementTree(root).write(source)
    out = prepare_model(source, tmp_path / "out.xml")
    prepared = ET.parse(out).getroot()
    pairs = {frozenset([n.get("body1"), n.get("body2")]) for n in prepared.findall("./contact/exclude")}
    expected = {frozenset([f"robot_gripper_r_inner_link{i}", "robot_gripper_r_inner_link2"]) for i in [1, 3, 4]}
    assert pairs == expected
    assert ET.tostring(prepared.find("equality")) == ET.tostring(root.find("equality"))
    second = prepare_model(out, tmp_path / "again.xml")
    assert len(ET.parse(second).findall("./contact/exclude")) == 3
