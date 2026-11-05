"""Fixed fixtures must remain physical obstacles and render at their true pose."""

import asyncio
import importlib

import mujoco
import numpy as np
import pytest

XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom type="plane" size="2 2 .1"/>
<body name="purple" pos="0 0 .15"><freejoint name="purple_free"/>
<geom type="box" size=".2 .2 .05"/></body>
<body name="medicine" pos="0 0 .5"><freejoint name="medicine_free"/>
<geom type="sphere" size=".04" mass=".1"/></body>
<body name="blue" pos="1 0 .15"><joint name="blue_free" type="free"/>
<geom type="box" size=".2 .2 .05"/></body>
<body name="arm" pos="0 1 1"><joint name="hinge"/>
<geom type="capsule" size=".02 .2"/></body>
</worldbody></mujoco>"""


def prepare(tmp_path):
    mod = importlib.import_module("examples.euler.medication_sorting.pi05.fixed_bins")
    source, target = tmp_path / "original.xml", tmp_path / "fixed.xml"
    source.write_text(XML)
    mapping = mod.prepare_fixed_bins(source, target, ["purple", "blue"])
    return mod, source, target, mapping


def test_static_bins_resist_force_and_still_support_falling_medicine(tmp_path):
    _, source, target, mapping = prepare(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(target))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [model.body(n).id for n in ("purple", "blue")]
    poses = data.xpos[ids].copy()
    quats = data.xquat[ids].copy()
    for body in ids:
        assert model.body_jntnum[body] == 0
        data.xfrc_applied[body] = [10000, -10000, 10000, 10000, 0, 0]
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.xpos[ids], poses)
    np.testing.assert_array_equal(data.xquat[ids], quats)
    assert 0.23 < data.xpos[model.body("medicine").id, 2] < 0.25
    assert source.read_text() == XML
    assert mapping.source_nq - mapping.physics_nq == 14


def test_render_mapping_preserves_joints_on_both_sides_of_removed_bins(tmp_path):
    mod, source, target, mapping = prepare(tmp_path)
    original = mujoco.MjModel.from_xml_path(str(source))
    fixed = mujoco.MjModel.from_xml_path(str(target))
    qpos = fixed.qpos0.copy()
    qpos[fixed.joint("medicine_free").qposadr[0]] = 0.13
    qpos[fixed.joint("hinge").qposadr[0]] = 0.42
    expanded = mapping.expand(qpos)
    assert expanded.shape == (original.nq,)
    d1, d2 = mujoco.MjData(original), mujoco.MjData(fixed)
    d1.qpos[:], d2.qpos[:] = expanded, qpos
    mujoco.mj_forward(original, d1)
    mujoco.mj_forward(fixed, d2)
    for name in ("purple", "blue", "medicine", "arm"):
        np.testing.assert_allclose(d1.xpos[original.body(name).id], d2.xpos[fixed.body(name).id])
        np.testing.assert_allclose(d1.xmat[original.body(name).id], d2.xmat[fixed.body(name).id])

    class Bridge:
        async def render(self, q, *args, **kwargs):
            np.testing.assert_array_equal(q, expanded)
            assert args == (1.2,)
            assert kwargs == {"simulate_index": 17, "request_idr": True}
            return "ok"

    wrapped = mod.FixedBinRenderBridge(Bridge(), mapping)
    assert asyncio.run(wrapped.render(qpos, 1.2, simulate_index=17, request_idr=True)) == "ok"
    with pytest.raises(ValueError):
        mapping.expand(np.zeros(original.nq))


def test_refuses_unknown_bins_and_does_not_overwrite_source(tmp_path):
    mod, source, _, _ = prepare(tmp_path)
    with pytest.raises(ValueError):
        mod.prepare_fixed_bins(source, source, ["purple", "blue"])
    with pytest.raises(ValueError):
        mod.prepare_fixed_bins(source, tmp_path / "bad.xml", ["purple", "missing"])
