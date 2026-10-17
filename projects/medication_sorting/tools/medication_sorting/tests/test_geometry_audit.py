import copy

import mujoco
import numpy as np
import pytest

from tools.medication_sorting import geometry_audit as audit


def test_projection_ignores_visual_geoms_and_includes_children():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object" quat=".70710678 0 0 .70710678">
        <geom type="box" size=".01 .04 .02"/>
        <geom type="sphere" size="1" contype="0" conaffinity="0"/>
        <body pos=".03 0 0"><geom type="sphere" size=".005"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(audit.body_interval(model, data, 'object', [1, 0, 0]),
                               [-.04, .04], atol=1e-8)
    np.testing.assert_allclose(audit.body_interval(model, data, 'object', [0, 1, 0]),
                               [-.01, .035], atol=1e-8)


def test_cylinder_projection_uses_radius_and_half_height_not_unused_third_size():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="bottle"><geom type="cylinder" size=".0315 .0511 .7"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(audit.body_interval(model, data, 'bottle', [1, 0, 0]),
                               [-.0315, .0315])
    np.testing.assert_allclose(audit.body_interval(model, data, 'bottle', [0, 0, 1]),
                               [-.0511, .0511])


def test_gap_uses_facing_surfaces_not_body_centers():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="positive" pos="0 0 .05"><geom type="box" size=".01 .01 .01"/></body>
      <body name="negative" pos="0 0 -.05"><geom type="box" size=".01 .01 .01"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert audit.facing_gap(model, data, 'positive', 'negative', [0, 0, 1]) == pytest.approx(.08)
    assert audit.facing_gap(model, data, 'positive', 'negative', [0, 0, -1]) == pytest.approx(.08)


def test_base_variant_is_independent_and_keeps_targets_stationary():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base" pos="1 2 3"><geom type="sphere" size=".1"/></body>
      <body name="target" pos="4 5 6"><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    before = copy.copy(model.body_pos)
    variant = audit.translated_base(model, 'base', [.05, 0, 0])
    np.testing.assert_array_equal(model.body_pos, before)
    np.testing.assert_allclose(variant.body('base').pos, [1.05, 2, 3])
    np.testing.assert_allclose(variant.body('target').pos, [4, 5, 6])


@pytest.mark.parametrize('axis', [[0, 0, 0], [1, 2], [float('nan'), 0, 1]])
def test_projection_rejects_invalid_axes(axis):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    with pytest.raises(ValueError):
        audit.body_interval(model, mujoco.MjData(model), 'world', axis)
