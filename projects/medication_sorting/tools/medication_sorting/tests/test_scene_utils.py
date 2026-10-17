"""离线验证分类规则、场景变换和诊断录制的有效性。"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from scene_utils import (  # noqa: E402
    analyze_layout,
    match_body_names,
    validate_observation,
)

LAYOUT = TASK_DIR.parents[1] / "data/medication_sorting/scenes/drag_classfication.json"


def test_real_layout_classifies_all_four_medicines():
    report = analyze_layout(json.loads(LAYOUT.read_text()))
    assert report["asset_count"] == 21
    assert report["missing_task_actors"] == []
    assert [item["target_bin"] for item in report["tasks"]] == [
        "blue_plastic_crates_1",
        "blue_plastic_crates_1",
        "purple_plastic_crates_1",
        "purple_plastic_crates_1",
    ]


def test_group_transform_changes_world_position():
    # 父节点缩放 2、绕 Z 旋转 90°、平移 10m。子节点 (1,0,0) 的世界坐标为 (10,2,0)。
    q = 2**-0.5
    layout = {
        "actors": [{
            "name": "group", "path": "/group", "type": "GroupActor",
            "transform": {"position": [10, 0, 0], "rotation": [q, 0, 0, q], "scale": 2},
            "children": [{
                "name": "item", "path": "/group/item", "type": "AssetActor",
                "transform": {"position": [1, 0, 0], "rotation": [1, 0, 0, 0], "scale": 1},
            }],
        }],
    }
    report = analyze_layout(layout)
    np.testing.assert_allclose(report["assets"][0]["world_position"], [10, 2, 0], atol=1e-12)


def test_duplicate_paths_are_rejected():
    layout = json.loads(LAYOUT.read_text())
    layout["actors"].append(copy.deepcopy(layout["actors"][-1]))
    with pytest.raises(ValueError, match="重复"):
        analyze_layout(layout)


def test_invalid_quaternion_is_rejected():
    layout = json.loads(LAYOUT.read_text())
    layout["actors"][0]["transform"]["rotation"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="四元数"):
        analyze_layout(layout)


def test_body_matching_does_not_bind_similarly_named_actor():
    available = ["unitree_g1_1", "unitree_g1_1_pelvis", "unitree_g1_10_pelvis", "other"]
    assert match_body_names("unitree_g1_1", available) == ["unitree_g1_1", "unitree_g1_1_pelvis"]


def test_body_matching_uses_asset_suffix_for_spawned_object_names():
    available = [
        "[3160016717367]_Box_Purple",
        "[3331815409207]_Box_Blue",
        "[3503614101047]_Medical_32",
        "openloong_gripper_2f85_fix_base_usda_Medical_31",
        "[3641053054519]_Medical_23",
        "[3684002727479]_Medical_12",
        "Medical_320_wrong_suffix",
    ]
    expected = {
        "blue_plastic_crates_1": ["[3331815409207]_Box_Blue"],
        "purple_plastic_crates_1": ["[3160016717367]_Box_Purple"],
        "calcium_gluconate_tablets_bottle_1": ["[3503614101047]_Medical_32"],
        "melatonin_tablet_bottle_1": ["openloong_gripper_2f85_fix_base_usda_Medical_31"],
        "fu_shi_gao_herb_drink_packaging_box_1": ["[3641053054519]_Medical_23"],
        "pharyngitis_tablets_box_1": ["[3684002727479]_Medical_12"],
    }
    paths = {
        "blue_plastic_crates_1": "assets/e071469a36d3c8aa/autoproductionline_warehouse/autoproductionline_warehouse/prop/box_01/box_blue",
        "purple_plastic_crates_1": "assets/e071469a36d3c8aa/autoproductionline_warehouse/autoproductionline_warehouse/prop/box_01/box_purple",
        "calcium_gluconate_tablets_bottle_1": "assets/e071469a36d3c8aa/medical/medical/medical_32/medical_32",
        "melatonin_tablet_bottle_1": "assets/e071469a36d3c8aa/medical/medical/medical_31/medical_31",
        "fu_shi_gao_herb_drink_packaging_box_1": "assets/e071469a36d3c8aa/medical/medical/medical_23/medical_23",
        "pharyngitis_tablets_box_1": "assets/e071469a36d3c8aa/medical/medical/medical_12/medical_12",
    }
    for actor, names in expected.items():
        assert match_body_names(actor, available, asset_path=paths[actor]) == names


def test_observation_nan_and_size_mismatch_are_rejected():
    with pytest.raises(ValueError, match="非有限"):
        validate_observation([float("nan")], [0], [0], nq=1, nv=1)
    with pytest.raises(ValueError, match="维度"):
        validate_observation([1, 2], [0], [0], nq=1, nv=1)


def test_observation_keeps_free_joint_dimensions():
    # free joint 的 qpos 为 7 维、qvel/qacc 为 6 维，不能错误地要求三者同维。
    validate_observation(np.zeros(7), np.zeros(6), np.zeros(6), nq=7, nv=6)
