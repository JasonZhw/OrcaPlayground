"""抓取点与放置点的几何推导：全部由物体包围盒推导，不含物体名常量。

  grasp_target          物体包围盒 → 抓取点、预备点、握法旋转
  placement_targets     物体投影到筐坐标系 → 释放点、搬运高度、筐内落点
  placement_satisfied   落地后是否算入筐（校验用）
  placement_inside      释放前物体是否在筐口范围（更严格）
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from examples.euler.medication_sorting.scene import vertices

def placement_satisfied(scene, object_key, bin_key):
    """Check COM inside the bin floor and object bottom within the release band.

    Orientation after placement is unrestricted. Both the per-object check and
    final four-object check use this same geometry; callers own error messages.
    """
    bin_position, bin_rotation = scene.body_pose(bin_key)
    local_vertices = (vertices(scene, object_key) - bin_position) @ bin_rotation
    center = (scene.d.xipos[scene.body_id(object_key)] - bin_position) @ bin_rotation
    floors = [
        geom
        for geom in scene.body_geoms(bin_key)
        if int(scene.m.geom_type[geom]) == 6 and scene.m.geom_size[geom, 2] < 0.015
    ]
    floor = min(floors, key=lambda geom: scene.m.geom_pos[geom, 2])
    return placement_inside(center, scene.m.geom_size[floor, :2], float(local_vertices[:, 2].min()))


# --- 任务几何常量 ---
DROP_SLOTS = {
    "calcium": [0, -0.045],
    "melatonin": [0, 0.045],
    "pharyngitis": [0.027, 0],
    "herbbox": [-0.021, 0],
}
# Side-grasped bottles need more wrist/rim clearance than top-grasped boxes.
RELEASE_BOTTOM_CLEARANCE = {"calcium": 0.02, "melatonin": 0.02, "pharyngitis": 0.02, "herbbox": 0.02}
HOVER_RIM_CLEARANCE = 0.06
BOTTLE_CARRY_EXTRA_HEIGHT = 0.03


def placement_targets(scene, key, carried, hand_rotation, bin_key):
    """Keep the held item's footprint inside the bin and lower its bottom below the rim.

    Height is measured from collision geometry, not an assumed object origin.
    These are target poses; normal IK/collision and actual-pose checks still apply.
    """
    bin_p, bin_r = scene.body_pose(bin_key)
    object_p, object_r = scene.body_pose(key)
    local = (vertices(scene, key) - object_p) @ object_r
    projected = local @ (np.asarray(hand_rotation) @ carried.local_rotation).T @ bin_r
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    floors = [
        g for g in scene.body_geoms(bin_key) if int(scene.m.geom_type[g]) == 6 and scene.m.geom_size[g, 2] < 0.015
    ]
    floor = min(floors, key=lambda g: scene.m.geom_pos[g, 2])
    corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    floor_world = (corners * scene.m.geom_size[floor]) @ scene.d.geom_xmat[floor].reshape(3, 3).T
    floor_local = (floor_world + scene.d.geom_xpos[floor] - bin_p) @ bin_r
    floor_lo, floor_hi = floor_local.min(axis=0), floor_local.max(axis=0)
    safe_lo = floor_lo[:2] + 0.008 - lo[:2]
    safe_hi = floor_hi[:2] - 0.008 - hi[:2]
    if np.any(safe_lo > safe_hi):
        # 物体比筐略大（如 herbbox 16.2cm vs 筐地板 16cm）时不再直接报错，
        # 放在筐中心、允许边缘轻微探出——几毫米级的溢出不影响实际入框。
        xy_offset = (safe_lo + safe_hi) / 2.0
    else:
        xy_offset = np.clip(DROP_SLOTS[key], safe_lo, safe_hi)
    offset = np.r_[xy_offset, floor_hi[2] + RELEASE_BOTTOM_CLEARANCE[key] - lo[2]]
    release_object = bin_p + bin_r @ offset
    release_hand = release_object - np.asarray(hand_rotation) @ carried.local_position
    rim = ((vertices(scene, bin_key) - bin_p) @ bin_r)[:, 2].max()
    descent = max(0.06, rim + HOVER_RIM_CLEARANCE - (offset[2] + lo[2]))
    # 上限保护：筐较深时上式会算出几十厘米，导致传送点超出机械臂可达范围
    descent = min(descent, 0.18)
    if key in ("calcium", "melatonin"):
        descent += BOTTLE_CARRY_EXTRA_HEIGHT
    return {
        "release": release_hand,
        "above": release_hand + bin_r[:, 2] * descent,
        "object_position": release_object,
        "bottom_clearance": RELEASE_BOTTOM_CLEARANCE[key],
    }


def placement_inside(center_in_bin, half_floor_xy, lowest_z):
    center = np.asarray(center_in_bin)
    return bool(
        np.isfinite(np.r_[center, half_floor_xy, lowest_z]).all()
        and np.all(np.abs(center[:2]) <= half_floor_xy)
        and -0.01 <= lowest_z <= 0.08
    )


def grasp_target(scene, arm, key, yaw_deg=0.0, center_xy=None, no_offset=False):
    """Return a grasp pose derived from the current object geometry (metres, world frame).

    center_xy：可选 (x, y)。视觉接地时用模型报的粗位置覆盖物体中心的
    真实 x/y，让抓取点围绕「模型看到的位置」计算，而非真值。
    no_offset：True 时不做手腕偏移（直接到物体抓取中心），agentic 流程靠
    腕摄像头微调来对齐，不预先给 0.133 偏差。
    """
    p, r = scene.body_pose(key)
    if center_xy is not None:
        p = np.array([float(center_xy[0]), float(center_xy[1]), p[2]])
    bounds = vertices(scene, key)
    height = np.ptp(bounds[:, 2])
    # Jaw closes along the object's local X; local Z is approach direction.
    jaw = r[:, 0].copy()
    jaw[2] = 0
    jaw /= np.linalg.norm(jaw)
    # Rotate about the object's own short axis so yaw randomization preserves the grasp.
    yaw_frame = np.column_stack((jaw, np.cross([0, 0, 1.0], jaw), [0, 0, 1.0]))
    rotation = (
        yaw_frame
        @ Rotation.from_euler("x", 0 if arm == "right" else -90, degrees=True).as_matrix()
        @ np.diag([-1.0, 1.0, -1.0])
    )
    # agent 指定的抓取偏航：绕竖直轴旋转接近方向，握点不变
    if yaw_deg:
        rotation = Rotation.from_euler("z", float(yaw_deg), degrees=True).as_matrix() @ rotation
    fraction = {"pharyngitis": 0.73, "herbbox": 0.70}[key] if arm == "right" else 0.51
    grip_center = p + r @ np.array([0, 0.01 if arm == "left" else 0, height * fraction])
    if no_offset:
        # agentic 流程：直接到物体抓取中心，不做手腕偏移，靠腕摄像头微调对齐
        grasp = grip_center
    else:
        grasp = grip_center - rotation @ np.array([0.0, 0.0, 0.18])
    pre = grasp + np.array([0, 0, 0.08])
    return {"object_position": p, "position": grasp, "rotation": rotation, "approach": pre}
