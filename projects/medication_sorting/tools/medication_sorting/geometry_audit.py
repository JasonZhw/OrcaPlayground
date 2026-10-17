"""只读碰撞尺寸与离线逆运动学诊断；不改场景，不产生VLA动作。"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools.medication_sorting.robot_control import (
    GripperCalibration, HOME_POSITION, PREFIX, RIGHT_JOINTS, SITE,
    TOP_DOWN_ROTATION, solve_site_ik,
)
from tools.medication_sorting.run_expert import (
    box_bounds, effective_placement_offset, grasp_rotation, unique_body, validate_live_profile,
)
from tools.medication_sorting.run_four_objects import DROP_OFFSETS
from tools.medication_sorting.task_specs import TARGETS, target_spec

PROFILE = 'randomized_tabletop_70'


def collision_geoms(model, name):
    bodies = {model.body(name).id}
    for b in range(model.nbody):
        if model.body_parentid[b] in bodies:
            bodies.add(b)
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def body_interval(model, data, name, axis):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError('投影轴必须是有限非零三维向量')
    axis = axis / np.linalg.norm(axis)
    intervals = []
    for g in collision_geoms(model, name):
        kind, size = model.geom_type[g], model.geom_size[g]
        direction = data.geom_xmat[g].reshape(3, 3).T @ axis
        center = float(data.geom_xpos[g] @ axis)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            radius = float(np.abs(direction) @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0] + size[1] * abs(direction[2])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radius = np.linalg.norm(direction * size)
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            projection = model.mesh_vert[start:start + count] @ direction + center
            intervals.append([float(projection.min()), float(projection.max())])
            continue
        else:
            raise ValueError(f'不支持的碰撞类型：{kind}')
        intervals.append([center - radius, center + radius])
    if not intervals:
        raise ValueError(f'{name} 没有碰撞几何')
    return np.min(intervals, axis=0)[0], np.max(intervals, axis=0)[1]


def facing_gap(model, data, first, second, axis):
    a, b = sorted([body_interval(model, data, name, axis) for name in (first, second)])
    return float(b[0] - a[1])


def translated_base(model, name, offset):
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('基座偏移必须是有限三维向量')
    variant = copy.copy(model)
    body = variant.body(name).id
    if variant.body_jntnum[body] or variant.body_parentid[body] != 0:
        raise ValueError('只支持世界坐标下的固定基座')
    variant.body_pos[body] += offset
    return variant


def measure(model):
    data = mujoco.MjData(model)
    calibration = GripperCalibration(model)
    gap_curve = []
    for closure in (0., .25, .5, .75, 1.):
        data.qpos[calibration.indices] = calibration.targets(closure)
        mujoco.mj_forward(model, data)
        axis = data.site_xmat[model.site(SITE).id].reshape(3, 3)[:, 2]
        gap = facing_gap(model, data, *(model.body(b).name for b in calibration.tips), axis)
        gs = [collision_geoms(model, model.body(b).name) for b in calibration.tips]
        distance = min(mujoco.mj_geomDistance(model, data, a, b, 1., np.zeros(6))
                       for a in gs[0] for b in gs[1])
        gap_curve.append({'closure': closure, 'conservative_slab_gap_m': gap,
                          'closest_tip_distance_m': float(distance)})
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    opening = gap_curve[0]['conservative_slab_gap_m']
    objects, baskets = {}, {}
    for target, spec in TARGETS.items():
        name = unique_body(model, spec['body_suffix'])
        low, high = box_bounds(model, data, name, frame_name=name)
        width_by_angle = []
        for degrees in (0., 5., 10., 15., 30.):
            # 手指关闭轴固定，旋转药品；等价地在原药品上旋转投影轴。
            axis = Rotation.from_euler('z', -degrees, degrees=True).apply([1., 0., 0.])
            left, right = body_interval(model, data, name, axis)
            width_by_angle.append({'relative_yaw_deg': degrees, 'width_m': float(right - left),
                                   'opening_minus_width_m': float(opening - (right - left))})
        objects[target] = {'body': name, 'local_collision_size_m': (high-low).tolist(),
                           'mass_kg': float(model.body(name).mass[0]),
                           'width_by_relative_yaw': width_by_angle}
        basket = unique_body(model, spec['basket_suffix'])
        b = model.body(basket).id
        floors = [g for g in collision_geoms(model, basket)
                  if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                  and model.geom_size[g, 2] < .2 * min(model.geom_size[g, :2])]
        floor = max(floors, key=lambda g: np.prod(model.geom_size[g, :2]))
        size = model.geom_size[floor]
        baskets[spec['basket_suffix']] = {'body': basket, 'world_position_m': data.xpos[b].tolist(),
                                          'floor_size_m': (2 * size).tolist()}
        offset = effective_placement_offset(DROP_OFFSETS[target], target_spec(target, profile=PROFILE))
        obj_lo, obj_hi = box_bounds(model, data, name, frame_name=basket)
        half = (obj_hi[:2] - obj_lo[:2]) / 2
        margins = size[:2] - np.abs(offset - model.geom_pos[floor, :2]) - half
        objects[target]['upright_drop_offset_m'] = offset.tolist()
        objects[target]['upright_floor_margin_xy_m'] = margins.tolist()
    return {'gripper_gap_curve': gap_curve, 'objects': objects, 'baskets': baskets,
            'gap_caveat': '正净空是整个指尖沿关闭轴的保守投影间隙；负值不能证明任意局部夹持都不可能。'
                          '闭合曲线为无物体的运动学目标，负值代表无载目标重叠，非实际穿透。',
            'packing_caveat': '落筐余量仅为直立姿态的底板几何筛查，不等于带物路径或连续分类成功。'}


def reachable(model, qpos, position, rotation, attempts):
    try:
        solution = solve_site_ik(model, qpos, RIGHT_JOINTS, SITE, position, rotation,
                                 position_tolerance=.008, rotation_tolerance=.09,
                                 attempts=attempts, joint_regularization=.001)
        limits = np.array([model.joint(n).range for n in RIGHT_JOINTS])
        margins = np.minimum(solution['joints'] - limits[:, 0], limits[:, 1] - solution['joints'])
        return {'reachable': True, 'position_error_m': solution['position_error'],
                'rotation_error_rad': solution['rotation_error'],
                'minimum_joint_limit_margin_rad': float(margins.min()),
                'joints': solution['joints'].tolist()}, solution['qpos']
    except RuntimeError as exc:
        return {'reachable': False, 'error': str(exc)}, qpos


def reachability(model, *, attempts=4):
    variants = {'original': [0., 0., 0.], 'base_x_plus5cm': [.05, 0., 0.],
                'base_x_minus5cm': [-.05, 0., 0.], 'base_y_plus5cm': [0., .05, 0.],
                'base_y_minus5cm': [0., -.05, 0.], 'base_z_plus5cm': [0., 0., .05],
                'base_z_minus5cm': [0., 0., -.05],
                'base_y_minus1_5cm': [0., -.015, 0.], 'base_y_minus2_5cm': [0., -.025, 0.]}
    results = {}
    for variant, offset in variants.items():
        current = translated_base(model, PREFIX+'pelvis', offset)
        data = mujoco.MjData(current)
        mujoco.mj_forward(current, data)
        home, seed = reachable(current, data.qpos, HOME_POSITION, TOP_DOWN_ROTATION, attempts)
        rows = []
        for target, original_spec in TARGETS.items():
            spec = target_spec(target, profile=PROFILE)
            body = unique_body(current, spec['body_suffix'])
            low, high = box_bounds(current, data, body)
            xy = (low[:2]+high[:2])/2
            basket = current.body(unique_body(current, spec['basket_suffix'])).id
            drop = effective_placement_offset(DROP_OFFSETS[target], spec)
            destination = (data.xpos[basket] + data.xmat[basket].reshape(3, 3) @ np.r_[drop, 0.])[:2]
            rotation = grasp_rotation(target, profile=PROFILE)
            goals = [('approach', [*xy, min(high[2]+.14, spec['travel_z'])], rotation),
                     ('grasp', [*xy, high[2]+spec.get('grasp_clearance', -.005)], rotation),
                     ('lift', [*xy, spec['travel_z']], rotation),
                     ('carry', [*destination, spec['travel_z']], rotation),
                     ('lower', [*destination, spec.get('lower_z', 4.99)], rotation)]
            qpos = seed.copy()
            for phase, position, orientation in goals:
                status, qpos = reachable(current, qpos, position, orientation, attempts)
                rows.append({'target': target, 'phase': phase, 'position_m': list(position), **status})
            # 只筛查原位周围四个工作空间角点，不把有限采样当作全空间保证。
            for dx, dy in ((-.02, -.02), (-.02, .02), (.02, -.02), (.02, .02)):
                position = [xy[0]+dx, xy[1]+dy, high[2]+spec.get('grasp_clearance', -.005)]
                status, _ = reachable(current, seed, position, rotation, attempts)
                rows.append({'target': target, 'phase': 'grasp_corner', 'offset_xy_m': [dx, dy],
                             'position_m': position, **status})
        results[variant] = {'base_translation_m': offset, 'home': home,
                            'reachable_count': sum(row['reachable'] for row in rows),
                            'waypoint_count': len(rows), 'waypoints': rows}
        print(variant, results[variant]['reachable_count'], '/', len(rows), flush=True)
    return {'variants': results, 'attempts_per_waypoint': attempts,
            'caveat': '有限次数数值IK仅筛查离散末端位姿；未求解不等于数学证明不可达。'
                      '未检查整段路径、碰撞、夹持力或动力学；带物中心反馈会改变实际末端目标。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=4)
    parser.add_argument('--skip-ik', action='store_true')
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('attempts至少为1')
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    validate_live_profile(model, profile=PROFILE)
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'source_xml': str(args.xml.resolve()),
              'source_xml_sha256': hashlib.sha256(args.xml.read_bytes()).hexdigest(),
              'mode': 'offline_geometry_and_ik_only', 'scene_modified': False,
              'training_started': False, 'vla_evaluated': False, 'measurements': measure(model)}
    path = args.output/'geometry.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.skip_ik:
        report['kinematic_screen'] = reachability(model, attempts=args.attempts)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('报告', path, flush=True)


if __name__ == '__main__':
    main()
