"""只读物理快照与离线IK复核；诊断解不发送给机器人。"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.medication_sorting.robot_control import RIGHT_JOINTS, SITE, solve_site_ik


def capture_snapshot(model, data, *, target, basket, site, joint_names):
    body = model.body(target).id
    container = model.body(basket).id
    end = model.site(site).id
    contacts = []
    for i, contact in enumerate(data.contact[:data.ncon]):
        names = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
        if target not in names and not any(name.startswith('unitree_g1_1_') for name in names):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        contacts.append({'bodies': names, 'geoms': [int(contact.geom1), int(contact.geom2)],
                         'position_m': contact.pos.tolist(), 'distance_m': float(contact.dist),
                         'normal_force_n': float(force[0]), 'contact_wrench': force.tolist(),
                         'force_world_n': (contact.frame.reshape(3, 3).T @ force[:3]).tolist()})
    joints = {}
    for name in joint_names:
        joint = model.joint(name)
        value = float(data.qpos[joint.qposadr[0]])
        joints[name] = {'position_rad': value, 'range_rad': joint.range.tolist(),
                        'limit_margin_rad': float(min(value-joint.range[0], joint.range[1]-value))}
    return {'time': float(data.time), 'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
            'actual_ctrl': data.ctrl.tolist(), 'actuator_force': data.actuator_force.tolist(),
            'ee_position_m': data.site_xpos[end].tolist(),
            'ee_rotation': data.site_xmat[end].reshape(3, 3).tolist(),
            'target_position_m': data.xpos[body].tolist(),
            'target_quaternion_wxyz': data.xquat[body].tolist(),
            'basket_position_m': data.xpos[container].tolist(),
            'basket_quaternion_wxyz': data.xquat[container].tolist(),
            'joints': joints, 'contacts': contacts,
            'force_sampling': 'MuJoCo最后一次动力学计算的接触力，非控制步内峰值'}


def compare_ik_attempts(model, qpos, joint_names, site, position, rotation, *, attempts=(1, 8, 32)):
    rows = []
    indices = [model.joint(name).qposadr[0] for name in joint_names]
    limits = np.array([model.joint(name).range for name in joint_names])
    for count in attempts:
        row = {'seed_attempts': count, 'solved': False}
        try:
            result = solve_site_ik(model, qpos, joint_names, site, position, rotation,
                                   attempts=count, position_tolerance=.008, rotation_tolerance=.09,
                                   joint_regularization=.001)
            q = result['joints']
            row.update(solved=True, position_error_m=result['position_error'],
                       rotation_error_rad=result['rotation_error'], joints=q.tolist(),
                       largest_joint_target_change_rad=float(np.max(np.abs(q-np.asarray(qpos)[indices]))),
                       minimum_joint_limit_margin_rad=float(np.minimum(q-limits[:, 0], limits[:, 1]-q).min()))
        except RuntimeError as exc:
            row['error'] = str(exc)
        rows.append(row)
    solved = [row for row in rows if row['solved']]
    classification = 'no_feasible_solution_found'
    if solved:
        classification = 'single_seed_solved' if solved[0]['seed_attempts'] == 1 else 'multi_seed_solved'
    return {'classification': classification, 'attempts': rows,
            'mathematically_proven_unreachable': False, 'executed_on_robot': False,
            'collision_or_path_checked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.report.read_text())
    failure = original['failed_target']
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    seed = np.asarray(failure['ik_seed_qpos'])
    result = compare_ik_attempts(model, seed, RIGHT_JOINTS, SITE,
                                 failure['position'], failure['rotation'])
    result.update(source_report=str(args.report.resolve()), source_xml=str(args.xml.resolve()),
                  failed_phase=failure['phase'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
