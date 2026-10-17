"""四药品严格连续专家试抓；不采训练RGB、不调用VLA、不自动训练。"""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools.medication_sorting.infer_policy import grasp_streak
from tools.medication_sorting.randomization import sample_reset
from tools.medication_sorting.robot_control import FixedBaseRobot, home_joints
from tools.medication_sorting.run_expert import execute, placement_geometry, unique_body
from tools.medication_sorting.run_four_objects import DROP_OFFSETS
from tools.medication_sorting.run_randomized_experiment import Experiment, LAYOUT
from tools.medication_sorting.task_specs import TARGETS

ORDER = tuple(reversed(TARGETS))
PROFILE = 'randomized_tabletop_70'


def round_drop_offset(target, *, herb_y_correction=0.):
    if not np.isfinite(herb_y_correction) or abs(herb_y_correction) > .03:
        raise ValueError('落点补偿对照不得超过正负3厘米')
    x, y = DROP_OFFSETS[target]
    if target == 'fu_shi_gao_herb_drink_packaging_box_1':
        y += herb_y_correction
    return x, y


class GraspLossGuard:
    def __init__(self):
        self.missing = 0

    def check(self, phase, contacts):
        self.missing = self.missing + 1 if phase in ('lift', 'carry', 'upright', 'lower') and not contacts else 0
        if self.missing >= 2:
            raise RuntimeError('夹持丢失：连续两个控制步无指尖接触，停止追踪掉落物')


class RoundEvidence:
    def __init__(self, start_z, *, per_target_limit=150, round_limit=600):
        self.start_z = start_z
        self.streak = dict.fromkeys(start_z, 0)
        self.maximum = dict.fromkeys(start_z, 0)
        self.recent = deque(maxlen=20)
        self.total_steps = self.target_steps = 0
        self.per_target_limit, self.round_limit = per_target_limit, round_limit

    def begin_target(self, target):
        if target not in self.start_z:
            raise ValueError('未知物品')
        self.target_steps = 0

    def check_budget(self):
        if self.total_steps >= self.round_limit or self.target_steps >= self.per_target_limit:
            raise RuntimeError('连续任务动作预算已耗尽')

    def append(self, status):
        if set(status) != set(self.start_z):
            raise ValueError('每个物理帧必须记录全部四个物品')
        self.total_steps += 1
        self.target_steps += 1
        for name, row in status.items():
            self.streak[name] = grasp_streak(self.streak[name], set(row['contact_sides']),
                                            row['position'][2] - self.start_z[name])
            self.maximum[name] = max(self.maximum[name], self.streak[name])
        self.recent.append(status)

    def target_passed(self, target):
        return bool(self.maximum[target] >= 3 and len(self.recent) == 20 and all(
            row[target]['inside'] and row[target]['released'] and row[target]['settled']
            for row in self.recent))

    def final_passed(self):
        return all(self.target_passed(name) for name in self.start_z)


def park_left_arm_at_initialization(robot):
    name = 'unitree_g1_1_left_shoulder_roll_joint'
    joint = robot.model.joint(name)
    angle = 1.2
    if not joint.range[0] <= angle <= joint.range[1]:
        raise ValueError('左臂避让初态超出实际模型限位')
    robot.data.qpos[joint.qposadr[0]] = angle
    robot.ctrl[robot.model.actuator(name).id] = angle


def initialize_round(robot, *, seed, randomize, park_left_arm=False, evidence_path=None):
    if type(seed) is not int or seed < 0:
        raise ValueError('整轮种子必须是非负整数')
    robot.reset(arm_home=home_joints(robot), closure=.5)
    if park_left_arm:
        park_left_arm_at_initialization(robot)
    initial = {}
    bodies = {}
    for i, (target, spec) in enumerate(TARGETS.items()):
        body = robot.model.body(unique_body(robot.model, spec['body_suffix'])).id
        bodies[body] = target
        joint = robot.model.body_jntadr[body]
        if joint < 0 or robot.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError('所有药品都必须是自由刚体')
        qadr = robot.model.jnt_qposadr[joint]
        offset, yaw = sample_reset(seed*4+i, profile='tabletop_v1') if randomize else (np.zeros(2), 0.)
        robot.data.qpos[qadr:qadr+2] += offset
        quat = Rotation.from_euler('z', yaw).as_quat()
        robot.data.qpos[qadr+3:qadr+7] = [quat[3], *quat[:3]]
        initial[target] = {'offset_m': offset.tolist(), 'yaw_rad': float(yaw)}
    mujoco.mj_forward(robot.model, robot.data)
    for body, name in bodies.items():
        initial[name]['position'] = robot.data.xpos[body].tolist()
        initial[name]['quaternion_wxyz'] = robot.data.xquat[body].tolist()
    # 子刚体也归入同一药品，碰撞检查不把本体与瓶盖当成两个物品。
    for b in range(robot.model.nbody):
        if robot.model.body_parentid[b] in bodies and b not in bodies:
            bodies[b] = bodies[robot.model.body_parentid[b]]
    overlaps = []
    for contact in robot.data.contact[:robot.data.ncon]:
        names = [bodies.get(robot.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
        if all(names) and names[0] != names[1] and contact.dist < -.001:
            overlaps.append({'objects': names, 'penetration_m': float(-contact.dist)})
    if evidence_path is not None:
        evidence_path.write_text(json.dumps({'reset_count': 1, 'initial': initial,
                                 'object_overlaps': overlaps}, ensure_ascii=False, indent=2))
    if overlaps:
        raise RuntimeError('轮首药品相互穿插，保留该种子而不静默换样：'+json.dumps(overlaps))
    return initial


def scene_status(robot):
    status = {}
    for target, spec in TARGETS.items():
        body = unique_body(robot.model, spec['body_suffix'])
        bid = robot.model.body(body).id
        basket = unique_body(robot.model, spec['basket_suffix'])
        sides, held = set(), False
        descendants = {bid}
        for b in range(robot.model.nbody):
            if robot.model.body_parentid[b] in descendants:
                descendants.add(b)
        for contact in robot.data.contact[:robot.data.ncon]:
            ids = [robot.model.geom_bodyid[g] for g in (contact.geom1, contact.geom2)]
            if not any(b in descendants for b in ids):
                continue
            names = [robot.model.body(b).name for b in ids]
            held |= any('gripper_' in n for n in names)
            for side in ('inner', 'outer'):
                if any('gripper_r_'+side+'_link4' in n for n in names):
                    sides.add(side)
        geometry = placement_geometry(robot.model, robot.data, body, basket)
        speed = float(np.linalg.norm(robot.data.cvel[bid, 3:]))
        status[target] = {'inside': geometry['inside'], 'released': not held, 'settled': speed < .02,
                          'linear_speed_m_s': speed, 'placement': geometry,
                          'position': robot.data.xpos[bid].tolist(),
                          'quaternion_wxyz': robot.data.xquat[bid].tolist(),
                          'contact_sides': sorted(sides)}
    return status


def execute_round(robot, output, *, seed, randomize=True, extra_grasp_tilt_x_degrees=0.,
                  park_left_arm=False, herb_y_correction=0., upright_herb=False,
                  melatonin_extra_tilt=None, herb_extra_tilt=None):
    output.mkdir(parents=True, exist_ok=False)
    report = {'role': 'continuous_expert_physics', 'success': False, 'seed': seed,
              'randomized_all_four': randomize, 'control_hz': 5, 'max_steps_per_target': 150,
              'max_steps_round': 600, 'reset_count': None, 'objects': [],
              'expert_profile': PROFILE, 'target_order': ORDER,
              'extra_grasp_tilt_x_degrees': extra_grasp_tilt_x_degrees,
              'left_arm_initial_pose': {'shoulder_roll_rad': 1.2} if park_left_arm else 'original',
              'herb_y_correction_m': herb_y_correction,
              'upright_herb_before_lower': upright_herb,
              'melatonin_extra_tilt_x_degrees': melatonin_extra_tilt,
              'herb_extra_tilt_x_degrees': herb_extra_tilt,
              'training_started': False, 'vla_success_rate_measured': False}
    attempted = set()
    evidence = None
    try:
        initial = initialize_round(robot, seed=seed, randomize=randomize, park_left_arm=park_left_arm,
                                   evidence_path=output/'initialization.json')
        report.update(initial=initial, reset_count=1)
        evidence = RoundEvidence({name: row['position'][2] for name, row in initial.items()})
        previous_time = robot.sim_time
        with (output/'continuous_trace.jsonl').open('w') as stream:
            for target in ORDER:
                evidence.begin_target(target)
                guard = GraspLossGuard()
                last_action = None
                row = {'target': target, 'attempted': True, 'success': False, 'reset_at_start': False}
                attempted.add(target)
                report['objects'].append(row)

                def record(r, trace_row):
                    nonlocal previous_time, last_action
                    if abs(r.sim_time-previous_time-.2) > 1e-6:
                        raise RuntimeError('物理时间不连续，疑似中途重置或控制频率改变')
                    previous_time = r.sim_time
                    last_action = np.asarray(trace_row['action'])
                    status = scene_status(r)
                    evidence.append(status)
                    stream.write(json.dumps({'time': r.sim_time, 'target': target,
                                 'phase': trace_row['phase'], 'state': r.state().tolist(),
                                 'action': last_action.tolist(), 'objects': status})+'\n')
                    stream.flush()
                    guard.check(trace_row['phase'], status[target]['contact_sides'])

                try:
                    print('连续专家开始', target, 'seed=', seed, flush=True)
                    tilt = extra_grasp_tilt_x_degrees
                    if 'melatonin' in target and melatonin_extra_tilt is not None:
                        tilt = melatonin_extra_tilt
                    if target.startswith('fu_shi') and herb_extra_tilt is not None:
                        tilt = herb_extra_tilt
                    execute(robot, output/target, target=target, profile=PROFILE, reset=False,
                            placement_offset=round_drop_offset(target, herb_y_correction=herb_y_correction),
                            align_grasp_to_object=True,
                            diagnostics=True, after_step=record, max_steps=150,
                            extra_grasp_tilt_x_degrees=tilt,
                            upright_before_lower=upright_herb and target.startswith('fu_shi'))
                    while not evidence.target_passed(target):
                        evidence.check_budget()
                        robot.step(last_action)
                        record(robot, {'action': last_action, 'phase': 'strict_stable_hold'})
                    row['success'] = True
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                row.update(steps=evidence.target_steps, max_grasp_streak=evidence.maximum[target])
                (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print('连续专家结果', json.dumps(row, ensure_ascii=False), flush=True)
                if not row['success']:
                    break
        report['success'] = bool(len(attempted) == 4 and all(r['success'] for r in report['objects'])
                                 and evidence.final_passed())
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if (output/'initialization.json').exists():
            report.update(json.loads((output/'initialization.json').read_text()))
        report['objects'].extend({'target': name, 'attempted': False, 'success': False}
                                  for name in ORDER if name not in attempted)
        if evidence is not None:
            report.update(total_steps=evidence.total_steps, max_grasp_streak=evidence.maximum,
                          joint_final_twenty_steps_passed=evidence.final_passed(),
                          final_scene=evidence.recent[-1] if evidence.recent else {})
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=70000)
    parser.add_argument('--nominal', action='store_true', help='只作工程冒烟，使用原始位置而非随机摆放')
    parser.add_argument('--extra-grasp-tilt-x-degrees', type=float, choices=[0., -10.], default=0.)
    parser.add_argument('--park-left-arm', action='store_true', help='仅在轮首设置闲置左臂避让初态')
    parser.add_argument('--herb-y-correction', type=float, choices=[0., -.015, -.025], default=0.)
    parser.add_argument('--upright-herb', action='store_true')
    parser.add_argument('--melatonin-extra-tilt', type=float, choices=[0., -10.])
    parser.add_argument('--herb-extra-tilt', type=float, choices=[0., -10.])
    args = parser.parse_args()
    if args.rounds <= 0 or args.seed < 0:
        parser.error('轮数必须为正，种子必须非负')
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = Experiment(args.output/'app') if args.launch else None
    robot = None
    summary = {'role': 'continuous_expert_physics', 'created_at': datetime.now(timezone.utc).isoformat(),
               'requested_rounds': args.rounds, 'results': [], 'complete': False,
               'training_started': False, 'vla_success_rate_measured': False}
    if experiment is not None:
        experiment.status.update(role='continuous_expert_physics', training_steps_requested=0,
                                 training_started=False, inference_started=False)
    try:
        if experiment is not None:
            experiment.start_app_with_retry()
        robot = FixedBaseRobot(frame_skip=200)
        paths = [Path(__file__), Path(__file__).with_name('run_expert.py'),
                 Path(__file__).with_name('robot_control.py'), Path(__file__).with_name('task_specs.py'),
                 LAYOUT, Path(robot.xml_path)]
        summary['source_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for i in range(args.rounds):
            row = execute_round(robot, args.output/f'round_{i:04d}', seed=args.seed+i,
                                randomize=not args.nominal,
                                extra_grasp_tilt_x_degrees=args.extra_grasp_tilt_x_degrees,
                                park_left_arm=args.park_left_arm, herb_y_correction=args.herb_y_correction,
                                upright_herb=args.upright_herb, melatonin_extra_tilt=args.melatonin_extra_tilt,
                                herb_extra_tilt=args.herb_extra_tilt)
            summary['results'].append(row)
            summary['successes'] = sum(r['success'] for r in summary['results'])
            summary['attempted_rounds'] = len(summary['results'])
            (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        summary['complete'] = True
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if robot is not None:
            robot.disconnect()
        try:
            if experiment is not None:
                experiment.stop_app()
        finally:
            (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('严格连续专家试验：', summary.get('successes', 0), '/', args.rounds,
          '；不是VLA成功率', flush=True)
    if summary.get('successes', 0) != args.rounds:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
