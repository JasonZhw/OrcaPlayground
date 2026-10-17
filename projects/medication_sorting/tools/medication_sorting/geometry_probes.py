"""独立专家物理诊断，不采训练数据、不运行策略、不覆盖原场景。"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.medication_sorting.infer_policy import grasp_streak
from tools.medication_sorting.robot_control import FixedBaseRobot
from tools.medication_sorting.run_expert import execute, placement_geometry, unique_body
from tools.medication_sorting.run_randomized_experiment import Experiment, LAYOUT, PYTHON, ROOT
from tools.medication_sorting.task_specs import TARGETS


def strict_evidence(trace, final_samples, *, start_z):
    streak = maximum = 0
    for row in trace:
        streak = grasp_streak(streak, set(row['contact_sides']), row['box_pos'][2]-start_z)
        maximum = max(maximum, streak)
    stable = len(final_samples) >= 20 and all(
        row['inside'] and row['released'] and row['settled'] for row in final_samples[-20:])
    return {'success': bool(maximum >= 3 and stable), 'max_grasp_streak': maximum,
            'final_twenty_steps_passed': bool(stable)}


def probe_cases():
    rows = [{'name': 'nominal_'+target, 'target': target} for target in TARGETS]
    herb = 'fu_shi_gao_herb_drink_packaging_box_1'
    rows += [{'name': 'herb_yaw15_fixed', 'target': herb,
              'reset_yaw': np.deg2rad(15.), 'align_grasp_to_object': False},
             {'name': 'herb_yaw15_aligned', 'target': herb,
              'reset_yaw': np.deg2rad(15.), 'align_grasp_to_object': True},
             {'name': 'herb_slot_bias', 'target': herb, 'placement_offset': (.018, 0.)},
             {'name': 'herb_slot_centered_y', 'target': herb, 'placement_offset': (.018, -.025)},
             {'name': 'melatonin_far_corner', 'target': 'melatonin_tablet_bottle_1',
              'reset_offset': (.02, -.02)},
             {'name': 'herb_far_corner', 'target': herb, 'reset_offset': (-.02, -.02)}]
    return rows


def make_base_layout(source, dy):
    if not np.isfinite(dy) or abs(dy) > .05:
        raise ValueError('基座对照偏移限于正负5厘米')
    result = copy.deepcopy(source)
    robots = [row for row in result['actors'] if row['name'] == 'unitree_g1_1']
    if len(robots) != 1:
        raise ValueError('诊断场景需要唯一的顶层G1')
    robots[0]['transform']['position'][1] += dy
    return result


def run_cases(output, names=None, *, layout=LAYOUT, diagnostics=False, extra_grasp_tilt_x_degrees=0.):
    cases = [row for row in probe_cases() if names is None or row['name'] in names]
    if names is not None and set(names) != {row['name'] for row in cases}:
        raise ValueError('未知或重复的诊断用例')
    robot = FixedBaseRobot(frame_skip=200)
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'role': 'expert_physics_diagnosis',
              'control_hz': 5, 'independent_reset_per_case': True, 'vla_success_rate_measured': False,
              'original_scene_overwritten': False, 'source_layout': str(layout.resolve()),
              'source_layout_sha256': hashlib.sha256(layout.read_bytes()).hexdigest(),
              'initial_base_position_m': robot.base_reference.tolist(),
              'source_xml': str(robot.xml_path),
              'source_xml_sha256': hashlib.sha256(Path(robot.xml_path).read_bytes()).hexdigest(),
              'extra_grasp_tilt_x_degrees':extra_grasp_tilt_x_degrees,
              'cases': [], 'complete': False}
    try:
        for case in cases:
            params = dict(case)
            name = params.pop('name')
            target = params['target']
            folder = output/name
            body = unique_body(robot.model, TARGETS[target]['body_suffix'])
            basket = unique_body(robot.model, TARGETS[target]['basket_suffix'])
            b = robot.model.body(body).id
            qadr = robot.model.jnt_qposadr[robot.model.body_jntadr[b]]
            start_z = float(robot.model.qpos0[qadr+2])
            row = {'case': case, 'success': False, 'start_z': start_z}
            print('物理诊断开始', name, flush=True)
            try:
                execute(robot, folder, profile='randomized_tabletop_70', diagnostics=diagnostics,
                        extra_grasp_tilt_x_degrees=extra_grasp_tilt_x_degrees, **params)
            except Exception as exc:
                row['expert_error'] = f'{type(exc).__name__}: {exc}'
            trace_path = folder/'trace.jsonl'
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
            samples = []
            try:
                if not trace:
                    raise RuntimeError('未产生物理动作轨迹')
                # 原专家末段只有3秒；保持最后执行命令补足独立4秒验收，不添加救援动作。
                action = np.asarray(trace[-1]['action'])
                for _ in range(20):
                    robot.step(action)
                    geometry = placement_geometry(robot.model, robot.data, body, basket)
                    released = not any(
                        body in [robot.model.body(robot.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
                        and any('gripper_r' in robot.model.body(robot.model.geom_bodyid[g]).name
                                for g in (c.geom1, c.geom2))
                        for c in robot.data.contact[:robot.data.ncon])
                    samples.append({'time': robot.sim_time, 'inside': geometry['inside'],
                                    'released': released,
                                    'settled': bool(np.linalg.norm(robot.data.cvel[b, 3:]) < .02),
                                    'placement': geometry,
                                    'box_pos': robot.body_pose(body)['Pos'].tolist()})
                row.update(strict_evidence(trace, samples, start_z=start_z))
                row['original_expert_success'] = json.loads((folder/'report.json').read_text())['success']
                if 'expert_error' in row:
                    row['success'] = False
            except Exception as exc:
                row['verification_error'] = f'{type(exc).__name__}: {exc}'
                row['success'] = False
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'strict_verification.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
            (folder/'verification_trace.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
            result['cases'].append(row)
            (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print('物理诊断结果', json.dumps(row, ensure_ascii=False), flush=True)
        result['complete'] = len(result['cases']) == len(cases)
        (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        robot.disconnect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true', help='启动并在结束时关闭本任务独占的OrcaLab')
    parser.add_argument('--cases', nargs='+', choices=[row['name'] for row in probe_cases()])
    parser.add_argument('--base-offset-y', type=float, choices=[0., -.015, -.025], default=0.)
    parser.add_argument('--diagnostics', action='store_true', help='保存完整物理状态、接触力和失败IK种子')
    parser.add_argument('--extra-grasp-tilt-x-degrees', type=float, choices=[0., -5., -10.], default=0.)
    args = parser.parse_args()
    if args.base_offset_y and not args.launch:
        parser.error('基座对照必须由本程序启动独立场景')
    args.output.mkdir(parents=True, exist_ok=False)
    layout = LAYOUT
    if args.base_offset_y:
        layout = args.output/'diagnostic_layout.json'
        layout.write_text(json.dumps(make_base_layout(json.loads(LAYOUT.read_text()), args.base_offset_y),
                                     ensure_ascii=False, indent=2))
        (args.output/'variant.json').write_text(json.dumps({
            'source': str(LAYOUT.resolve()), 'source_sha256': hashlib.sha256(LAYOUT.read_bytes()).hexdigest(),
            'base_translation_m': [0., args.base_offset_y, 0.], 'other_changes': [],
            'original_scene_overwritten': False}, ensure_ascii=False, indent=2))
    experiment = Experiment(args.output/'app') if args.launch else None
    if experiment is not None:
        experiment.status.update(role='expert_physics_diagnosis', training_steps_requested=0,
                                 training_started=False, inference_started=False)
    try:
        if experiment is not None:
            experiment.start_app_with_retry()
            if args.base_offset_y:
                experiment.run([PYTHON, str(ROOT/'tools/medication_sorting/prepare_scene.py'),
                                '--layout', str(layout.resolve()), '--record-seconds', '0'],
                               'load_diagnostic_variant', timeout=300)
        result = run_cases(args.output, names=args.cases, layout=layout, diagnostics=args.diagnostics,
                           extra_grasp_tilt_x_degrees=args.extra_grasp_tilt_x_degrees)
    finally:
        if experiment is not None:
            experiment.stop_app()
    print('诊断完成', len(result['cases']), '个用例；不是VLA成功率或连续任务验收', flush=True)


if __name__ == '__main__':
    main()
