"""按物品补采合格回合；断点续跑不覆盖失败证据，不自动启动训练。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.medication_sorting.verify_episodes import verify_episode


def review_attempt(path, *, target, seed, configuration):
    quality = verify_episode(path)
    if quality['accepted'] and 'expert_profile' in configuration:
        expected = {'object_name': target, 'seed': seed, 'randomized_reset': True,
                    'expert_profile': configuration['expert_profile'],
                    'randomization_profile': configuration['randomization_profile'],
                    'experiment_layout_sha256': configuration['layout_sha256'],
                    'fps': configuration['control_hz']}
        if 'randomization_config' in configuration:
            expected['randomization_config'] = configuration['randomization_config']
        try:
            metadata = json.loads((Path(path)/'metadata.json').read_text())
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError('回合标签、种子或实验配置与批次不符')
        except (OSError, ValueError) as error:
            quality = {**quality, 'accepted': False, 'errors': [*quality['errors'], str(error)]}
    return quality


def collect_balanced(output, *, attempt, targets, per_target, max_attempts, seed_start,
                     configuration, max_new_attempts=None):
    if (not targets or len(set(targets)) != len(targets)
        or any(type(n) is not int or n <= 0 for n in (per_target, max_attempts))
        or max_attempts < per_target or max_attempts >= 10000
        or type(seed_start) is not int or seed_start < 0
        or (max_new_attempts is not None and (type(max_new_attempts) is not int or max_new_attempts <= 0))):
        raise ValueError('物品、配额、尝试上限和种子参数无效')
    output = Path(output)
    run = {'targets': list(targets), 'per_target': per_target, 'max_attempts': max_attempts,
           'seed_start': seed_start, 'configuration': configuration}
    output.mkdir(parents=True, exist_ok=True)
    run_path = output/'run_configuration.json'
    if run_path.exists():
        if json.loads(run_path.read_text()) != run:
            raise ValueError('断点续跑配置与已有批次不一致，拒绝覆盖')
    else:
        if any(output.iterdir()):
            raise ValueError('输出目录非空且缺少批次配置，拒绝覆盖')
        run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2))
    reviewed = 0
    last_reviewed = None
    def publish_review(state):
        status = {'state': state, 'reviewed_attempts': reviewed, 'last_episode': last_reviewed,
                  'updated_at': datetime.now(timezone.utc).isoformat()}
        temporary = output/'review_status.json.tmp'
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        temporary.replace(output/'review_status.json')

    publish_review('reviewing_history')
    results = {}
    for target in targets:
        folder = output/target
        folder.mkdir(exist_ok=True)
        results[target] = []
        for path in sorted(folder.glob('episode_*')):
            index = int(path.name.removeprefix('episode_'))
            quality = review_attempt(path, target=target,
                                     seed=seed_start+targets.index(target)*10000+index,
                                     configuration=configuration)
            results[target].append({'path': str(path.resolve()), 'index': index,
                                    'seed': seed_start+targets.index(target)*10000+index,
                                    'accepted': quality['accepted'], 'errors': quality['errors']})
            reviewed += 1
            last_reviewed = str(path.resolve())
            publish_review('reviewing_history')
    publish_review('history_review_complete')

    def publish():
        by_object = {target: {'attempts': len(rows),
                             'accepted': sum(row['accepted'] for row in rows),
                             'required': per_target} for target, rows in results.items()}
        report = {'complete': all(row['accepted'] == per_target for row in by_object.values()),
                  'accepted': sum(row['accepted'] for row in by_object.values()),
                  'attempts': sum(row['attempts'] for row in by_object.values()),
                  'by_object': by_object, 'results': results, 'configuration': configuration,
                  'training_started': False, 'continuous_sorting_validated': False}
        temporary = output/'progress.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(output/'progress.json')
        return report

    report = publish()
    new_attempts = 0
    while not report['complete']:
        attempted = False
        for target_index, target in enumerate(targets):
            counts = report['by_object'][target]
            if counts['accepted'] >= per_target or counts['attempts'] >= max_attempts:
                continue
            if max_new_attempts is not None and new_attempts >= max_new_attempts:
                return report
            index = max((row['index'] for row in results[target]), default=-1)+1
            if index >= max_attempts:
                continue
            seed = seed_start+target_index*10000+index
            path = output/target/f'episode_{index:04d}'
            error = None
            try:
                attempt(target, path, seed)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                path.mkdir(exist_ok=True)
                (path/'attempt_error.json').write_text(json.dumps({'seed': seed, 'error': error}, ensure_ascii=False))
            quality = review_attempt(path, target=target, seed=seed, configuration=configuration)
            results[target].append({'path': str(path.resolve()), 'index': index, 'seed': seed,
                                    'accepted': quality['accepted'], 'errors': quality['errors'],
                                    'execution_error': error})
            new_attempts += 1
            attempted = True
            report = publish()
            print('批量进度', json.dumps(report['by_object'], ensure_ascii=False), flush=True)
        if not attempted:
            break
    return report


def main():
    from tools.medication_sorting.collect_episodes import collect
    from tools.medication_sorting.randomization import PROFILES, profile_config
    from tools.medication_sorting.rendered_rgb import RenderedRGB
    from tools.medication_sorting.robot_control import FixedBaseRobot
    from tools.medication_sorting.scaled_layout import resolve_experiment_layout
    from tools.medication_sorting.task_specs import EXPERT_PROFILES, TARGETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-target', type=int, default=40)
    parser.add_argument('--max-attempts', type=int, default=120)
    parser.add_argument('--max-new-attempts', type=int)
    parser.add_argument('--seed-start', type=int, default=2026101500)
    parser.add_argument('--profile', choices=EXPERT_PROFILES, default='randomized_tabletop_70')
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--randomization-profile', choices=PROFILES, default='tabletop_v1')
    args = parser.parse_args()
    layout = resolve_experiment_layout(args.layout, profile=args.profile)
    configuration = {'expert_profile': args.profile, 'layout': str(layout.resolve()),
                     'layout_sha256': hashlib.sha256(layout.read_bytes()).hexdigest(),
                     'randomization_profile': args.randomization_profile,
                     'randomization_config': profile_config(args.randomization_profile), 'control_hz': 5}
    robot = FixedBaseRobot(frame_skip=200)
    rgb = None
    try:
        rgb = RenderedRGB(robot)
        def attempt(target, output, seed):
            return collect(robot, rgb, output, target=target, seed=seed, randomize=True,
                           profile=args.profile, layout=layout,
                           randomization_profile=args.randomization_profile)
        report = collect_balanced(args.output, attempt=attempt, targets=list(TARGETS),
                                  per_target=args.per_target, max_attempts=args.max_attempts,
                                  seed_start=args.seed_start, configuration=configuration,
                                  max_new_attempts=args.max_new_attempts)
    finally:
        if rgb is not None:
            rgb.close()
        robot.disconnect()
    print(json.dumps({key: report[key] for key in ('complete', 'accepted', 'attempts', 'by_object')}, ensure_ascii=False, indent=2))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
