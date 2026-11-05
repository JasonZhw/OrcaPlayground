"""Resume the medication ACT baseline from 20k to 40k, without changing LeRobot."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from examples.euler.medication_sorting.act.run import verify_source, write_json

SOURCE = Path('/root/gpufree-data/training/med20_act_20261104/act20k_v1/checkpoints/020000')
DATASET = Path('/root/gpufree-data/datasets/lerobot/local/g1_med_sort_v2')
ROWS = Path('/root/gpufree-data/training/med20_pi05_continue_20261105/eval_20000/rows.json')
START, END = 20000, 40000
RESERVE = 1536 * 1024**2


def validate_resume(step, optimizer_steps, expected):
    if step != expected or not optimizer_steps or any(s != expected for s in optimizer_steps):
        raise ValueError(f'Resume mismatch: step={step}, optimizer counters={set(optimizer_steps)}')


def validate_space(free, checkpoint_bytes, saves, reserve=RESERVE):
    required = checkpoint_bytes * saves + reserve
    if free < required:
        raise RuntimeError(f'Insufficient disk: free={free}, required={required}')


def action_metrics(pred, target):
    import numpy as np
    if pred.shape != (50, 16) or target.shape != (50, 16):
        raise ValueError('Expected action chunks of shape (50, 16)')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite action')
    error = np.abs(pred - target)
    arms = [i for i in range(16) if i not in (7, 15)]
    return {f'{name}_mae_{n}': float(error[:n, columns].mean())
            for name, columns in [('arm', arms), ('closure', [7, 15])]
            for n in (1, 50)}


def manifest(path):
    result = {}
    for p in sorted(path.rglob('*')):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            result[str(p.relative_to(path))] = {'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
    return result


def make_resume_config(root):
    from lerobot.configs.train import TrainPipelineConfig
    cfg = TrainPipelineConfig.from_pretrained(SOURCE / 'pretrained_model', local_files_only=True)
    cfg.resume = True
    cfg.output_dir = root / 'act40k_v1'
    cfg.job_name = 'act40k_v1'
    cfg.steps = END
    cfg.save_freq = 10000
    cfg.eval_freq = 0
    cfg.wandb.enable = False
    sys.argv = [sys.argv[0], f'--config_path={SOURCE / "pretrained_model/train_config.json"}']
    cfg.validate()
    assert cfg.checkpoint_path.resolve() == SOURCE.resolve()
    assert Path(cfg.dataset.root).resolve() == DATASET.resolve()
    assert cfg.batch_size == 8 and cfg.policy.chunk_size == cfg.policy.n_action_steps == 50
    assert cfg.optimizer.lr == 1e-5 and cfg.optimizer.grad_clip_norm == 10
    assert not cfg.policy.use_amp and not cfg.dataset.image_transforms.enable
    return cfg


def train(root):
    import torch
    import lerobot.scripts.train as official
    from lerobot.common.utils.utils import init_logging
    cfg = make_resume_config(root)
    assert not cfg.output_dir.exists(), 'Never overwrite or implicitly restart a continuation'
    source_hashes = verify_source(DATASET)
    write_json(root / 'dataset_manifest.json', source_hashes)
    checkpoint_bytes = sum(p.stat().st_size for p in SOURCE.rglob('*') if p.is_file())
    validate_space(shutil.disk_usage(root).free, checkpoint_bytes, 2)
    original_load, original_update = official.load_training_state, official.update_policy
    original_make_policy = official.make_policy
    restored = False
    count = 0
    started = time.monotonic()

    def checked_make_policy(*args, **kwargs):
        from safetensors.torch import load_file
        policy = original_make_policy(*args, **kwargs)
        saved = load_file(SOURCE / 'pretrained_model/model.safetensors')
        state = policy.state_dict()
        assert state.keys() == saved.keys()
        assert all(torch.equal(state[key].cpu(), value) for key, value in saved.items()), 'Weights or normalization changed during load'
        write_json(root / 'policy_restore_verified.json', {'passed': True, 'tensors': len(saved),
                   'includes_normalization_buffers': True, 'time_unix': time.time()})
        return policy

    def checked_load(*args, **kwargs):
        nonlocal restored
        step, optimizer, scheduler = original_load(*args, **kwargs)
        counters = [int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
        validate_resume(step, counters, START)
        assert all(group['lr'] == 1e-5 for group in optimizer.param_groups)
        write_json(root / 'resume_verified.json', {
            'step': step, 'optimizer_steps': sorted(set(counters)),
            'optimizer_states': len(counters), 'passed': True, 'time_unix': time.time()})
        restored = True
        return step, optimizer, scheduler

    def checked_update(*args, **kwargs):
        nonlocal count
        assert restored, 'Refuse updates before verified optimizer restoration'
        if count % 50 == 0:
            # No checkpoint deletion; stop before an update if disk reserve is breached.
            validate_space(shutil.disk_usage(root).free, 0, 0)
        absolute = START + count + 1
        if absolute in (30000, 40000):
            validate_space(shutil.disk_usage(root).free, checkpoint_bytes, 1)
        metrics, detail = original_update(*args, **kwargs)
        count += 1
        values = {'loss': float(metrics.loss.val), 'grad_norm': float(metrics.grad_norm.val),
                  **{key: float(value) for key, value in detail.items()}}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f'Non-finite update at {absolute}')
        if count == 1 or count % 10 == 0:
            row = {'step': absolute, 'new_updates': count, **values,
                   'elapsed_seconds': time.monotonic() - started, 'time_unix': time.time(),
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            with (root / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if count == 1:
                write_json(root / 'first_update.json', row)
        return metrics, detail

    official.load_training_state, official.update_policy = checked_load, checked_update
    official.make_policy = checked_make_policy
    init_logging()
    official.train(cfg)
    assert count == END - START
    write_json(root / 'training_result.json', {'passed': True, 'new_updates': count,
               'total_updates': END, 'elapsed_seconds': time.monotonic() - started,
               'time_unix': time.time()})


def evaluate(root, checkpoint):
    import numpy as np
    import torch
    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    cfg = make_resume_config(root)
    dataset = make_dataset(cfg)
    indices = json.loads((root / 'eval_indices.json').read_text())
    policy = ACTPolicy.from_pretrained(checkpoint / 'pretrained_model', local_files_only=True).cuda().eval()
    assert all(torch.isfinite(value).all() for value in policy.state_dict().values())
    report_dir = root / f'eval_{int(checkpoint.name)}'
    report_dir.mkdir(exist_ok=False)
    predictions, targets, rows = [], [], []
    for index in indices:
        sample = dataset[index]
        # These fixed indices are away from the episode tail; padding must never enter MAE.
        assert not sample['action_is_pad'].any()
        batch = {k: v.unsqueeze(0).cuda() for k, v in sample.items()
                 if k.startswith('observation.') and isinstance(v, torch.Tensor)}
        policy.reset()
        with torch.inference_mode():
            pred = torch.stack([policy.select_action(batch)[0] for _ in range(50)]).cpu().numpy()
        target = sample['action'].numpy()
        row = {'index': index, **action_metrics(pred, target),
               'closure_pred_first': pred[0, [7, 15]].tolist(),
               'closure_target_first': target[0, [7, 15]].tolist()}
        rows.append(row)
        predictions.append(pred)
        targets.append(target)
    pred, target = np.stack(predictions), np.stack(targets)
    # Separate stable-open, stable-closed, opening and closing labels per hand.
    phases = {}
    for hand, column in [('left', 7), ('right', 15)]:
        first, delta = target[:, 0, column], target[:, -1, column] - target[:, 0, column]
        masks = {'opening': delta < -.05, 'closing': delta > .05,
                 'open': (first < .05) & (np.abs(delta) <= .05),
                 'closed': (first > .95) & (np.abs(delta) <= .05)}
        for phase, mask in masks.items():
            phases[f'{hand}_{phase}'] = {'count': int(mask.sum()),
                'closure_mae_1': float(np.abs(pred[mask, 0, column] - first[mask]).mean()) if mask.any() else None,
                'closure_mae_50': float(np.abs(pred[mask, :, column] - target[mask, :, column]).mean()) if mask.any() else None}
    np.savez_compressed(report_dir / 'predictions.npz', predictions=pred, targets=target, indices=indices)
    write_json(report_dir / 'rows.json', rows)
    write_json(report_dir / 'result.json', {'passed': True, 'finite': True, 'shape': list(pred.shape),
        'checkpoint': str(checkpoint), 'samples': len(rows),
        'metrics': {key: float(np.mean([row[key] for row in rows])) for key in action_metrics(pred[0], target[0])},
        'phases': phases, 'independent_validation': False, 'simulation_tested': False,
        'time_unix': time.time()})


def pipeline(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'process.json').exists(), 'A run already exists; diagnose instead of restarting'
        started = time.time()
        write_json(root / 'process.json', {'pid': os.getpid(), 'started_unix': started})
        code = 1
        try:
            hashes = manifest(SOURCE)
            validate_space(shutil.disk_usage(root).free, sum(x['bytes'] for x in hashes.values()), 2)
            write_json(root / 'source_manifest.json', hashes)
            write_json(root / 'authorization.json', {'request': '继续 训练act', 'source_step': START,
                'target_step': END, 'save_steps': [30000, 40000], 'dataset_unchanged': True,
                'controller_unchanged': True, 'time_unix': started})
            write_json(root / 'eval_indices.json', sorted({row['index'] for row in json.loads(ROWS.read_text())}))
            stages = [('eval_20000', 'eval', SOURCE), ('training', 'train', None)]
            stages += [(f'eval_{step}', 'eval', root / f'act40k_v1/checkpoints/{step:06d}') for step in (30000, 40000)]
            for name, mode, checkpoint in stages:
                cmd = [sys.executable, '-m', 'examples.euler.medication_sorting.act.continue_train', mode,
                       '--root', str(root)]
                if checkpoint:
                    cmd += ['--checkpoint', str(checkpoint)]
                with (root / f'{name}.log').open('x') as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    write_json(root / f'{name}.process.json', {'pid': child.pid, 'parent_pid': os.getpid(), 'started_unix': time.time()})
                    write_json(root / 'status.json', {'phase': name, 'child_pid': child.pid, 'time_unix': time.time()})
                    while child.poll() is None:
                        free = shutil.disk_usage(root).free
                        with (root / 'resources.jsonl').open('a') as stream:
                            stream.write(json.dumps({'time_unix': time.time(), 'free_bytes': free, 'phase': name}) + '\n')
                        if free < RESERVE:
                            os.killpg(child.pid, 15)
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, 9)
                                child.wait()
                            (root / f'{name}.exitcode').write_text(str(child.returncode) + '\n')
                            raise RuntimeError('Disk reserve breached; child stopped, all artifacts retained')
                        time.sleep(10)
                    (root / f'{name}.exitcode').write_text(str(child.returncode) + '\n')
                    if child.returncode != 0:
                        raise RuntimeError(f'{name} failed: exit {child.returncode}')
                if mode == 'eval':
                    result = json.loads((root / name / 'result.json').read_text())
                    assert result['passed'] and result['finite'] and result['samples'] == 45
            assert manifest(SOURCE) == hashes, 'Original checkpoint changed'
            write_json(root / 'source_unchanged.json', {'passed': True, 'time_unix': time.time()})
            for step in (30000, 40000):
                saved = root / f'act40k_v1/checkpoints/{step:06d}/training_state/training_step.json'
                assert json.loads(saved.read_text())['step'] == step
            write_json(root / 'status.json', {'phase': 'complete', 'total_updates': END,
                'elapsed_seconds': time.time() - started, 'simulation_tested': False, 'time_unix': time.time()})
            code = 0
        except Exception as exc:
            write_json(root / 'status.json', {'phase': 'blocked', 'error': str(exc), 'time_unix': time.time()})
            raise
        finally:
            (root / 'pipeline.exitcode').write_text(str(code) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pipeline', 'train', 'eval', 'preflight'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    if args.mode == 'pipeline':
        pipeline(args.root)
    elif args.mode == 'train':
        train(args.root)
    elif args.mode == 'eval':
        evaluate(args.root, args.checkpoint)
    else:
        cfg = make_resume_config(args.root)
        print(json.dumps({'resume': cfg.resume, 'source': str(cfg.checkpoint_path),
                          'target_steps': cfg.steps, 'output': str(cfg.output_dir)}))


if __name__ == '__main__':
    main()
