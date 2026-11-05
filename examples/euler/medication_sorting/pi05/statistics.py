"""Numeric-only norm pass, with the same episode-clamped action chunks as reader.

No image transformations affect state/actions. Reading parquet directly avoids
decoding three videos for every frame just to discard those pixels for statistics.
Real reader comparisons at episode boundaries guard this optimization.
"""

import hashlib
import json

import numpy as np

from .adapter import ACTION_DIM, ACTION_HORIZON, STATE_DIM


def action_windows(actions, start, stop, horizon=ACTION_HORIZON):
    if actions.ndim != 2 or not len(actions) or not 0 <= start < stop <= len(actions) or horizon < 1:
        raise ValueError("Invalid episode action window")
    indices = np.minimum(np.arange(start, stop)[:, None] + np.arange(horizon), len(actions) - 1)
    return actions[indices]


def compute_stats(config, data_root):
    import pyarrow.parquet as pq
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import data_loader

    output = config.assets_dirs / config.data.repo_id
    if (output / "norm_stats.json").exists():
        raise FileExistsError(f"Refusing to overwrite statistics: {output}")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, ACTION_HORIZON, config.model)
    info = json.loads((data_root / "meta/info.json").read_text())
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    rows, comparisons, sources = 0, 0, []
    for episode in range(info["total_episodes"]):
        path = data_root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        table = pq.read_table(path)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.shape != (len(actions), STATE_DIM) or actions.shape[1:] != (ACTION_DIM,):
            raise ValueError(f"Bad numeric shapes in {path}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite numeric data: {path}")
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(len(actions)))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(len(actions), episode))
        for index in sorted({0, len(actions) // 2, max(0, len(actions) - 49), len(actions) - 1}):
            sample = dataset[rows + index]
            np.testing.assert_array_equal(sample["observation.state"], states[index])
            np.testing.assert_array_equal(sample["action"], action_windows(actions, index, index + 1)[0])
            comparisons += 1
        # Accumulate in float64 to avoid cancellation in nearly constant joints.
        for start in range(0, len(actions), 128):
            stop = min(start + 128, len(actions))
            stats["state"].update(states[start:stop].astype(np.float64))
            stats["actions"].update(action_windows(actions, start, stop).astype(np.float64))
        rows += len(actions)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"NORM episode {episode}: {len(actions)} rows; total {rows}", flush=True)
    if rows != info["total_frames"] or rows != len(dataset):
        raise ValueError("Norm statistics did not cover the entire dataset")
    result = {key: acc.get_statistics() for key, acc in stats.items()}
    # Check the public normalization pair against every numeric row.
    roundtrip_error = {"state": 0.0, "actions": 0.0}
    forward = transforms.Normalize(result, use_quantiles=True)
    reverse = transforms.Unnormalize(result, use_quantiles=True)
    for source in sources:
        table = pq.read_table(source["path"], columns=["observation.state", "action"])
        original = {
            "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            "actions": np.asarray(table["action"].to_pylist(), dtype=np.float32),
        }
        restored = reverse(forward(original))
        for key in original:
            error = float(np.max(np.abs(original[key] - restored[key])))
            roundtrip_error[key] = max(roundtrip_error[key], error)
            np.testing.assert_allclose(restored[key], original[key], rtol=1e-6, atol=1e-6)
    normalize.save(output, result)
    report = dict(
        stats_sha256=hashlib.sha256((output / "norm_stats.json").read_bytes()).hexdigest(),
        frames=rows,
        episodes=info["total_episodes"],
        state_dim=30,
        action_dim=16,
        action_horizon=50,
        state_vectors=rows,
        action_vectors=rows * 50,
        official_reader_comparisons=comparisons,
        roundtrip_max_abs_error=roundtrip_error,
        sources=sources,
        method="OpenPI RunningStats; every state once and every episode-clamped 50-step window",
    )
    (output / "norm_provenance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))
