# pi0.5 cloud pilot: G1 tray transfer

This directory is a standalone adapter and launcher for a single CUDA GPU with
48 GB VRAM. Start with a batch of four and ten training steps, then run a longer
pilot after the data and checkpoint checks pass. GPU training and inference have
not been executed locally; 48 GB fit and throughput still need the cloud smoke
run. Eight demonstrations are sufficient to check the training path and attempt
an overfit baseline, not to establish generalization.

The source contract is pinned to
[OpenPI `215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)
and its locked [LeRobot `0cf8648`](https://github.com/huggingface/lerobot/tree/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5)
dependency (dataset format v2.1). Use the JAX trainer: this OpenPI revision does
not support LoRA in its PyTorch trainer. The launcher imports public functions
from the pinned checkout; it does not patch OpenPI's model or config registry.

## Data and model contract

| Item | Contract |
| --- | --- |
| Dataset | `local/g1_lightblue_box_tray_v1`, LeRobot v2.1, 50 fps |
| State | `observation.state`: measured right arm7 + measured gripper8 = 15 values |
| Actions | `action`: absolute right arm7 targets + gripper `a,d` = 9 values |
| Gripper reconstruction | `[0,0,a,d,0,0,-a,d]`; retain the dataset joint ordering |
| Cameras | `observation.images.head` and `observation.images.right_wrist`, RGB 224×224 |
| Camera mapping | head → `base_0_rgb`; right wrist → `right_wrist_0_rgb` |
| Missing camera | `left_wrist_0_rgb` is zeros with mask `False` |
| Language | Original English task text from each episode's `task_index`; all three paraphrases remain |
| Model | `Pi0Config(pi05=True)`, action dimension 32, horizon 50 |
| Adaptation | `gemma_2b_lora` + `gemma_300m_lora`, model freeze filter, EMA off |
| Initialization | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Normalization | Own state15/action9 quantiles; model transforms subsequently pad both to 32 |

`TrayInputs` accepts the actual LeRobot reader's CHW float images in `[0,1]`,
and HWC `uint8` RGB images for inference. It validates the unbatched state and
50-step action shapes. `TrayOutputs` returns all nine unnormalized values.
Neither adapter performs delta conversion: gripper state8 and gripper action2
are different representations, so subtracting a state prefix would be wrong.

At 50 Hz, a 50-step chunk describes one second. This is not a claim of 50 Hz
model inference. A later simulator controller must expand `a,d`, apply its
normal joint limits, execute a selected chunk prefix, and replan. This kit's
inference command only saves an action array; it does not control the simulator.

The converted data has no offline augmentation. The pinned OpenPI JAX training
path still applies its built-in image augmentation: head-image crop/rotation
and color jitter. The stock training config has no augmentation toggle. This
pilot preserves that upstream behavior; it is not an augmentation-free model
baseline. Color-jitter sensitivity is a relevant later ablation because the
instruction refers to the light blue box.

## Prepare a separate cloud environment

Use an Ubuntu CUDA host with a compatible NVIDIA driver. These commands create
an environment separate from the Orca simulation environment. Install `uv` on
the host first if needed. Leave the pinned dependency lock unchanged.

```bash
export OPENPI_ROOT=/workspace/openpi
git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
cd "$OPENPI_ROOT"
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git submodule update --init --recursive
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps -e .
```

Copy this whole `pi05_training` directory to `/workspace/pi05_training`, and
place the converted dataset as follows. Copying only `data/` is insufficient:
the reader also needs the metadata and both camera video trees.

```text
/workspace/lerobot/local/g1_lightblue_box_tray_v1/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  data/...
  videos/...
  provenance/...
```

Set paths and build a reusable Bash command array. `--data-home` is the parent
of `local/`, not the dataset directory itself. The launcher sets
`HF_LEROBOT_HOME` before importing LeRobot. All output directories are explicit.

```bash
export TRAY_KIT=/workspace/pi05_training
export HF_LEROBOT_HOME=/workspace/lerobot
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export OPENPI_DATA_HOME=/workspace/openpi-cache
RUN=("$OPENPI_ROOT/.venv/bin/python" "$TRAY_KIT/run.py"
     --openpi-dir "$OPENPI_ROOT"
     --data-home "$HF_LEROBOT_HOME"
     --assets-dir /workspace/tray-assets
     --checkpoint-dir /workspace/tray-checkpoints
     --batch-size 4 --num-workers 0)
```

## Check the loader, compute statistics, then train

The first command reads one actual OpenPI data-loader batch and validates
images, masks, state padding, action padding, and language tokenization. It
loads no model weights, but the tokenizer may download its SentencePiece file
on first use. Run with network access or a prepopulated OpenPI cache.

```bash
JAX_PLATFORMS=cpu "${RUN[@]}" data-smoke --skip-norm-stats
JAX_PLATFORMS=cpu "${RUN[@]}" norm
JAX_PLATFORMS=cpu "${RUN[@]}" data-smoke
```

Expected model shapes are state `(4,32)`, actions `(4,50,32)`, three images of
`(4,224,224,3)`, and language tokens `(4,200)`. The raw observations and action
targets remain 15 and 9 dimensions before padding. Normalization processes
every frame using the upstream statistics helper with batch size one, avoiding
its incomplete-batch dropping. Statistics are saved to
`/workspace/tray-assets/pi05_g1_tray_lora/local/g1_lightblue_box_tray_v1/norm_stats.json`.
Existing statistics are preserved; select a new assets directory to recompute.

Start a new, non-overwriting experiment for each run:

```bash
"${RUN[@]}" train --exp-name smoke10_v1 --steps 10
"${RUN[@]}" infer \
  --checkpoint /workspace/tray-checkpoints/pi05_g1_tray_lora/smoke10_v1/9 \
  --index 0 --output /workspace/tray-predictions/smoke10_frame0.npy
"${RUN[@]}" train --exp-name pilot1000_v1 --steps 1000
```

The trainer downloads the pi0.5 base weights on first use. The ten-step run is
a compile, finite-loss, and checkpoint test. Its first steps are still within
the configured 100-step learning-rate warmup. Check loss values and measured
VRAM usage before starting the 1,000-step pilot. The pilot uses a peak learning
rate of `5e-5`, then cosine decay toward `5e-6` over 10,000 steps; these are
initial settings, not a tuned result. Checkpoints save every 100 steps and at
the final step. Upstream names checkpoint directories by its zero-based loop
index, so a completed ten-step run writes `9/` and a 1,000-step run writes
`999/`. W&B is disabled.

Use `--resume` only with the same experiment name and a larger total step
target after interruption. There is no overwrite flag. The offline inference
wrapper uses normalization saved inside the trained checkpoint and refuses to
overwrite an existing prediction file. Finite `(50,9)` output establishes the
inference interface only; simulator rollouts on unseen initial conditions are
still required to assess task success.

## Local adapter check

The adapter tests need only NumPy. They do not import OpenPI or download weights:

```bash
python -m unittest discover -s /path/to/pi05_training -p test_adapter.py -v
```
