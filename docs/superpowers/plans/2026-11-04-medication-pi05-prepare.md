# Medication π0.5 preparation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline, task by task.

**Goal:** Prepare the verified 20-episode medication dataset for π0.5 LoRA; prove real data loading, normalization, a 10-step smoke train and checkpoint reload without simulator actuation or upload.

**Architecture:** A small NumPy adapter next to the medication example connects state30/action16 and three native cameras to pinned OpenPI. An external config/runner calls public upstream training APIs. Independent cloud source, environment, assets, checkpoints and logs live on the data disk.

**Tech Stack:** OpenPI 215abfb217dbac7d5f1273282331b9b1866c0479, LeRobot 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5, JAX LoRA, Python3.12, L40.

**Spec:** ../../medication_pi05_training_inference.md; user approved preparation following this description.

## Global constraints

- Preserve raw and converted data, original tasks, three 640×480 cameras, 50Hz, state30/action16 and medication_closure_v1.
- Absolute joint actions; no generic DeltaActions. Model width32 and horizon50; remove padding on output.
- No HF upload, simulator controls, new collection or long training run in this preparation task.
- Isolated JAX environment; do not mutate orca or med_lerobot. Keep official image transforms/augmentation and inspect previews.
- Reuse existing feature worktree; no unrelated changes, commits or publishing required.

## Review focus

- Swapped camera/joint identities must fail metadata checks, not silently train.
- Float CHW and uint8 HWC must yield identical RGB; invalid range/NaN rejected.
- State/action differ in length; no accidental subtraction or right-arm misalignment.
- Norm stats must cover all frames and use actual unpadded dimensions; no foreign stats.
- Launch must reject existing experiment overwrite and wrong versions; preserve logs on failure.

### Task 1: Adapter and contract checks

Files: examples/euler/medication_sorting/pi05/{__init__,adapter}.py; tests/test_medication_pi05.py.
- [x] Write tests for three unique camera colors, exact unchanged state/action, invalid dimensions/ranges/tasks and output slicing.
- [x] Run tests to see missing adapter failure.
- [x] Implement MedicationInputs/Outputs and validate_info(info, validation); preserve absolute action semantics.
- [x] Run focused tests and medication regression suite; record unrelated baseline failures separately.

### Task 2: Isolated environment and pinned config

Files: pi05/config.py, pi05/run.py; cloud /root/gpufree-data/training/med20_20261104.
- [ ] Fetch fixed official source and dependencies with process/exit/log tracking; verify JAX GPU and package imports.
- [x] Build make_config using pi05_base, gemma_2b_lora/gemma_300m_lora, freeze filter, EMA off, initial batch2, 10 steps, horizon50.
- [x] Validate dataset names, resolution, version and success report before constructing loaders.
- [ ] Read real batch before/after own quantile stats; save input/augmentation preview and sample provenance.

### Task 3: Bounded smoke and handoff

- [ ] Check disk budget for base weights and checkpoint; download official base into data disk with durable logs.
- [ ] Run 10 steps in a persistent process; record trainable parameters, finite losses, memory, exact configuration.
- [ ] Reload resulting checkpoint and infer one real observation to finite (50,16) output without actuation.
- [ ] Independent review of added training code, fix material findings and verify; document tested commands/results and remaining formal-training decisions.

## Execution ledger

- Ruling: user already selected π0.5 LoRA and approved the documented preparation sequence. Continue without repeating model/design questions.
- Ruling: start batch2 to accommodate three camera views and compilation, measure before increasing. Cost: lower smoke throughput, no data change.
- Initial state: conversion verified complete 20/219594; cloud only med_lerobot venv, no OpenPI source/base weights found in standard cache; L40 46,068MiB, 4,557MiB used, disk ~82.8GiB free.

- Adapter/regression verification: 152 passed, 1 skipped, 27 subtests passed; scoped Ruff passed. Full repository test collection is blocked by the existing joystick example `examples/embodied/g1/utils/test_xbox.py` calling `sys.exit()`; no unrelated changes were made.
- Independent review found three preparation safeguards to add: preserve the original launch audit on resume, verify source files beyond uv.lock, and bind normalization statistics to data hashes and dimensions. Five failing regression cases were added, all now pass.
- Official base download is generation-pinned, resumable, and tracked independently. A bounded network probe showed local-to-cloud relay slower than direct GCS; temporary relay was removed. Direct official GCS endpoint and 8 object downloads replace the slower JSON media endpoint; existing partial bytes are retained.
- Initial environment installation failed on GitHub Git HTTP/2 during LeRobot fetch. Recovering by importing the exact shallow official commit into the dependency cache; not changing the frozen lock or the simulation/conversion environments.
- GPU/data smoke/norm/10-step train/reload are NOT yet verified. Do not mark this plan complete from unit tests or a successful package download.

- Latest user authorization supersedes the earlier preparation-only limit: a simple successful smoke test should automatically launch the first long training run. Added a separate gated supervisor for 10,000 steps, batch2, experiment med20_lora_10k_v1. No HF upload or robot actuation authorized by this update. Waiting for downloads is not reported as training started.
