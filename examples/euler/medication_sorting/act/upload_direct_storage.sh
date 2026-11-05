#!/usr/bin/env bash
# User-run alternative: API through SSH, ordinary LFS object uploads direct.
# Stop the previous uploader and wait for its exit before running this file.
set -euo pipefail
export HTTPS_PROXY=http://127.0.0.1:18977
export HTTP_PROXY=http://127.0.0.1:18977
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export NO_PROXY=localhost,127.0.0.1,.amazonaws.com,cas-bridge.xethub.hf.co,cdn-lfs.hf.co,cdn-lfs-us-1.hf.co
export no_proxy="$NO_PROXY"
cd /root/gpufree-data/OrcaPlayground
exec /root/gpufree-data/envs/med_pi05/bin/python -u \
  -m examples.euler.medication_sorting.act.upload_dataset \
  --folder /root/gpufree-data/training/med20_act_20261104/hf_upload/g1_med_sort \
  --execute
