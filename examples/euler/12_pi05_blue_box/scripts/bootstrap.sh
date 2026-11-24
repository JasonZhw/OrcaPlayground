#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v conda >/dev/null || { echo "Install Miniconda first."; exit 1; }
conda env create -f environment.yml
conda run -n orca python -m pip install \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
if [[ "${1:-}" == "--training-only" ]]; then
  conda run -n orca python -m pip install -r requirements-training.txt
else
  conda run -n orca python -m pip install -r requirements.txt -r requirements-simulation.txt
  conda run -n orca python -m pip install --no-build-isolation \
    "nvidia-curobo[cu12-torch] @ git+https://github.com/NVlabs/curobo.git@78fd485fa82d9b9a063fb4985e371814587e666a"
fi
conda run -n orca python -m pip check
echo "Environment ready. Run: conda activate orca"
echo "Then: python reproduce.py check"
