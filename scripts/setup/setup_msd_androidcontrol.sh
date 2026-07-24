#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_name="${MSD_CONDA_ENV:-msd-androidcontrol}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
  conda create -n "${env_name}" python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate "${env_name}"
python -m pip install --upgrade pip
python -m pip install -r "${repo_root}/requirements-msd-qwen2vl.txt"
python -m pip install -e "${repo_root}"
python -m pip check

python - <<'PY'
import torch
import transformers
import guiaccel
from guiaccel.model.msd.core.ea_model import EaModel

print(
    {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "msd_runtime": EaModel.__module__,
        "guiaccel": guiaccel.__file__,
    }
)
PY
