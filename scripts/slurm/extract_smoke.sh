#!/bin/bash
#SBATCH --job-name=reghead-extract-smoke
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_smoke/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_smoke/slurm-%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────
cd /dkucc/home/rw335/GUIAccel
source .env 2>/dev/null || true

PYTHON=/dkucc/home/rw335/.conda/envs/skillreuse-fa2/bin/python3

echo "=== Environment ==="
echo "Python: $PYTHON"
echo "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU 0: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "GPU free mem: $($PYTHON -c 'import torch; print(f"{torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB") if torch.cuda.is_available() else print("N/A")')"
echo "==================="

# ── Run extraction (smoke test: 3 episodes) ─────────────────────────────
CUDA_VISIBLE_DEVICES=0 $PYTHON experiments/regression_head.py \
    --mode extract \
    --output-dir outputs/regression_head \
    --split train \
    --episode-limit 3 \
    --device 0 \
    --dtype bfloat16 \
    --max-new-tokens 512

echo "=== Extraction smoke test complete ==="
