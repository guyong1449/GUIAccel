#!/bin/bash
#SBATCH --job-name=reghead-extract-smoke
#SBATCH --partition=h20-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/outputs/slurm/%j_extract_smoke.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/outputs/slurm/%j_extract_smoke.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────
cd /dkucc/home/rw335/GUIAccel
source .env 2>/dev/null || true

# Activate conda
if [[ -d "${GUIACCEL_CONDA_PREFIX:-}" ]]; then
    source "${GUIACCEL_CONDA_PREFIX}/bin/activate"
elif [[ -d /dkucc/home/rw335/.conda/envs/skillreuse-fa2 ]]; then
    source /dkucc/home/rw335/.conda/envs/skillreuse-fa2/bin/activate
fi

echo "=== Environment ==="
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU 0: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "==================="

# ── Run extraction (smoke test: 3 episodes) ─────────────────────────────
CUDA_VISIBLE_DEVICES=0 python3 experiments/regression_head.py \
    --mode extract \
    --output-dir outputs/regression_head \
    --split train \
    --episode-limit 3 \
    --device 0 \
    --dtype bfloat16 \
    --max-new-tokens 512

echo "=== Extraction smoke test complete ==="
