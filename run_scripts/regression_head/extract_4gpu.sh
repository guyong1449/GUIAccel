#!/usr/bin/env bash
#SBATCH --job-name=reghead-extract
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00

set -euo pipefail

# --------------------------------------------------------------------------- #
# Coordinate Regression Head — Phase 1: Hidden State Extraction (4 GPU)
#
# Each GPU loads its own Qwen3-VL-8B copy and processes 1/4 of the episodes.
# Output: outputs/regression_head/extracted/<split>_hidden_states.pt
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true

# Timestamp for unique output directory
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/outputs/regression_head/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Redirect SLURM logs alongside output
SLURM_LOG_DIR="${OUTPUT_DIR}"
exec > >(tee -a "${SLURM_LOG_DIR}/extract.log") 2>&1

echo "============================================"
echo "GUIAccel — Regression Head Extraction"
echo "============================================"
echo "Timestamp:    ${TIMESTAMP}"
echo "Repo root:    ${REPO_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Python:       ${PYTHON}"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo ""

# Environment check
"${PYTHON}" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, '
          f'{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"
SPLIT="${EXTRACT_SPLIT:-train}"
EPISODE_LIMIT="${EXTRACT_EPISODE_LIMIT:-}"
THINKING_MODE="${EXTRACT_THINKING_MODE:-template}"
EXTRACT_POINT="${EXTRACT_EXTRACT_POINT:-action}"

echo ""
echo "Config: num_gpus=${NUM_GPUS}, split=${SPLIT}, episode_limit=${EPISODE_LIMIT:-all}"
echo "        thinking_mode=${THINKING_MODE}, extract_point=${EXTRACT_POINT}"
echo ""

EXTRACT_ARGS=(
    "${PYTHON}" "${REPO_ROOT}/experiments/regression_head.py"
    --mode extract
    --output-dir "${OUTPUT_DIR}"
    --split "${SPLIT}"
    --num-gpus "${NUM_GPUS}"
    --dtype bfloat16
    --max-new-tokens 512
    --thinking-mode "${THINKING_MODE}"
    --extract-point "${EXTRACT_POINT}"
)

if [[ -n "${EPISODE_LIMIT}" ]]; then
    EXTRACT_ARGS+=(--episode-limit "${EPISODE_LIMIT}")
fi

echo "Command: ${EXTRACT_ARGS[*]}"
echo ""

"${EXTRACT_ARGS[@]}"

echo ""
echo "============================================"
echo "Extraction complete at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
