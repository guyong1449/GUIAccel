#!/usr/bin/env bash
#SBATCH --job-name=reghead-train
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/train_cpu/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/train_cpu/slurm-%j.err

set -euo pipefail

# --------------------------------------------------------------------------- #
# Coordinate Regression Head — Phase 2: MLP Training (CPU / common partition)
#
# Tiny MLP on pre-extracted (hidden_state, gt_coord) pairs — no backbone, no GPU.
# Output: <OUTPUT_DIR>/trained/coord_head_best.pth
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

# Default: Phase-1 full extract run (46,581 samples)
EXTRACT_SRC="${EXTRACT_SRC:-${REPO_ROOT}/outputs/regression_head/20260723_050425/extracted}"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true

mkdir -p "${REPO_ROOT}/logs/regression_head/train_cpu"

# Use allocated CPUs; keep training off any accidental GPU.
N_CPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${N_CPUS}"
export MKL_NUM_THREADS="${N_CPUS}"
export OPENBLAS_NUM_THREADS="${N_CPUS}"
export NUMEXPR_NUM_THREADS="${N_CPUS}"
export CUDA_VISIBLE_DEVICES=""

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/outputs/regression_head/train_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Point train code at existing extract artifacts without copying (~764 MB)
ln -sfn "${EXTRACT_SRC}" "${OUTPUT_DIR}/extracted"

exec > >(tee -a "${OUTPUT_DIR}/train.log") 2>&1

echo "============================================"
echo "GUIAccel — Regression Head Training (CPU)"
echo "============================================"
echo "Timestamp:    ${TIMESTAMP}"
echo "Repo root:    ${REPO_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Extract src:  ${EXTRACT_SRC}"
echo "Python:       ${PYTHON}"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "CPUs:         ${N_CPUS} (OMP/MKL threads)"
echo "Partition:    ${SLURM_JOB_PARTITION:-common}"
echo ""

"${PYTHON}" -c "
import os
import torch
n = int(os.environ.get('OMP_NUM_THREADS', '1'))
torch.set_num_threads(n)
torch.set_num_interop_threads(max(1, min(4, n)))
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'torch.get_num_threads: {torch.get_num_threads()}')
print(f'torch.get_num_interop_threads: {torch.get_num_interop_threads()}')
"

HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-256}"
EPOCHS="${TRAIN_EPOCHS:-50}"
LR="${TRAIN_LR:-3e-4}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
DROPOUT="${TRAIN_DROPOUT:-0.1}"
PATIENCE="${TRAIN_PATIENCE:-10}"
EXTRACT_POINT="${TRAIN_EXTRACT_POINT:-action}"
THINKING_MODE="${TRAIN_THINKING_MODE:-template}"

echo ""
echo "Config: hidden_dim=${HIDDEN_DIM} epochs=${EPOCHS} lr=${LR} batch_size=${BATCH_SIZE}"
echo "        extract_point=${EXTRACT_POINT} thinking_mode=${THINKING_MODE}"
echo ""

TRAIN_ARGS=(
    "${PYTHON}" "${REPO_ROOT}/experiments/regression_head.py"
    --mode train
    --output-dir "${OUTPUT_DIR}"
    --train-split train
    --hidden-dim "${HIDDEN_DIM}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --batch-size "${BATCH_SIZE}"
    --dropout "${DROPOUT}"
    --patience "${PATIENCE}"
    --weight-decay 1e-4
    --smooth-l1-beta 0.01
    --val-fraction 0.2
    --seed 42
    --grad-clip 1.0
    --extract-point "${EXTRACT_POINT}"
    --thinking-mode "${THINKING_MODE}"
)

echo "Command: ${TRAIN_ARGS[*]}"
echo ""

"${TRAIN_ARGS[@]}"

echo ""
echo "============================================"
echo "Training complete at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
