#!/usr/bin/env bash
#SBATCH --job-name=reghead-decode-eval
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/eval_decode_4gpu/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/eval_decode_4gpu/slurm-%j.err

set -euo pipefail

# --------------------------------------------------------------------------- #
# Coordinate Regression Head — Phase 3: Full AR Decode vs CoordHead (4 GPU)
#
# For each CLICK/LONG_PRESS on the eval split:
#   1) model.generate() → parse mobile_use coords (AR baseline)
#   2) GT-Forcing Prefill + CoordHead → coords (regression)
# Compare MAE vs GT, bbox hit, latency ratio.
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

# Trained head from Phase-2 (LayerNorm + lr=3e-4)
TRAIN_RUN="${TRAIN_RUN:-${REPO_ROOT}/outputs/regression_head/train_20260723_121025}"
SPLIT="${EVAL_SPLIT:-test}"
EPISODE_LIMIT="${EVAL_EPISODE_LIMIT:-}"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/outputs/regression_head/decode_eval_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
ln -sfn "${TRAIN_RUN}/trained" "${OUTPUT_DIR}/trained"

exec > >(tee -a "${OUTPUT_DIR}/decode_eval.log") 2>&1

echo "============================================"
echo "GUIAccel — Regression Head Decode Eval"
echo "============================================"
echo "Timestamp:    ${TIMESTAMP}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Train run:    ${TRAIN_RUN}"
echo "Split:        ${SPLIT}"
echo "Episode lim:  ${EPISODE_LIMIT:-all}"
echo "thinking_mode:${EVAL_THINKING_MODE:-template}"
echo "extract_point:${EVAL_EXTRACT_POINT:-action}"
echo "Python:       ${PYTHON}"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo ""

"${PYTHON}" -c "
import torch, sys
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
n = torch.cuda.device_count()
print(f'GPU count: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, '
          f'{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
need = int('${EVAL_NUM_GPUS:-4}')
if n < need:
    print(f'ERROR: need {need} GPUs, visible={n}', file=sys.stderr)
    sys.exit(1)
"

# Prefer explicit EVAL_NUM_GPUS; fall back to SLURM allocation; default 4.
NUM_GPUS="${EVAL_NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
if [[ "${NUM_GPUS}" -lt 4 ]]; then
    echo "ERROR: decode-eval requires 4 GPUs, got NUM_GPUS=${NUM_GPUS}"
    exit 1
fi
NUM_GPUS=4

THINKING_MODE="${EVAL_THINKING_MODE:-template}"
EXTRACT_POINT="${EVAL_EXTRACT_POINT:-action}"

EVAL_ARGS=(
    "${PYTHON}" "${REPO_ROOT}/experiments/regression_head.py"
    --mode eval
    --output-dir "${OUTPUT_DIR}"
    --split "${SPLIT}"
    --num-gpus "${NUM_GPUS}"
    --dtype bfloat16
    --max-new-tokens 512
    --thinking-mode "${THINKING_MODE}"
    --extract-point "${EXTRACT_POINT}"
)

if [[ -n "${EPISODE_LIMIT}" ]]; then
    EVAL_ARGS+=(--episode-limit "${EPISODE_LIMIT}")
fi

echo ""
echo "Command: ${EVAL_ARGS[*]}"
echo ""

"${EVAL_ARGS[@]}"

echo ""
echo "============================================"
echo "Decode eval complete at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
