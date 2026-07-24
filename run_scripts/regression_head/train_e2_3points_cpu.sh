#!/usr/bin/env bash
#SBATCH --job-name=reghead-e2-train3
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/train_e2_3points_cpu/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/train_e2_3points_cpu/slurm-%j.err

set -euo pipefail

# --------------------------------------------------------------------------- #
# E2: train three CoordHeads (thinking_end / action / coord_bracket) from one
# multi-point extract artifact. Same T3 hyperparams. New timestamped dirs.
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

# Required: path to multi-point extracted/ dir (or parent run dir)
EXTRACT_SRC="${EXTRACT_SRC:?Set EXTRACT_SRC to multi-point extract run or .../extracted}"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true
mkdir -p "${REPO_ROOT}/logs/regression_head/train_e2_3points_cpu"

# Resolve extract dir
if [[ -f "${EXTRACT_SRC}/train_hidden_states.pt" ]]; then
    EXTRACT_DIR="${EXTRACT_SRC}"
elif [[ -f "${EXTRACT_SRC}/extracted/train_hidden_states.pt" ]]; then
    EXTRACT_DIR="${EXTRACT_SRC}/extracted"
else
    echo "ERROR: no train_hidden_states.pt under EXTRACT_SRC=${EXTRACT_SRC}"
    exit 1
fi

N_CPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${N_CPUS}"
export MKL_NUM_THREADS="${N_CPUS}"
export OPENBLAS_NUM_THREADS="${N_CPUS}"
export NUMEXPR_NUM_THREADS="${N_CPUS}"
export CUDA_VISIBLE_DEVICES=""

BATCH_TS="$(date '+%Y%m%d_%H%M%S')"
BATCH_DIR="${REPO_ROOT}/outputs/regression_head/e2_train3_${BATCH_TS}"
mkdir -p "${BATCH_DIR}"

exec > >(tee -a "${BATCH_DIR}/train3.log") 2>&1

echo "============================================"
echo "GUIAccel — E2 train 3 extract-point heads"
echo "============================================"
echo "Batch dir:    ${BATCH_DIR}"
echo "Extract dir:  ${EXTRACT_DIR}"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo ""

"${PYTHON}" - <<PY
import torch
from pathlib import Path
p = Path("${EXTRACT_DIR}") / "train_hidden_states.pt"
data = torch.load(p, map_location="cpu", weights_only=False)
assert data.get("multi_point") or all(k in data for k in ("h_thinking_end","h_action","h_coord_bracket"))
print(f"Loaded {p}: N={data['hidden_states'].shape[0]} multi={data.get('multi_point')}")
for k in ("h_thinking_end","h_action","h_coord_bracket"):
    print(f"  {k}: {tuple(data[k].shape)}")
PY

HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-256}"
EPOCHS="${TRAIN_EPOCHS:-50}"
LR="${TRAIN_LR:-3e-4}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
DROPOUT="${TRAIN_DROPOUT:-0.1}"
PATIENCE="${TRAIN_PATIENCE:-10}"

SUMMARY_JSON="${BATCH_DIR}/e2_train3_summary.json"
echo "{" > "${SUMMARY_JSON}.tmp"
FIRST=1

for POINT in thinking_end action coord_bracket; do
    TS="$(date '+%Y%m%d_%H%M%S')"
    OUT="${REPO_ROOT}/outputs/regression_head/train_e2_${POINT}_${TS}"
    mkdir -p "${OUT}"
    ln -sfn "${EXTRACT_DIR}" "${OUT}/extracted"
    echo ""
    echo "======== Training extract_point=${POINT} → ${OUT} ========"
    "${PYTHON}" "${REPO_ROOT}/experiments/regression_head.py" \
        --mode train \
        --output-dir "${OUT}" \
        --train-split train \
        --hidden-dim "${HIDDEN_DIM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --batch-size "${BATCH_SIZE}" \
        --dropout "${DROPOUT}" \
        --patience "${PATIENCE}" \
        --weight-decay 1e-4 \
        --smooth-l1-beta 0.01 \
        --val-fraction 0.2 \
        --seed 42 \
        --grad-clip 1.0 \
        --extract-point "${POINT}" \
        --thinking-mode template

    # Record pointer for batch summary
    ln -sfn "${OUT}" "${BATCH_DIR}/${POINT}"
    VAL_MAE="$("${PYTHON}" -c "
import json
from pathlib import Path
h=json.loads(Path('${OUT}/trained/training_history.json').read_text())
best=min(h, key=lambda r: r['val_loss'])
print(f\"{best['val_mae_999']:.4f}\")
")"
    BEST_EPOCH="$("${PYTHON}" -c "
import json
from pathlib import Path
s=json.loads(Path('${OUT}/trained/training_summary.json').read_text())
print(s['best_epoch'])
")"
    echo "POINT=${POINT} OUT=${OUT} best_epoch=${BEST_EPOCH} val_MAE@999=${VAL_MAE}"
    if [[ "${FIRST}" -eq 1 ]]; then
        FIRST=0
    else
        echo "," >> "${SUMMARY_JSON}.tmp"
    fi
    printf '  "%s": {"output_dir": "%s", "best_epoch": %s, "val_mae_999": %s}' \
        "${POINT}" "${OUT}" "${BEST_EPOCH}" "${VAL_MAE}" >> "${SUMMARY_JSON}.tmp"
done

echo "" >> "${SUMMARY_JSON}.tmp"
echo "}" >> "${SUMMARY_JSON}.tmp"
mv "${SUMMARY_JSON}.tmp" "${SUMMARY_JSON}"

echo ""
echo "============================================"
echo "E2 3-head training complete"
echo "Batch summary: ${SUMMARY_JSON}"
cat "${SUMMARY_JSON}"
echo "============================================"
