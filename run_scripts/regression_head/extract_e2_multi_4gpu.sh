#!/usr/bin/env bash
#SBATCH --job-name=reghead-e2-extract
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_e2_multi_4gpu/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_e2_multi_4gpu/slurm-%j.err

set -euo pipefail

# --------------------------------------------------------------------------- #
# E2 full train multi-point extract (4×A40): one forward → three h_* tensors.
# New timestamped outputs/regression_head/<TS>/; never overwrite baselines.
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/outputs/regression_head/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs/regression_head/extract_e2_multi_4gpu"

exec > >(tee -a "${OUTPUT_DIR}/extract.log") 2>&1

echo "============================================"
echo "GUIAccel — E2 Multi-Point Extraction (4 GPU)"
echo "============================================"
echo "Timestamp:    ${TIMESTAMP}"
echo "Repo root:    ${REPO_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Python:       ${PYTHON}"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo ""

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
EXTRACT_POINT="${EXTRACT_EXTRACT_POINT:-multi}"

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
echo "=== Post-merge verify ==="
"${PYTHON}" - <<PY
from pathlib import Path
import torch
out = Path("${OUTPUT_DIR}") / "extracted" / "${SPLIT}_hidden_states.pt"
data = torch.load(out, map_location="cpu", weights_only=False)
hs = data["hidden_states"]
meta = data.get("metadata") or data.get("meta") or []
print(f"N={hs.shape[0]} h_dim={hs.shape[1]} meta={len(meta)} multi={data.get('multi_point')}")
assert len(meta) == hs.shape[0]
for key in ("h_thinking_end", "h_action", "h_coord_bracket"):
    assert data[key].shape == hs.shape, key
assert torch.allclose(hs, data["h_action"])
print("VERIFY_OK", out)
PY

echo ""
echo "============================================"
echo "E2 multi-point extraction complete at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
