#!/usr/bin/env bash
#SBATCH --job-name=reghead-e1a-smoke
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_smoke_1gpu/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_smoke_1gpu/slurm-%j.err

set -euo pipefail

# --------------------------------------------------------------------------- #
# T1 / E1-A smoke: thinking-aware GT-Forcing extract on ≤20 train episodes.
# Writes a NEW timestamped output dir (never overwrites baselines).
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/outputs/regression_head/e1a_smoke_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs/regression_head/extract_smoke_1gpu"

# If launched via sbatch without -o/-e, tee into the output dir.
exec > >(tee -a "${OUTPUT_DIR}/extract_smoke.log") 2>&1

EPISODE_LIMIT="${EXTRACT_EPISODE_LIMIT:-20}"
THINKING_MODE="${EXTRACT_THINKING_MODE:-template}"
EXTRACT_POINT="${EXTRACT_EXTRACT_POINT:-action}"

echo "============================================"
echo "GUIAccel — E1-A extract smoke (1 GPU)"
echo "============================================"
echo "Timestamp:    ${TIMESTAMP}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Python:       ${PYTHON}"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "episode_limit=${EPISODE_LIMIT} thinking_mode=${THINKING_MODE} extract_point=${EXTRACT_POINT}"
echo ""

"${PYTHON}" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info(0)
    print(f'GPU free/total: {free/1e9:.1f}/{total/1e9:.1f} GB')
"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${REPO_ROOT}/experiments/regression_head.py" \
    --mode extract \
    --output-dir "${OUTPUT_DIR}" \
    --split train \
    --episode-limit "${EPISODE_LIMIT}" \
    --device 0 \
    --dtype bfloat16 \
    --max-new-tokens 512 \
    --thinking-mode "${THINKING_MODE}" \
    --extract-point "${EXTRACT_POINT}"

echo ""
echo "=== Smoke complete ==="
echo "Output: ${OUTPUT_DIR}"
"${PYTHON}" - <<PY
from pathlib import Path
import torch
out = Path("${OUTPUT_DIR}") / "extracted" / "train_hidden_states.pt"
if not out.exists():
    raise SystemExit(f"MISSING {out}")
data = torch.load(out, map_location="cpu", weights_only=False)
hs = data["hidden_states"]
meta = data.get("metadata") or data.get("meta") or []
print(f"N={hs.shape[0]} h_dim={hs.shape[1]} len(metadata)={len(meta)}")
assert hs.shape[1] == 4096, hs.shape
assert len(meta) == hs.shape[0], (len(meta), hs.shape[0])
if meta:
    print("sample meta keys:", sorted(meta[0].keys()))
    print("thinking_mode:", meta[0].get("thinking_mode"))
    print("extract_point:", meta[0].get("extract_point"))
    print("prefix_token_len:", meta[0].get("prefix_token_len") or meta[0].get("generated_tokens"))
print("VERIFY_OK")
PY
