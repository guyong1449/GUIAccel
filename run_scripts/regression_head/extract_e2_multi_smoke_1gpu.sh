#!/usr/bin/env bash
#SBATCH --job-name=reghead-e2-smoke
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_e2_multi_smoke_1gpu/slurm-%j.out
#SBATCH --error=/dkucc/home/rw335/GUIAccel/logs/regression_head/extract_e2_multi_smoke_1gpu/slurm-%j.err

set -euo pipefail

# --------------------------------------------------------------------------- #
# E2 smoke: multi-point GT-Forcing extract (thinking_end + action + coord_bracket)
# in ONE forward, ≤20 train episodes. New timestamped dir; never overwrite.
# --------------------------------------------------------------------------- #

REPO_ROOT="${GUIACCEL_REPO_ROOT:-/dkucc/home/rw335/GUIAccel}"
CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"
source .env 2>/dev/null || true

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${REPO_ROOT}/outputs/regression_head/e2_multi_smoke_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs/regression_head/extract_e2_multi_smoke_1gpu"

exec > >(tee -a "${OUTPUT_DIR}/extract_smoke.log") 2>&1

EPISODE_LIMIT="${EXTRACT_EPISODE_LIMIT:-20}"
THINKING_MODE="${EXTRACT_THINKING_MODE:-template}"
EXTRACT_POINT="${EXTRACT_EXTRACT_POINT:-multi}"

echo "============================================"
echo "GUIAccel — E2 multi-point extract smoke (1 GPU)"
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
assert data.get("multi_point") is True or all(
    k in data for k in ("h_thinking_end", "h_action", "h_coord_bracket")
), "missing multi-point tensors"
for key in ("h_thinking_end", "h_action", "h_coord_bracket"):
    t = data[key]
    assert t.shape == hs.shape, (key, t.shape, hs.shape)
# primary hidden_states should equal h_action
assert torch.allclose(hs, data["h_action"]), "hidden_states != h_action"
# three points should not be identical (almost surely)
diff_te = (data["h_thinking_end"] - data["h_action"]).abs().mean().item()
diff_cb = (data["h_coord_bracket"] - data["h_action"]).abs().mean().item()
print(f"mean|h_te-h_act|={diff_te:.6f} mean|h_cb-h_act|={diff_cb:.6f}")
assert diff_te > 0 or diff_cb > 0, "all three points identical — indexing bug?"
if meta:
    m0 = meta[0]
    print("sample meta keys:", sorted(m0.keys()))
    print("thinking_mode:", m0.get("thinking_mode"))
    print("extract_point:", m0.get("extract_point"))
    print(
        "prefix_lens:",
        m0.get("prefix_token_len_thinking_end"),
        m0.get("prefix_token_len_action"),
        m0.get("prefix_token_len_coord_bracket"),
    )
    assert m0.get("extract_point") == "multi"
    assert m0.get("thinking_mode") == "template"
    te = int(m0["prefix_token_len_thinking_end"])
    act = int(m0["prefix_token_len_action"])
    cb = int(m0["prefix_token_len_coord_bracket"])
    assert 1 <= te <= act <= cb, (te, act, cb)
print("VERIFY_OK")
PY
