#!/usr/bin/env bash
# =============================================================================
# run_visionzip_eval.sh — AndroidControl V0 eval with VisionZip (local transformers)
#
# No LoRA, no vLLM. Uses configs/androidcontrol/visionzip/default.json (4 worker GPUs).
#
# Usage:
#   bash scripts/androidcontrol/eval/run_visionzip_eval.sh
#   bash scripts/androidcontrol/eval/run_visionzip_eval.sh --output-dir outputs/visionzip_androidcontrol_smoke
#   bash scripts/androidcontrol/eval/run_visionzip_eval.sh --measure-e2e-latency
#
# Slurm (4× GPU, SpectralMAE-style params):
#   sbatch scripts/androidcontrol/slurm/submit_visionzip_androidcontrol.sh
# =============================================================================
set -euo pipefail

# shellcheck disable=SC1091
_SKILLREUSE_SEARCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    # shellcheck disable=SC1090
    source "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh"
    skillreuse_init_from "${BASH_SOURCE[0]}"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done

PROJ="${SKILLREUSE_REPO_ROOT}"
CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:?Set SKILLREUSE_CONDA_PREFIX in .env}"
PYTHON="${CONDA_PREFIX}/bin/python"
CONFIG="${PROJ}/configs/androidcontrol/visionzip/default.json"
OUTPUT_DIR="${PROJ}/outputs/visionzip_androidcontrol"
VARIANT="V0"
SPLIT=""
MEASURE_E2E=0

usage() {
  cat <<EOF
Usage: ${0} [options]

Options:
  --config PATH         JSON config (default: ${CONFIG})
  --output-dir PATH     Eval output directory (default: ${OUTPUT_DIR})
  --variant V0|V1|V2|V3 Eval variant (default: ${VARIANT}; V0 needs no skill library)
  --split NAME          Optional split override, e.g. validation or test
  --measure-e2e-latency Record per-step end-to-end latency
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --measure-e2e-latency)
      MEASURE_E2E=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${PROJ}/logs" "${OUTPUT_DIR}"

# Slurm sets CUDA_VISIBLE_DEVICES for the allocation; log for debugging.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
  fi
fi

log() { echo "[$(date '+%F %T')] $*"; }

log "VisionZip AndroidControl eval"
log "  host=$(hostname)"
log "  job_id=${SLURM_JOB_ID:-<interactive>}"
log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
log "  config=${CONFIG}"
log "  variant=${VARIANT}"
log "  output_dir=${OUTPUT_DIR}"

cd "$PROJ"

cmd=(
  "$PYTHON" scripts/core/run_evaluation.py
  --benchmark AndroidControl
  --config "$CONFIG"
  --variant "$VARIANT"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$SPLIT" ]]; then
  cmd+=(--split "$SPLIT")
fi
if [[ "$MEASURE_E2E" -eq 1 ]]; then
  cmd+=(--measure-e2e-latency)
fi

log "Launch: ${cmd[*]}"
exec "${cmd[@]}"
