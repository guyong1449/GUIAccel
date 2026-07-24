#!/usr/bin/env bash
# DivPrune evaluation wrapper for AndroidControl.
# Resume: pass --resume and the same --output-dir as the interrupted run.
# Usage:
#   bash scripts/androidcontrol/eval/run_divprune_eval.sh [--resume] [--split test]
#   bash scripts/androidcontrol/eval/run_divprune_eval.sh --smoke
#   bash scripts/androidcontrol/eval/run_divprune_eval.sh --config configs/androidcontrol/divprune/smoke.json
#
# --smoke uses configs/androidcontrol/divprune/smoke.json (episode_limit=1 sanity check).
# Submits via SLURM (default --time=7-00:00:00).

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

PYTHON="${SKILLREUSE_CONDA_PREFIX:?Set SKILLREUSE_CONDA_PREFIX in .env}/bin/python"
CONFIG="${REPO_ROOT}/configs/androidcontrol/divprune/default.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/divprune_keep0.098"
JOB_NAME="divprune-eval"

SMOKE=0
CONFIG_EXPLICIT=0
OUTPUT_DIR_EXPLICIT=0

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=1
            shift
            ;;
        --dry-run)
            echo "warning: --dry-run is deprecated for config selection; use --smoke" >&2
            SMOKE=1
            shift
            ;;
        --config)
            if [[ $# -lt 2 ]]; then
                echo "error: --config requires a path argument" >&2
                exit 1
            fi
            if [[ "$2" = /* ]]; then
                CONFIG="$2"
            else
                CONFIG="${REPO_ROOT}/$2"
            fi
            CONFIG_EXPLICIT=1
            shift 2
            ;;
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --output-dir requires a path argument" >&2
                exit 1
            fi
            if [[ "$2" = /* ]]; then
                OUTPUT_DIR="$2"
            else
                OUTPUT_DIR="${REPO_ROOT}/$2"
            fi
            OUTPUT_DIR_EXPLICIT=1
            shift 2
            ;;
        --episode-limit)
            echo "error: --episode-limit is not supported by run_evaluation.py; use --smoke or --config configs/androidcontrol/divprune/smoke.json" >&2
            exit 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${SMOKE}" -eq 1 ]]; then
    if [[ "${CONFIG_EXPLICIT}" -eq 0 ]]; then
        CONFIG="${REPO_ROOT}/configs/androidcontrol/divprune/smoke.json"
    fi
    if [[ "${OUTPUT_DIR_EXPLICIT}" -eq 0 ]]; then
        OUTPUT_DIR="${REPO_ROOT}/outputs/divprune_keep0.098_smoke"
    fi
    JOB_NAME="divprune-smoke"
fi

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib/submit_gpu_job.sh"

submit_gpu_job "${JOB_NAME}" "${REPO_ROOT}/logs" -- \
    "${PYTHON}" "${REPO_ROOT}/scripts/core/run_evaluation.py" \
    --benchmark AndroidControl \
    --config "${CONFIG}" \
    --variant V0 \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
