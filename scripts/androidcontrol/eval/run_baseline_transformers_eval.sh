#!/usr/bin/env bash
# Local transformers baseline evaluation wrapper for AndroidControl.
# Resume: pass --resume and the same --output-dir as the interrupted run.
# Usage:
#   bash scripts/androidcontrol/eval/run_baseline_transformers_eval.sh [--resume] [--split test]
#   bash scripts/androidcontrol/eval/run_baseline_transformers_eval.sh --dry-run
#   bash scripts/androidcontrol/eval/run_baseline_transformers_eval.sh --smoke
#   bash scripts/androidcontrol/eval/run_baseline_transformers_eval.sh --smoke --dry-run
#   bash scripts/androidcontrol/eval/run_baseline_transformers_eval.sh --config configs/androidcontrol/baseline/smoke.json
#
# --dry-run: resolve everything and PRINT the exact sbatch command that WOULD run,
#            then exit WITHOUT submitting any SLURM job (no GPU allocation).
# --smoke:   use configs/androidcontrol/baseline/smoke.json (episode_limit=1)
#            (episode_limit in JSON), a distinct output dir + job name, and SUBMIT normally.
#            Combine with --dry-run to print the smoke command without submitting.
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
CONFIG="${REPO_ROOT}/configs/androidcontrol/baseline/default.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/baseline_transformers"
JOB_NAME="baseline-transformers-eval"

DRY_RUN=0
SMOKE=0
CONFIG_EXPLICIT=0
OUTPUT_DIR_EXPLICIT=0

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --smoke)
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
            echo "error: --episode-limit is not supported by run_evaluation.py; use --smoke or --config configs/androidcontrol/baseline/smoke.json" >&2
            exit 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# --smoke takes over the old "small config" behavior: small episode_limit config,
# a distinct output dir + job name. Explicit --config/--output-dir still win.
if [[ "${SMOKE}" -eq 1 ]]; then
    if [[ "${CONFIG_EXPLICIT}" -eq 0 ]]; then
        CONFIG="${REPO_ROOT}/configs/androidcontrol/baseline/smoke.json"
    fi
    if [[ "${OUTPUT_DIR_EXPLICIT}" -eq 0 ]]; then
        OUTPUT_DIR="${REPO_ROOT}/outputs/baseline_transformers_smoke"
    fi
    JOB_NAME="baseline-transformers-smoke"
fi

# A true dry-run must never reach sbatch and must not create side-effect dirs.
if [[ "${DRY_RUN}" -eq 1 ]]; then
    export SLURM_DRY_RUN=1
else
    # Be authoritative: neutralize any inherited dry-run env so a real run always submits.
    export SLURM_DRY_RUN=0
    mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs"
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib/submit_gpu_job.sh"

submit_gpu_job "${JOB_NAME}" "${REPO_ROOT}/logs" -- \
    "${PYTHON}" "${REPO_ROOT}/scripts/core/run_evaluation.py" \
    --benchmark AndroidControl \
    --config "${CONFIG}" \
    --variant V0 \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
