#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# DivPrune evaluation SLURM script (VisionZip eval_visionzip_slurm.sh pattern)
#
# Dual-mode:
#   - Login node (no SLURM_JOB_ID): sbatch-submit this script as a job
#   - Compute node (SLURM_JOB_ID set): run DivPrune integration test or full eval
#
# DivPrune uses local transformers workers (no vLLM). Base Qwen3-VL only.
# --------------------------------------------------------------------------- #

if [[ -n "${DIVPRUNE_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${DIVPRUNE_REPO_ROOT}"
  SCRIPT_DIR="${REPO_ROOT}/run_scripts"
else
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
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib/load_env.sh"
skillreuse_load_env "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/run_scripts/_lib/slurm_helpers.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/androidcontrol/divprune/default.json"
DEFAULT_JOB_NAME="divprune-eval"
DEFAULT_PARTITION="l20-gpu"
DEFAULT_ACCOUNT="faculty"
DEFAULT_TIME="7-00:00:00"
DEFAULT_GPUS="4"
DEFAULT_CPUS="16"
DEFAULT_MEM="128G"
DEFAULT_CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:-${REPO_ROOT}/.conda/envs/skillreuse}"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs/divprune_keep0.098"
DEFAULT_MODE="eval"
DEFAULT_SPLIT=""
DEFAULT_MEASURE_E2E="false"
DEFAULT_RESUME="false"

# Server is US time; +12h gives Beijing date for journal folders.
cst_date() { date -d "+12 hours" "$@"; }

usage() {
  cat <<EOF
Usage:
  ${0} [options]

Behavior:
  - On login node: automatically sbatch-submits this script as a DivPrune eval job
  - On compute node: activates conda env and runs DivPrune eval (local transformers, no vLLM)
  - Writes Slurm stdout/stderr to training_journal/run_* or slurm_logs/divprune/smoke/

Examples:
  ${0} --smoke --gpus 1
  ${0} --partition l20-gpu --account faculty --gpus 4
  ${0} --output-dir ${REPO_ROOT}/outputs/divprune_keep0.098 --comment full-test

Options:
  --job-name NAME          Slurm job name (default: ${DEFAULT_JOB_NAME})
  --config PATH            JSON config (default: ${DEFAULT_CONFIG})
  --mode MODE              eval|smoke (default: ${DEFAULT_MODE}; --smoke equals --mode smoke)
  --smoke                  Single-GPU integration test (scripts/divprune/integration_test_divprune.py)
  --gpus N                 GPU count (default: ${DEFAULT_GPUS}; smoke recommends 1)
  --partition NAME         Slurm partition (default: ${DEFAULT_PARTITION})
  --account NAME           Slurm account (default: ${DEFAULT_ACCOUNT})
  --time HH:MM:SS          Job time limit (default: ${DEFAULT_TIME})
  --cpus N                 CPUs per task (default: ${DEFAULT_CPUS})
  --mem SIZE               Memory (default: ${DEFAULT_MEM})
  --conda-prefix PATH      Conda env path (default: ${DEFAULT_CONDA_PREFIX})
  --output-dir PATH        Eval output directory (default: ${DEFAULT_OUTPUT_DIR}_<run_stamp>)
  --split NAME             AndroidControl split override (e.g. validation / test)
  --measure-e2e-latency    Measure per-step end-to-end latency during eval
  --resume                 Resume from checkpoint in output-dir (don't clean existing checkpoint)
  --comment TEXT           Comment tag appended to job name
  --dry-run                Only print sbatch command, don't submit
  -h, --help               Show help

Log layout:
  - smoke:            ${REPO_ROOT}/slurm_logs/divprune/smoke/run_<YYYYMMDD_HHMMSS>/
  - eval tentative:   ${REPO_ROOT}/training_journal/.tentative/run_<YYYYMMDD_HHMMSS>/
  - eval promoted:    ${REPO_ROOT}/training_journal/<YYYY_MM_DD>/run_<YYYYMMDD_HHMMSS>/
  - Slurm native:     <run-dir>/slurm.out | <run-dir>/slurm.err
  - terminal.log:     symlink to slurm.out (legacy tail compatibility)
EOF
}

sanitize_name() {
  local value="$1"
  value="${value// /_}"
  value="${value//\//_}"
  value="${value//:/-}"
  value="${value//,/__}"
  printf '%s' "${value}"
}

submit_job() {
  local run_stamp
  run_stamp="$(cst_date '+%Y%m%d_%H%M%S')"
  local run_date
  run_date="$(cst_date '+%Y_%m_%d')"
  local journal_run_dir
  if [[ "${MODE}" == "smoke" ]]; then
    mkdir -p "${REPO_ROOT}/slurm_logs/divprune/smoke"
    journal_run_dir="${REPO_ROOT}/slurm_logs/divprune/smoke/run_${run_stamp}"
  else
    mkdir -p "${REPO_ROOT}/training_journal/.tentative"
    journal_run_dir="${REPO_ROOT}/training_journal/.tentative/run_${run_stamp}"
  fi
  mkdir -p "${journal_run_dir}"

  if [[ "${OUTPUT_DIR_EXPLICIT}" != "true" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_${run_stamp}"
  fi

  local output_pattern="${journal_run_dir}/slurm.out"
  local error_pattern="${journal_run_dir}/slurm.err"

  local export_vars=(
    "ALL"
    "DIVPRUNE_REPO_ROOT=${REPO_ROOT}"
    "DIVPRUNE_CONFIG=${CONFIG_PATH}"
    "DIVPRUNE_MODE=${MODE}"
    "DIVPRUNE_CONDA_PREFIX=${CONDA_PREFIX_PATH}"
    "DIVPRUNE_OUTPUT_DIR=${OUTPUT_DIR}"
    "DIVPRUNE_SPLIT=${SPLIT}"
    "DIVPRUNE_MEASURE_E2E=${MEASURE_E2E}"
    "DIVPRUNE_RESUME=${RESUME}"
    "DIVPRUNE_RUN_DATE=${run_date}"
    "DIVPRUNE_RUN_STAMP=${run_stamp}"
    "DIVPRUNE_JOURNAL_RUN_DIR=${journal_run_dir}"
    "DIVPRUNE_JOB_SUBMITTED_AT=${run_stamp}"
    "DIVPRUNE_JOB_COMMENT=${COMMENT}"
    "DIVPRUNE_GPUS=${GPUS}"
  )

  local partition="${PARTITION:-${DEFAULT_PARTITION}}"
  local sbatch_cmd=(sbatch)
  sbatch_cmd+=(--job-name "${JOB_NAME}")
  sbatch_cmd+=(--nodes 1 --ntasks 1)
  sbatch_cmd+=(--partition "${partition}")
  sbatch_cmd+=(--gres "$(slurm_gres_string "${partition}" "${GPUS}")")
  sbatch_cmd+=(--cpus-per-task "${CPUS}")
  sbatch_cmd+=(--mem "${MEM}")
  sbatch_cmd+=(--time "${TIME_LIMIT}")
  sbatch_cmd+=(--chdir "${REPO_ROOT}")
  sbatch_cmd+=(--output "${output_pattern}")
  sbatch_cmd+=(--error "${error_pattern}")
  sbatch_cmd+=(--export "$(IFS=,; echo "${export_vars[*]}")")
  if [[ -n "${ACCOUNT}" ]]; then
    sbatch_cmd+=(--account "${ACCOUNT}")
  fi
  sbatch_cmd+=("${0}")

  local submit_output
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run sbatch command:" >&2
    printf '  %q' "${sbatch_cmd[@]}" >&2
    printf '\n' >&2
    echo "Journal dir: ${journal_run_dir}"
    echo "Output dir: ${OUTPUT_DIR}"
    return 0
  fi

  if ! submit_output="$("${sbatch_cmd[@]}" 2>&1)"; then
    echo "sbatch failed: ${submit_output}" >&2
    exit 1
  fi

  local job_id
  job_id="$(awk '{print $NF}' <<<"${submit_output}")"
  if [[ -z "${job_id}" ]]; then
    echo "Failed to parse sbatch output: ${submit_output}" >&2
    exit 1
  fi

  echo "${submit_output}"
  echo "Journal dir: ${journal_run_dir}"
  if [[ "${MODE}" != "smoke" ]]; then
    echo "Promoted run dir: ${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
  fi
  echo "Queue status: squeue -j ${job_id}"
  echo "Slurm output: tail -f ${output_pattern}"
}

run_job() {
  local config_path="${DIVPRUNE_CONFIG:-${DEFAULT_CONFIG}}"
  local mode="${DIVPRUNE_MODE:-${DEFAULT_MODE}}"
  local conda_prefix="${DIVPRUNE_CONDA_PREFIX:-${DEFAULT_CONDA_PREFIX}}"
  local output_dir="${DIVPRUNE_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
  local split="${DIVPRUNE_SPLIT:-}"
  local measure_e2e="${DIVPRUNE_MEASURE_E2E:-false}"
  local resume="${DIVPRUNE_RESUME:-false}"
  local run_date="${DIVPRUNE_RUN_DATE:-$(cst_date '+%Y_%m_%d')}"
  local run_stamp="${DIVPRUNE_RUN_STAMP:-$(cst_date '+%Y%m%d_%H%M%S')}"
  local journal_run_dir="${DIVPRUNE_JOURNAL_RUN_DIR:-}"
  local comment="${DIVPRUNE_JOB_COMMENT:-}"
  local requested_gpus="${DIVPRUNE_GPUS:-${GPUS}}"

  if [[ -z "${journal_run_dir}" ]]; then
    if [[ "${mode}" == "smoke" ]]; then
      mkdir -p "${REPO_ROOT}/slurm_logs/divprune/smoke"
      journal_run_dir="${REPO_ROOT}/slurm_logs/divprune/smoke/run_${run_stamp}"
    else
      mkdir -p "${REPO_ROOT}/training_journal/${run_date}"
      journal_run_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
    fi
  fi
  mkdir -p "${journal_run_dir}"
  ln -sf slurm.out "${journal_run_dir}/terminal.log"

  cleanup_on_failure() {
    if [[ "${mode}" == "smoke" ]]; then
      return 0
    fi
    if [[ -d "${journal_run_dir}" && "${journal_run_dir}" == *"/.tentative/"* ]]; then
      local aborted_dir="${REPO_ROOT}/training_journal/.aborted/$(basename "${journal_run_dir}")"
      mkdir -p "${aborted_dir}"
      for name in slurm.out slurm.err terminal.log config_snapshot.json pointers.json run_summary.md; do
        if [[ -f "${journal_run_dir}/${name}" ]]; then
          cp -f "${journal_run_dir}/${name}" "${aborted_dir}/${name}"
        fi
      done
      rm -rf "${journal_run_dir}"
    fi
  }
  trap cleanup_on_failure EXIT

  export DIVPRUNE_JOURNAL_RUN_DIR="${journal_run_dir}"
  export DIVPRUNE_RUN_DATE="${run_date}"
  export DIVPRUNE_RUN_STAMP="${run_stamp}"
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1

  # Ensure CUDA_VISIBLE_DEVICES is set for GPU jobs
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${requested_gpus}" -gt 0 ]]; then
    if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    else
      export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((requested_gpus - 1)))"
    fi
    echo "[$(cst_date '+%F %T')] Set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (fallback)"
  fi

  echo "[$(cst_date '+%F %T')] Starting DivPrune job ${SLURM_JOB_ID:-local} (mode=${mode})"
  echo "Repo root: ${REPO_ROOT}"
  echo "Config: ${config_path}"
  echo "Mode: ${mode}"
  echo "Requested GPUs: ${requested_gpus}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "Conda prefix: ${conda_prefix}"
  echo "Output dir: ${output_dir}"
  echo "Journal run dir: ${journal_run_dir}"

  cd "${REPO_ROOT}"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_prefix}"
  else
    export PATH="${conda_prefix}/bin:${PATH}"
  fi

  local python_bin="${conda_prefix}/bin/python"

  "${python_bin}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from skillreuse.journal import prepare_journal
from pathlib import Path
prepare_journal(
    Path('${journal_run_dir}'),
    config_path='${config_path}',
    output_dir='${output_dir}',
    mode='divprune_${mode}',
    comment='${comment}',
)
"

  local client_exit=0

  if [[ "${mode}" == "smoke" ]]; then
    local smoke_args=("${python_bin}" "${REPO_ROOT}/scripts/divprune/integration_test_divprune.py")
    echo "[$(cst_date '+%F %T')] Launching smoke test: ${smoke_args[*]}"
    "${smoke_args[@]}" || client_exit=$?
  else
    local eval_args=("${python_bin}" "${REPO_ROOT}/scripts/core/run_evaluation.py"
      --benchmark AndroidControl
      --config "${config_path}"
      --variant V0
      --output-dir "${output_dir}")

    if [[ -n "${split}" ]]; then
      eval_args+=(--split "${split}")
    fi
    if [[ "${measure_e2e}" == "true" ]]; then
      eval_args+=(--measure-e2e-latency)
    fi
    if [[ "${resume}" == "true" ]]; then
      eval_args+=(--resume)
    fi

    echo "[$(cst_date '+%F %T')] Launching eval: ${eval_args[*]}"
    "${eval_args[@]}" || client_exit=$?
  fi

  if [[ "${mode}" != "smoke" && ${client_exit} -eq 0 ]]; then
    local final_dir="${REPO_ROOT}/training_journal/${run_date}/run_${run_stamp}"
    if [[ "${journal_run_dir}" == *"/.tentative/"* ]]; then
      mkdir -p "$(dirname "${final_dir}")"
      mv "${journal_run_dir}" "${final_dir}"
      journal_run_dir="${final_dir}"
      export DIVPRUNE_JOURNAL_RUN_DIR="${journal_run_dir}"
      echo "[$(cst_date '+%F %T')] Journal promoted to: ${journal_run_dir}"
      trap - EXIT
    fi
  fi

  local final_status="completed"
  if [[ ${client_exit} -ne 0 ]]; then
    final_status="failed"
  fi

  if [[ "${mode}" != "smoke" ]]; then
    "${python_bin}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from skillreuse.journal import update_status
from pathlib import Path
update_status(Path('${journal_run_dir}'), '${final_status}')
"
  fi

  echo "[$(cst_date '+%F %T')] Done. Status: ${final_status} (exit code: ${client_exit})"
  exit ${client_exit}
}

JOB_NAME="${DEFAULT_JOB_NAME}"
CONFIG_PATH="${DIVPRUNE_CONFIG:-${DEFAULT_CONFIG}}"
MODE="${DEFAULT_MODE}"
GPUS="${DEFAULT_GPUS}"
PARTITION="${DEFAULT_PARTITION}"
ACCOUNT="${DEFAULT_ACCOUNT}"
TIME_LIMIT="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
CONDA_PREFIX_PATH="${DEFAULT_CONDA_PREFIX}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
SPLIT="${DEFAULT_SPLIT}"
MEASURE_E2E="${DEFAULT_MEASURE_E2E}"
RESUME="${DEFAULT_RESUME}"
COMMENT=""
DRY_RUN="false"
PARTITION_EXPLICIT="false"
OUTPUT_DIR_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)             JOB_NAME="$2"; shift 2 ;;
    --config)               CONFIG_PATH="$2"; shift 2 ;;
    --mode)                 MODE="$2"; shift 2 ;;
    --smoke)                MODE="smoke"; GPUS="1"; shift ;;
    --gpus)                 GPUS="$2"; shift 2 ;;
    --partition)            PARTITION="$2"; PARTITION_EXPLICIT="true"; shift 2 ;;
    --account)              ACCOUNT="$2"; shift 2 ;;
    --time)                 TIME_LIMIT="$2"; shift 2 ;;
    --cpus)                 CPUS="$2"; shift 2 ;;
    --mem)                  MEM="$2"; shift 2 ;;
    --conda-prefix)         CONDA_PREFIX_PATH="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT="true"; shift 2 ;;
    --split)                SPLIT="$2"; shift 2 ;;
    --measure-e2e-latency)  MEASURE_E2E="true"; shift ;;
    --resume)               RESUME="true"; shift ;;
    --comment)              COMMENT="$2"; shift 2 ;;
    --dry-run)              DRY_RUN="true"; shift ;;
    -h|--help)              usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${MODE}" == "smoke" && "${GPUS}" != "1" ]]; then
  echo "NOTE: smoke mode typically uses --gpus 1 (current: ${GPUS})"
fi

if [[ -z "${SLURM_JOB_ID:-}" && ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_job
else
  run_job
fi
