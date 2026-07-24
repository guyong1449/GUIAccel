#!/bin/bash
#SBATCH --chdir=/dkucc/home/rw335/SkillReuse
set -euo pipefail

PROJ=/dkucc/home/rw335/SkillReuse
SCRIPT_DIR="${PROJ}/run_scripts"
# shellcheck disable=SC1091
source "${REPO_ROOT}/run_scripts/_lib/journal_helpers.sh"

MERGED_PATH="${PROJ}/outputs/sft_merged/config.json"
SFT_DONE="${PROJ}/outputs/sft_output/.sft_done"
SFT_CONFIG="${PROJ}/configs/sft_phase_a.json"
OUTPUT_DIR="${PROJ}/outputs/sft_output"
CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:-${PROJ}/.conda/envs/skillreuse}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

journal_run_setup "${PROJ}" "${JOURNAL_RUN_DIR:-}" "${JOURNAL_RUN_DATE:-}" "${JOURNAL_RUN_STAMP:-}"
journal_run_dir="${JOURNAL_RUN_DIR}"

cd "$PROJ"
mkdir -p "${OUTPUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_PREFIX}"
else
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

python_bin="${CONDA_PREFIX}/bin/python"
if [[ -f "${SFT_CONFIG}" ]]; then
  config_path="${SFT_CONFIG}"
else
  config_path="${PROJ}/configs/androidcontrol/default.json"
fi

journal_prepare_python \
  "${PROJ}" \
  "${journal_run_dir}" \
  "${config_path}" \
  "${OUTPUT_DIR}" \
  "sft_phase_a" \
  "${python_bin}" \
  "RESUME_CKPT=${RESUME_CKPT:-}"

log "========================================================"
log "SFT Phase A job attempt (SLURM_JOB_ID=${SLURM_JOB_ID:-local})"
log "RESUME_CKPT=${RESUME_CKPT:-<unset>}"
log "Journal dir: ${journal_run_dir}"
log "========================================================"

client_exit=0
{
  if [ -f "$MERGED_PATH" ]; then
    log "Merge already complete: ${MERGED_PATH}"
    exit 0
  fi

  if [ -n "${RESUME_CKPT:-}" ]; then
    if [ ! -d "$RESUME_CKPT" ]; then
      log "ERROR: RESUME_CKPT set but path does not exist: $RESUME_CKPT"
      exit 1
    fi
    if [ ! -f "$RESUME_CKPT/adapter_config.json" ]; then
      log "ERROR: RESUME_CKPT missing adapter_config.json: $RESUME_CKPT"
      exit 1
    fi
    log "Will resume SFT from: $RESUME_CKPT"
  elif [ -f "$SFT_DONE" ]; then
    log "SFT done marker present; job will run merge only"
  else
    log "Fresh SFT start (no checkpoint resume)"
  fi

  export RESUME_CKPT="${RESUME_CKPT:-}"
  bash "${PROJ}/scripts/androidcontrol/sft/run_sft_2epoch_l20.sh"

  if [ -f "$MERGED_PATH" ]; then
    log "Phase A job SUCCESS: $MERGED_PATH"
  else
    log "ERROR: job finished but merge output missing: $MERGED_PATH"
    exit 1
  fi
} || client_exit=$?

final_status="completed"
if [[ ${client_exit} -ne 0 ]]; then
  final_status="failed"
else
  journal_promote_if_success "${PROJ}" "${client_exit}"
  journal_run_dir="${JOURNAL_RUN_DIR}"
fi

journal_finalize_status "${PROJ}" "${journal_run_dir}" "${final_status}" "${python_bin}"
exit "${client_exit}"
