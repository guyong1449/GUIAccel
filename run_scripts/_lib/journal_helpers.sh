#!/usr/bin/env bash
# Shared training_journal helpers for SkillReuse run_scripts.

journal_cst_date() { date -d "+12 hours" "$@"; }

journal_stamp_paths() {
  local repo_root="$1"
  JOURNAL_RUN_STAMP="$(journal_cst_date '+%Y%m%d_%H%M%S')"
  JOURNAL_RUN_DATE="$(journal_cst_date '+%Y_%m_%d')"
  JOURNAL_TENTATIVE_DIR="${repo_root}/training_journal/.tentative/run_${JOURNAL_RUN_STAMP}"
  JOURNAL_FINAL_DIR="${repo_root}/training_journal/${JOURNAL_RUN_DATE}/run_${JOURNAL_RUN_STAMP}"
}

journal_run_setup() {
  local repo_root="$1"
  local journal_run_dir="${2:-${JOURNAL_RUN_DIR:-}}"
  local run_date="${3:-${JOURNAL_RUN_DATE:-$(journal_cst_date '+%Y_%m_%d')}}"
  local run_stamp="${4:-${JOURNAL_RUN_STAMP:-$(journal_cst_date '+%Y%m%d_%H%M%S')}}"

  if [[ -z "${journal_run_dir}" ]]; then
    journal_stamp_paths "${repo_root}"
    journal_run_dir="${JOURNAL_TENTATIVE_DIR}"
  fi
  mkdir -p "${journal_run_dir}"

  JOURNAL_RUN_DIR="${journal_run_dir}"
  JOURNAL_RUN_DATE="${run_date}"
  JOURNAL_RUN_STAMP="${run_stamp}"
  export JOURNAL_RUN_DIR JOURNAL_RUN_DATE JOURNAL_RUN_STAMP

  exec > >(tee -a "${journal_run_dir}/terminal.log") 2>&1
  journal_register_failure_trap "${repo_root}" "${journal_run_dir}"
}

journal_register_failure_trap() {
  local repo_root="$1"
  local journal_run_dir="$2"

  cleanup_on_failure() {
    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
      return 0
    fi
    if [[ -d "${journal_run_dir}" && "${journal_run_dir}" == *"/.tentative/"* ]]; then
      local aborted_dir="${repo_root}/training_journal/.aborted/$(basename "${journal_run_dir}")"
      mkdir -p "${aborted_dir}"
      for name in slurm.out slurm.err terminal.log config_snapshot.json pointers.json run_summary.md; do
        if [[ -f "${journal_run_dir}/${name}" ]]; then
          cp -f "${journal_run_dir}/${name}" "${aborted_dir}/${name}"
        fi
      done
      rm -rf "${journal_run_dir}"
    fi
    return "${exit_code}"
  }
  trap cleanup_on_failure EXIT
}

journal_prepare_python() {
  local repo_root="$1"
  local journal_run_dir="$2"
  local config_path="$3"
  local output_dir="$4"
  local mode="$5"
  local python_bin="$6"
  local comment="${7:-}"

  "${python_bin}" -c "
import sys
sys.path.insert(0, '${repo_root}')
from guiaccel.journal import prepare_journal
from pathlib import Path
prepare_journal(
    Path('${journal_run_dir}'),
    config_path='${config_path}',
    output_dir='${output_dir}',
    mode='${mode}',
    comment='${comment}',
)
"
}

journal_promote_if_success() {
  local repo_root="$1"
  local exit_code="$2"

  if [[ ${exit_code} -ne 0 ]]; then
    return 0
  fi
  if [[ "${JOURNAL_RUN_DIR}" != *"/.tentative/"* ]]; then
    return 0
  fi

  local final_dir="${repo_root}/training_journal/${JOURNAL_RUN_DATE}/run_${JOURNAL_RUN_STAMP}"
  mkdir -p "$(dirname "${final_dir}")"
  mv "${JOURNAL_RUN_DIR}" "${final_dir}"
  JOURNAL_RUN_DIR="${final_dir}"
  export JOURNAL_RUN_DIR="${final_dir}"
  trap - EXIT
  echo "[$(journal_cst_date '+%F %T')] Journal promoted to: ${final_dir}"
}

journal_finalize_status() {
  local repo_root="$1"
  local journal_run_dir="$2"
  local status="$3"
  local python_bin="$4"

  "${python_bin}" -c "
import sys
sys.path.insert(0, '${repo_root}')
from guiaccel.journal import update_status
from pathlib import Path
update_status(Path('${journal_run_dir}'), '${status}')
"
}

journal_submit_slurm() {
  local repo_root="$1"
  local script_path="$2"
  local job_name="$3"
  local gpus="$4"
  local partition="$5"
  local account="$6"
  local time_limit="$7"
  local cpus="$8"
  local mem="$9"
  shift 9
  local -a extra_export=("$@")

  journal_stamp_paths "${repo_root}"
  mkdir -p "${JOURNAL_TENTATIVE_DIR}"

  local -a export_vars=(
    "ALL"
    "SKILLREUSE_REPO_ROOT=${repo_root}"
    "JOURNAL_RUN_DIR=${JOURNAL_TENTATIVE_DIR}"
    "JOURNAL_RUN_DATE=${JOURNAL_RUN_DATE}"
    "JOURNAL_RUN_STAMP=${JOURNAL_RUN_STAMP}"
  )
  export_vars+=("${extra_export[@]}")

  local -a sbatch_cmd=(
    sbatch
    --job-name "${job_name}"
    --nodes 1
    --ntasks 1
    --gres "gpu:${gpus}"
    --cpus-per-task "${cpus}"
    --mem "${mem}"
    --time "${time_limit}"
    --chdir "${repo_root}"
    --output "${JOURNAL_TENTATIVE_DIR}/slurm.out"
    --error "${JOURNAL_TENTATIVE_DIR}/slurm.err"
    --export "$(IFS=,; echo "${export_vars[*]}")"
  )
  if [[ -n "${account}" ]]; then
    sbatch_cmd+=(--account "${account}")
  fi
  if [[ -n "${partition}" ]]; then
    sbatch_cmd+=(--partition "${partition}")
  fi
  sbatch_cmd+=("${script_path}")

  local submit_output
  submit_output="$("${sbatch_cmd[@]}")"
  echo "${submit_output}"
  echo "Journal dir: ${JOURNAL_TENTATIVE_DIR}"
  echo "Promoted run dir: ${JOURNAL_FINAL_DIR}"
}
