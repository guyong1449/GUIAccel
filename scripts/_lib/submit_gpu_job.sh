#!/usr/bin/env bash
# Source this file to submit GPU jobs with repo SLURM defaults.
# Usage:
#   source scripts/_lib/submit_gpu_job.sh
#   submit_gpu_job JOB_NAME LOG_DIR -- command args...

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/repo_root.sh"
source "${_SCRIPT_DIR}/slurm_defaults.sh"

# Opt-in faithful dry-run. Disabled unless SLURM_DRY_RUN is set to a truthy value,
# so existing callers that never set it are completely unaffected.
_slurm_dry_run_enabled() {
    case "${SLURM_DRY_RUN:-}" in
        ""|0|false|False|FALSE|no|No|NO|off|Off|OFF) return 1 ;;
        *) return 0 ;;
    esac
}

submit_gpu_job() {
    local job_name="$1"
    local log_dir="$2"
    shift 2
    [[ "$1" == "--" ]] && shift
    local -a cmd=("$@")

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        exec "${cmd[@]}"
    fi

    local wrap_cmd=""
    local arg
    for arg in "${cmd[@]}"; do
        wrap_cmd+="$(printf '%q ' "${arg}")"
    done
    wrap_cmd="${wrap_cmd% }"

    local -a sbatch_args=(
        --gres="gpu:${SLURM_GPUS:-${DEFAULT_SLURM_GPUS}}"
        --time="${SLURM_TIME:-${DEFAULT_SLURM_TIME}}"
        --cpus-per-task="${SLURM_CPUS:-${DEFAULT_SLURM_CPUS}}"
        --mem="${SLURM_MEM:-${DEFAULT_SLURM_MEM}}"
        --job-name="${job_name}"
        --output="${log_dir}/${job_name}_%j.log"
        --error="${log_dir}/${job_name}_%j.log"
        --chdir="$(guiaccel_repo_root_from "${_SCRIPT_DIR}")"
        --wrap="${wrap_cmd}"
    )

    if _slurm_dry_run_enabled; then
        local dry_partition="${SLURM_PARTITION:-${DEFAULT_SLURM_PARTITION}}"
        echo "[dry-run] SLURM_DRY_RUN is set; NOT submitting any SLURM job." >&2
        echo "[dry-run] Command that WOULD be submitted:" >&2
        printf 'sbatch --partition=%q' "${dry_partition}"
        printf ' %q' "${sbatch_args[@]}"
        printf '\n'
        return 0
    fi

    mkdir -p "${log_dir}"

    submit_sbatch_gpu_with_partition_fallback \
        "${SLURM_PARTITION:-}" \
        "${sbatch_args[@]}"
}
