#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config_path="${MSD_CONFIG:-${repo_root}/configs/androidcontrol/msd_qwen2vl.json}"
conda_env="${MSD_CONDA_ENV:-msd-androidcontrol}"
partition="${MSD_PARTITION:-common-gpu}"
gpus="${MSD_GPUS:-1}"
time_limit="${MSD_TIME:-04:00:00}"
step_limit="${MSD_STEP_LIMIT:-16}"
instruction_mode="${MSD_INSTRUCTION_MODE:-both}"
baseline="${MSD_BASELINE:-false}"
dry_run="${MSD_DRY_RUN:-false}"

run_stamp="$(date '+%Y%m%d_%H%M%S')"
output_root="${MSD_OUTPUT_ROOT:-/work/rw335/GUIAccel/outputs/msd_androidcontrol}"
output_dir="${MSD_OUTPUT_DIR:-${output_root}/run_${run_stamp}}"
log_dir="${repo_root}/logs/msd_androidcontrol/eval_msd_qwen2vl_slurm"
mkdir -p "${log_dir}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  gres="gpu:a40:${gpus}"
  if [[ "${partition}" == "l20-gpu" ]]; then
    gres="gpu:l20:${gpus}"
  elif [[ "${partition}" == "h20-gpu" ]]; then
    gres="gpu:h20:${gpus}"
  fi
  submit_cmd=(
    sbatch
    --job-name msd-androidcontrol
    --partition "${partition}"
    --gres "${gres}"
    --cpus-per-task 8
    --mem 80G
    --time "${time_limit}"
    --chdir "${repo_root}"
    --output "${log_dir}/slurm-%j.out"
    --error "${log_dir}/slurm-%j.err"
    --export "ALL,MSD_CONFIG=${config_path},MSD_CONDA_ENV=${conda_env},MSD_OUTPUT_DIR=${output_dir},MSD_STEP_LIMIT=${step_limit},MSD_INSTRUCTION_MODE=${instruction_mode},MSD_BASELINE=${baseline}"
    "$0"
  )
  if [[ "${dry_run}" == "true" ]]; then
    printf ' %q' "${submit_cmd[@]}"
    printf '\n'
    exit 0
  fi
  mkdir -p "${output_root}"
  "${submit_cmd[@]}"
  exit 0
fi

eval "$(conda shell.bash hook)"
conda activate "${conda_env}"

args=(
  python "${repo_root}/scripts/androidcontrol/eval_msd.py"
  --config "${config_path}"
  --output-dir "${output_dir}"
  --split test
  --instruction-mode "${instruction_mode}"
  --step-limit "${step_limit}"
)
if [[ "${baseline}" == "true" ]]; then
  args+=(--baseline)
fi

echo "Launching: ${args[*]}"
"${args[@]}"
