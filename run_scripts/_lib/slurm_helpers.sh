#!/usr/bin/env bash
# Shared Slurm submission helpers for SkillReuse run_scripts.

DEFAULT_SLURM_TIME="7-00:00:00"
DEFAULT_H20_PARTITION="common-gpu"
DEFAULT_L20_PARTITION="l20-gpu"

should_use_gpu_partition_fallback() {
  local is_smoke="${1:-false}"
  local gpus="${2:-1}"
  local partition_explicit="${3:-false}"

  [[ "${is_smoke}" != "true" ]] \
    && [[ "${gpus}" -eq 4 ]] \
    && [[ "${partition_explicit}" != "true" ]]
}

# Map partition name to GPU type for GRES (l20 or h20).
slurm_gpu_type_for_partition() {
  local partition="${1:-}"
  case "${partition}" in
    l20-gpu|l20) printf 'l20' ;;
    h20-gpu|h20) printf 'h20' ;;
    *) printf '' ;;
  esac
}

# Build --gres string for partition and GPU count.
# l20-gpu / h20-gpu (and short names) require gpu:l20:N or gpu:h20:N on DKUCC.
slurm_gres_string() {
  local partition="${1:-}"
  local gpus="${2:-1}"
  local gpu_type
  gpu_type="$(slurm_gpu_type_for_partition "${partition}")"
  if [[ -n "${gpu_type}" ]]; then
    printf 'gpu:%s:%s' "${gpu_type}" "${gpus}"
  else
    printf 'gpu:%s' "${gpus}"
  fi
}

# Append --partition and partition-specific --gres to a copy of sbatch_cmd.
build_sbatch_attempt() {
  local cmd_name="$1"
  local partition="$2"
  local gpus="$3"
  local out_name="$4"
  local -n cmd_ref="${cmd_name}"
  local -n out_ref="${out_name}"
  local -a built=("${cmd_ref[@]}")
  local script_path="${built[-1]}"
  built=("${built[@]:0:$((${#built[@]} - 1))}")

  if [[ -n "${partition}" ]]; then
    built+=(--partition "${partition}")
  fi
  built+=(--gres "$(slurm_gres_string "${partition}" "${gpus}")")
  built+=("${script_path}")

  out_ref=("${built[@]}")
}

# submit_sbatch_with_gpu_fallback SBATCH_CMD_VAR is_smoke gpus partition partition_explicit dry_run
# SBATCH_CMD_VAR: name of array variable (already built, without --partition or --gres)
submit_sbatch_with_gpu_fallback() {
  local cmd_name="$1"
  local is_smoke="${2:-false}"
  local gpus="${3:-1}"
  local partition="${4:-}"
  local partition_explicit="${5:-false}"
  local dry_run="${6:-false}"

  local partitions=()

  if should_use_gpu_partition_fallback "${is_smoke}" "${gpus}" "${partition_explicit}"; then
    partitions=("${DEFAULT_H20_PARTITION}" "${DEFAULT_L20_PARTITION}")
  elif [[ -n "${partition}" ]]; then
    partitions=("${partition}")
  else
    partitions=("")
  fi

  if [[ "${dry_run}" == "true" ]]; then
    local -a attempt=()
    local primary="${partitions[0]}"
    build_sbatch_attempt "${cmd_name}" "${primary}" "${gpus}" attempt
    if should_use_gpu_partition_fallback "${is_smoke}" "${gpus}" "${partition_explicit}"; then
      printf 'Dry run sbatch command (primary %s, fallback %s):\n' \
        "${DEFAULT_H20_PARTITION}" "${DEFAULT_L20_PARTITION}" >&2
    elif [[ -n "${partition}" ]]; then
      printf 'Dry run sbatch command:\n' >&2
    else
      printf 'Dry run sbatch command:\n' >&2
    fi
    printf '  %q' "${attempt[@]}" >&2
    printf '\n' >&2
    return 0
  fi

  local submit_output=""
  local last_error=""
  local idx=0
  for p in "${partitions[@]}"; do
    local -a attempt=()
    build_sbatch_attempt "${cmd_name}" "${p}" "${gpus}" attempt
    if submit_output="$("${attempt[@]}" 2>&1)"; then
      echo "${submit_output}"
      if (( idx > 0 )) && [[ -n "${p}" ]]; then
        echo "NOTE: submitted on fallback partition ${p}" >&2
      fi
      return 0
    fi
    last_error="${submit_output}"
    if [[ -n "${p}" ]]; then
      echo "WARNING: sbatch failed on partition ${p:-<default>}: ${last_error}" >&2
    fi
    idx=$((idx + 1))
  done

  echo "${last_error}" >&2
  return 1
}
