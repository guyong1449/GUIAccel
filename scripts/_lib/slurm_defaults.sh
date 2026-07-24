#!/usr/bin/env bash
# Shared SLURM defaults for GUIAccel GPU jobs (DKUCC partitions).

DEFAULT_SLURM_TIME="7-00:00:00"
DEFAULT_SLURM_PARTITION="common-gpu"
DEFAULT_SLURM_FALLBACK_PARTITION="common-gpu"
DEFAULT_SLURM_GPUS="4"
DEFAULT_SLURM_CPUS="16"
DEFAULT_SLURM_MEM="160G"

# GPU partitions — common-gpu primary, l20-gpu fallback.
SLURM_GPU_PARTITION_CANDIDATES=(
    "common-gpu"
    "l20-gpu"
)

_slurm_partition_uses_gpu_fallback() {
    local partition="${1:-}"
    if [[ -z "${partition}" ]]; then
        return 0
    fi
    local candidate
    for candidate in "${SLURM_GPU_PARTITION_CANDIDATES[@]}"; do
        if [[ "${partition}" == "${candidate}" ]]; then
            return 0
        fi
    done
    return 1
}

_submit_sbatch_on_partition() {
    local partition="$1"
    shift
    local -a sbatch_args=("$@")

    local output
    if ! output="$(sbatch --partition="${partition}" "${sbatch_args[@]}" 2>&1)"; then
        echo "[${partition}] sbatch rejected: ${output}" >&2
        return 1
    fi

    local job_id="${output##* }"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[${partition}] unexpected sbatch output: ${output}" >&2
        return 1
    fi

    sleep 3
    local attempt state reason
    for attempt in 1 2 3 4 5 6; do
        state="$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null | head -n1 || true)"
        if [[ -z "${state}" ]]; then
            printf '%s\n' "${output}"
            return 0
        fi
        if [[ "${state}" == "RUNNING" ]] || [[ "${state}" == "COMPLETING" ]]; then
            echo "[${partition}] job ${job_id} ${state}" >&2
            printf '%s\n' "${output}"
            return 0
        fi
        reason="$(squeue -j "${job_id}" -h -o '%R' 2>/dev/null | head -n1 || true)"
        if [[ "${state}" == "PENDING" ]] && {
            [[ "${reason}" == *"Resources"* ]] \
            || [[ "${reason}" == *"ReqNodeNotAvail"* ]] \
            || [[ "${reason}" == *"PartitionNodeLimit"* ]] \
            || [[ "${reason}" == *"Nodes required"* ]]
        }; then
            if [[ "${attempt}" -lt 6 ]]; then
                sleep 5
                continue
            fi
            echo "[${partition}] job ${job_id} still pending (${reason}); cancel and try next partition" >&2
            scancel "${job_id}" 2>/dev/null || true
            sleep 2
            return 1
        fi
        printf '%s\n' "${output}"
        return 0
    done

    return 1
}

# Submit sbatch (without --partition in sbatch_args). Tries h20-gpu 4-GPU queue first,
# then l20-gpu when sbatch fails or the job stays pending for lack of resources.
# Pass an explicit non-h20/l20 partition to skip fallback.
submit_sbatch_gpu_with_partition_fallback() {
    local explicit_partition="${1:-}"
    shift
    local -a sbatch_args=("$@")

    local -a candidates=()
    if _slurm_partition_uses_gpu_fallback "${explicit_partition}"; then
        candidates=("${SLURM_GPU_PARTITION_CANDIDATES[@]}")
    else
        candidates=("${explicit_partition}")
    fi

    local partition output
    local -a tried=()
    for partition in "${candidates[@]}"; do
        local seen="false"
        local prev
        for prev in "${tried[@]}"; do
            if [[ "${prev}" == "${partition}" ]]; then
                seen="true"
                break
            fi
        done
        [[ "${seen}" == "true" ]] && continue
        tried+=("${partition}")

        echo "Trying sbatch on ${partition} (gpu:${SLURM_GPUS:-${DEFAULT_SLURM_GPUS:-?}})..." >&2
        if output="$(_submit_sbatch_on_partition "${partition}" "${sbatch_args[@]}")"; then
            printf '%s\n' "${output}"
            return 0
        fi
    done

    echo "ERROR: sbatch failed on all GPU partitions: ${tried[*]}" >&2
    return 1
}
