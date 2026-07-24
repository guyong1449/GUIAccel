#!/bin/bash
# =============================================================================
# sft_phase_a_supervisor.sh — Login-node auto-resume supervisor for Phase A SFT
#
# Submits sbatch jobs, waits for completion, resumes from latest checkpoint on
# failure. Exits when outputs/sft_merged/config.json exists.
#
# Start (do not run automatically from repo setup):
#   nohup bash run_scripts/sft/sft_phase_a_supervisor.sh >> logs/sft_phase_a_supervisor.log 2>&1 &
# =============================================================================
set -euo pipefail

PROJ=/dkucc/home/rw335/SkillReuse
cd "$PROJ"
# shellcheck disable=SC1091
source "${PROJ}/run_scripts/_lib/slurm_helpers.sh"
# shellcheck disable=SC1091
source "${PROJ}/run_scripts/_lib/journal_helpers.sh"
LOG="${PROJ}/logs/sft_phase_a_supervisor.log"
LOCK="${PROJ}/logs/sft_phase_a_supervisor.lock"
MERGED_PATH="${PROJ}/outputs/sft_merged/config.json"
SFT_DONE="${PROJ}/outputs/sft_output/.sft_done"
JOB_SCRIPT="${PROJ}/run_scripts/sft/sft_phase_a_job.sh"

MAX_RESUMES=40000
POLL_RUNNING=120
POLL_FAILURE=30
SLURM_TIME="${DEFAULT_SLURM_TIME}"
PARTITION=""
PARTITION_EXPLICIT="false"

mkdir -p "${PROJ}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&2
}

if [ -f "$LOCK" ]; then
    EXISTING_PID=$(cat "$LOCK" 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Another supervisor is running (PID $EXISTING_PID). Exiting." | tee -a "$LOG" >&2
        exit 1
    fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

resume_count=0

find_latest_checkpoint() {
    ls -d "${PROJ}"/outputs/sft_output/v*/checkpoint-* 2>/dev/null | sort -V | tail -1 || true
}

job_in_queue() {
    local jid="$1"
    squeue -j "$jid" --noheader 2>/dev/null | grep -q .
}

get_sacct_state() {
    local jid="$1"
    # First line after header; State may be COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.
    sacct -j "$jid" --parsable2 --noheader -n -o State,ExitCode 2>/dev/null | head -1 | cut -d'|' -f1
}

wait_for_job() {
    local jid="$1"
    while job_in_queue "$jid"; do
        log "Job $jid still in queue; sleeping ${POLL_RUNNING}s"
        sleep "$POLL_RUNNING"
    done
}

submit_phase_a_job() {
    local ckpt="${1:-}"
    journal_stamp_paths "${PROJ}"
    mkdir -p "${JOURNAL_TENTATIVE_DIR}"

    local -a export_vars=(
        "ALL"
        "JOURNAL_RUN_DIR=${JOURNAL_TENTATIVE_DIR}"
        "JOURNAL_RUN_DATE=${JOURNAL_RUN_DATE}"
        "JOURNAL_RUN_STAMP=${JOURNAL_RUN_STAMP}"
    )
    if [ -n "$ckpt" ]; then
        export_vars+=("RESUME_CKPT=${ckpt}")
        log "Submitting sbatch with RESUME_CKPT=$ckpt"
    elif [ -f "$SFT_DONE" ]; then
        log "Submitting sbatch for merge-only (.sft_done present)"
    else
        log "Submitting sbatch for fresh SFT"
    fi

    local sbatch_cmd=(
        sbatch
        --chdir="$PROJ"
        --job-name=sft_phase_a
        --gres=gpu:4
        --cpus-per-task=16
        --mem=256G
        --time="${SLURM_TIME}"
        --export="$(IFS=,; echo "${export_vars[*]}")"
        --output="${JOURNAL_TENTATIVE_DIR}/slurm.out"
        --error="${JOURNAL_TENTATIVE_DIR}/slurm.err"
        "$JOB_SCRIPT"
    )

    local submit_out
    if ! submit_out="$(submit_sbatch_with_gpu_fallback \
        sbatch_cmd "false" "4" "${PARTITION}" "${PARTITION_EXPLICIT}" "false")"; then
        log "ERROR: sbatch submission failed"
        exit 1
    fi

    local jid
    jid=$(echo "$submit_out" | awk '{print $NF}')
    if [ -z "$jid" ] || ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        log "ERROR: failed to parse sbatch job id from: $submit_out"
        exit 1
    fi
    log "$submit_out"
    log "Journal dir: ${JOURNAL_TENTATIVE_DIR}"
    log "Promoted run dir: ${JOURNAL_FINAL_DIR}"
    echo "$jid"
}

log "Supervisor started (PID $$, MAX_RESUMES=$MAX_RESUMES)"

while true; do
    if [ -f "$MERGED_PATH" ]; then
        log "SUCCESS: merge complete at $MERGED_PATH (resume_count=$resume_count)"
        exit 0
    fi

    ckpt=""
    if [ ! -f "$SFT_DONE" ]; then
        ckpt=$(find_latest_checkpoint)
        if [ -n "$ckpt" ]; then
            log "Latest checkpoint: $ckpt"
        else
            log "No checkpoint found; will start fresh SFT"
        fi
    else
        log ".sft_done present but merge missing; merge-only submission"
    fi

    job_id=$(submit_phase_a_job "$ckpt")
    log "Submitted job $job_id"

    wait_for_job "$job_id"

    # sacct may lag briefly after job leaves queue
    sleep 5
    state=$(get_sacct_state "$job_id")
    state="${state%%+*}"  # CANCELLED+ -> CANCELLED
    log "Job $job_id final sacct State=$state"

    if [ -f "$MERGED_PATH" ]; then
        log "SUCCESS: merge complete after job $job_id (resume_count=$resume_count)"
        exit 0
    fi

    case "$state" in
        COMPLETED)
            log "Job $job_id COMPLETED but Phase A not finished (no merged config); retrying"
            ;;
        FAILED|TIMEOUT|CANCELLED|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|"")
            log "Job $job_id ended with state=${state:-UNKNOWN}; will resume"
            ;;
        *)
            log "Job $job_id ended with state=$state; will resume"
            ;;
    esac

    resume_count=$((resume_count + 1))
    if [ "$resume_count" -ge "$MAX_RESUMES" ]; then
        log "ERROR: reached MAX_RESUMES=$MAX_RESUMES; giving up"
        exit 1
    fi

    log "Resume attempt $resume_count/$MAX_RESUMES; sleeping ${POLL_FAILURE}s before resubmit"
    sleep "$POLL_FAILURE"
done
