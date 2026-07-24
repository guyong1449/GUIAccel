#!/bin/bash
# SLURM job auto-resubmission monitor
# Fixes: atomic state tracking, Pending/Running distinction, PID lock,
#        adaptive intervals, Pending escalation, defensive guards

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
_SKILLREUSE_SEARCH="${SCRIPT_DIR}"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    # shellcheck disable=SC1090
    source "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh"
    skillreuse_init_from "${BASH_SOURCE[0]}"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done
PROJECT_ROOT="${REPO_ROOT}"
SUBMIT_SCRIPT="${REPO_ROOT}/run_scripts/pipeline/submit_slurm.sh"
# GUIAccel adaptation: reference guiaccel package
LOG="${PROJECT_ROOT}/training_journal/monitor_log.txt"
STATE_FILE="${PROJECT_ROOT}/training_journal/.monitor_state"
LOCK_FILE="${PROJECT_ROOT}/training_journal/.monitor.lock"
MAX_RESUBMITS=300
RESUBMIT_COUNT=0
PENDING_COUNT=0

# --- PID lock: prevent duplicate monitor instances ---
if [ -f "$LOCK_FILE" ]; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Another monitor is running (PID $EXISTING_PID). Exiting." >&2
        exit 1
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# --- State file helpers ---
save_state() {
    local job_id="$1"
    local run_dir="$2"
    echo "${job_id}|${run_dir}" > "$STATE_FILE"
}

load_state() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    fi
}

# Find the latest job ID and run directory (fallback only)
find_latest_run() {
    LATEST_RUN=""
    LATEST_JOB=""

    # Check .tentative first
    for d in $(ls -td ${PROJECT_ROOT}/training_journal/.tentative/run_* 2>/dev/null); do
        if [ -d "$d" ]; then
            LATEST_RUN="$d"
            if [ -f "$d/slurm.out" ]; then
                LATEST_JOB=$(grep -oP 'Starting GUIAccel job \K\d+' "$d/slurm.out" 2>/dev/null | head -1)
            fi
            break
        fi
    done

    # If not found in .tentative, check date directories
    if [ -z "$LATEST_RUN" ]; then
        for dir in $(ls -td ${PROJECT_ROOT}/training_journal/2026_*/run_* 2>/dev/null); do
            if [ -d "$dir" ]; then
                LATEST_RUN="$dir"
                if [ -f "$dir/slurm.out" ]; then
                    LATEST_JOB=$(grep -oP 'Starting GUIAccel job \K\d+' "$dir/slurm.out" 2>/dev/null | head -1)
                fi
                break
            fi
        done
    fi

    echo "$LATEST_RUN|$LATEST_JOB"
}

# --- Initial state: prefer saved state, fall back to filesystem scan ---
SAVED=$(load_state)
if [ -n "$SAVED" ]; then
    RUN_DIR=$(echo "$SAVED" | cut -d'|' -f2)
    JOB_ID=$(echo "$SAVED" | cut -d'|' -f1)
    if [ ! -d "$RUN_DIR" ]; then
        # Saved state is stale, fall back
        INITIAL=$(find_latest_run)
        RUN_DIR=$(echo "$INITIAL" | cut -d'|' -f1)
        JOB_ID=$(echo "$INITIAL" | cut -d'|' -f2)
    fi
else
    INITIAL=$(find_latest_run)
    RUN_DIR=$(echo "$INITIAL" | cut -d'|' -f1)
    JOB_ID=$(echo "$INITIAL" | cut -d'|' -f2)
fi

echo "[$(date)] Monitor started (PID $$)" | tee -a "$LOG"
echo "[$(date)] Initial run dir: $RUN_DIR" | tee -a "$LOG"
echo "[$(date)] Initial job ID: $JOB_ID" | tee -a "$LOG"

while [ $RESUBMIT_COUNT -lt $MAX_RESUBMITS ]; do
    # Adaptive check interval: shorter for Pending jobs
    if [ "${STATE:-}" = "PD" ]; then
        sleep 300   # 5 min for Pending
    else
        sleep 600   # 10 min for Running
    fi

    echo "" | tee -a "$LOG"
    echo "=== CHECK: $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

    # Refresh RUN_DIR if it became stale (promoted .tentative → date dir)
    if [ -n "$RUN_DIR" ] && [ ! -d "$RUN_DIR" ]; then
        echo "RUN_DIR $RUN_DIR no longer exists. Re-scanning..." | tee -a "$LOG"
        SCAN=$(find_latest_run)
        NEW_SCAN_DIR=$(echo "$SCAN" | cut -d'|' -f1)
        if [ -n "$NEW_SCAN_DIR" ] && [ -d "$NEW_SCAN_DIR" ]; then
            RUN_DIR="$NEW_SCAN_DIR"
            save_state "$JOB_ID" "$RUN_DIR"
        fi
    fi

    # Guard: empty JOB_ID
    if [ -z "$JOB_ID" ]; then
        echo "WARNING: No job ID found. Attempting recovery..." | tee -a "$LOG"
        INITIAL=$(find_latest_run)
        RUN_DIR=$(echo "$INITIAL" | cut -d'|' -f1)
        JOB_ID=$(echo "$INITIAL" | cut -d'|' -f2)
        if [ -z "$JOB_ID" ]; then
            echo "WARNING: Still no job ID. Waiting..." | tee -a "$LOG"
            sleep 300
            continue
        fi
    fi

    # Check job status via squeue (robust parsing)
    QUEUE_STATUS=$(squeue -j "$JOB_ID" --noheader --parsable2 2>/dev/null)
    SQUEUE_EXIT=$?

    if [ $SQUEUE_EXIT -ne 0 ]; then
        echo "WARNING: squeue failed (SLURM controller issue?). Skipping check." | tee -a "$LOG"
        sleep 60
        continue
    fi

    if [ -n "$QUEUE_STATUS" ]; then
        STATE=$(echo "$QUEUE_STATUS" | cut -d'|' -f5)
        ELAPSED=$(echo "$QUEUE_STATUS" | cut -d'|' -f6)

        if [ "$STATE" = "PD" ]; then
            PENDING_COUNT=$((PENDING_COUNT + 1))
            PENDING_MIN=$((PENDING_COUNT * 5))
            echo "Job $JOB_ID PENDING, elapsed: $ELAPSED, pending: ${PENDING_MIN}min" | tee -a "$LOG"
            # Escalation: cancel + resubmit after 3 hours stuck in Pending
            if [ $PENDING_COUNT -ge 36 ]; then
                echo "Job stuck Pending ${PENDING_MIN}+ min. Canceling and resubmitting..." | tee -a "$LOG"
                scancel "$JOB_ID" 2>/dev/null
                # Fall through to resubmit logic below
            else
                # Get progress if slurm.out exists
                if [ -f "$RUN_DIR/slurm.out" ]; then
                    COMPLETED=$(grep -oP '"completed_examples":\s*\d+' "$RUN_DIR/slurm.out" 2>/dev/null | tail -1 | grep -oP '\d+')
                    if [ -n "$COMPLETED" ]; then
                        echo "Progress: $COMPLETED completed_examples" | tee -a "$LOG"
                    fi
                fi
                continue
            fi
        else
            PENDING_COUNT=0
            echo "Job $JOB_ID RUNNING, elapsed: $ELAPSED" | tee -a "$LOG"

            # Get progress
            if [ -f "$RUN_DIR/slurm.out" ]; then
                COMPLETED=$(grep -oP '"completed_examples":\s*\d+' "$RUN_DIR/slurm.out" 2>/dev/null | tail -1 | grep -oP '\d+')
                if [ -n "$COMPLETED" ]; then
                    echo "Progress: $COMPLETED completed_examples" | tee -a "$LOG"
                fi
            fi
            continue
        fi
    fi

    # Job no longer in queue - check what happened
    echo "Job $JOB_ID no longer in queue. Checking outcome..." | tee -a "$LOG"

    if [ -f "$RUN_DIR/slurm.out" ]; then
        FINAL_COUNT=$(grep -oP '"completed_examples":\s*\d+' "$RUN_DIR/slurm.out" 2>/dev/null | tail -1 | grep -oP '\d+')
        echo "Final completed_examples: ${FINAL_COUNT:-unknown}" | tee -a "$LOG"

        TAIL=$(tail -30 "$RUN_DIR/slurm.out" 2>/dev/null)

        # Check for successful completion
        if echo "$TAIL" | grep -qi "pipeline complete\|discovery complete\|successfully finished\|all examples processed"; then
            echo "[$(date)] SUCCESS! Pipeline completed." | tee -a "$LOG"
            echo "Final results:" | tee -a "$LOG"
            tail -5 "$RUN_DIR/slurm.out" | tee -a "$LOG"
            exit 0
        fi

        # Check for crash patterns
        ERROR_MSG=$(echo "$TAIL" | grep -i "error\|crash\|failed\|exception\|EngineCore\|CUDA\|OOM" | tail -3)
        echo "Job failed. Error:" | tee -a "$LOG"
        echo "$ERROR_MSG" | tee -a "$LOG"
    else
        echo "WARNING: slurm.out not found at $RUN_DIR" | tee -a "$LOG"
    fi

    # Resubmit
    RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1))
    PENDING_COUNT=0
    STATE=""
    echo "Resubmission #$RESUBMIT_COUNT of $MAX_RESUBMITS..." | tee -a "$LOG"

    # Move old run to .aborted
    if [ -d "$RUN_DIR" ]; then
        PARENT_DIR=$(dirname "$RUN_DIR")
        ABORTED_DIR="$PARENT_DIR/.aborted"
        mkdir -p "$ABORTED_DIR"
        mv "$RUN_DIR" "$ABORTED_DIR/" 2>/dev/null
        echo "Moved $RUN_DIR to $ABORTED_DIR/" | tee -a "$LOG"
    fi

    # Resubmit
    cd "${REPO_ROOT}"
    SUBMIT_OUTPUT=$(bash "${SUBMIT_SCRIPT}" --mode discovery --gpus 4 --resume --partition common-gpu --mem 160G 2>&1)
    echo "$SUBMIT_OUTPUT" | tee -a "$LOG"

    # Extract new job ID
    NEW_JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    if [ -n "$NEW_JOB_ID" ]; then
        JOB_ID=$NEW_JOB_ID
        echo "New job ID: $JOB_ID" | tee -a "$LOG"
    fi

    # Wait for run directory to be created, then find it precisely
    sleep 10

    # Parse tentative dir from submit output (more reliable than mtime scan)
    TENTATIVE_DIR=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Journal tentative dir: \K.*' | tail -1)
    if [ -n "$TENTATIVE_DIR" ] && [ -d "$TENTATIVE_DIR" ]; then
        RUN_DIR="$TENTATIVE_DIR"
    else
        # Fallback: scan filesystem
        NEW_RUN=$(find_latest_run)
        RUN_DIR=$(echo "$NEW_RUN" | cut -d'|' -f1)
    fi

    if [ -n "$RUN_DIR" ] && [ -n "$JOB_ID" ]; then
        save_state "$JOB_ID" "$RUN_DIR"
        echo "Saved state: $JOB_ID|$RUN_DIR" | tee -a "$LOG"
    fi
done

echo "[$(date)] Reached maximum resubmissions ($MAX_RESUBMITS). Stopping." | tee -a "$LOG"
exit 1
