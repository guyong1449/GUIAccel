#!/usr/bin/env bash
# 2-minute heart monitor for AndroidControl eval SLURM jobs.
# Usage: heart_monitor_eval.sh JOB_ID [JOB_ID...] [--log PATH] [--journal-dir PATH]
set -uo pipefail

INTERVAL_SEC=120
LOG=""
JOURNAL_DIR=""
JOBS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    --journal-dir) JOURNAL_DIR="$2"; shift 2 ;;
    --interval) INTERVAL_SEC="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 JOB_ID [JOB_ID...] [--log PATH] [--journal-dir PATH] [--interval SEC]"
      exit 0
      ;;
    *)
      JOBS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "No job IDs provided." >&2
  exit 1
fi

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SKILLREUSE_SEARCH="${_SCRIPT_DIR}"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    # shellcheck disable=SC1090
    source "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh"
    skillreuse_init_from "${BASH_SOURCE[0]}"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done

if [[ -z "${LOG}" ]]; then
  LOG="${REPO_ROOT}/training_journal/heart_monitor_eval.log"
fi
mkdir -p "$(dirname "${LOG}")"

if [[ -z "${JOURNAL_DIR}" ]]; then
  JOURNAL_DIR="${REPO_ROOT}/training_journal/.tentative/run_20260702_011954"
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

all_done() {
  local jid state
  for jid in "${JOBS[@]}"; do
    if squeue -j "${jid}" --noheader 2>/dev/null | grep -q .; then
      return 1
    fi
    state="$(sacct -j "${jid}" --format=State -n -P 2>/dev/null | head -1 | cut -d'|' -f1)"
    case "${state}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

log "Heart monitor started (PID $$, interval=${INTERVAL_SEC}s)"
log "Jobs: ${JOBS[*]}"
log "Journal: ${JOURNAL_DIR}"
log "Log: ${LOG}"

while true; do
  log "=== HEART CHECK ==="
  squeue -j "$(IFS=,; echo "${JOBS[*]}")" -o "%.10i %.20j %.10P %.8T %.12M %R" 2>/dev/null | tee -a "${LOG}" || true

  if [[ -f "${JOURNAL_DIR}/slurm.out" ]]; then
    partial="$(grep '^\[Partial metrics @' "${JOURNAL_DIR}/slurm.out" 2>/dev/null | tail -3 || true)"
    if [[ -n "${partial}" ]]; then
      log "Latest partial metrics:"
      while IFS= read -r line; do
        log "  ${line}"
      done <<< "${partial}"
    else
      log "No [Partial metrics @] lines yet (model load or early inference)."
      status_line="$(grep -E '^\[.*\] \[eval\]|^Loading checkpoint|^Starting ' "${JOURNAL_DIR}/slurm.out" 2>/dev/null | tail -1 || true)"
      if [[ -n "${status_line}" ]]; then
        log "Latest: ${status_line}"
      fi
    fi
    err_tail="$(grep -iE 'error|traceback|failed|oom|cuda' "${JOURNAL_DIR}/slurm.err" 2>/dev/null | tail -2 || true)"
    if [[ -n "${err_tail}" ]]; then
      log "WARNING stderr:"
      log "${err_tail}"
    fi
  else
    log "slurm.out not found at ${JOURNAL_DIR}"
  fi

  if all_done; then
    log "All monitored jobs finished."
    for jid in "${JOBS[@]}"; do
      sacct -j "${jid}" --format=JobID,JobName,State,ExitCode,Elapsed -n 2>/dev/null | tee -a "${LOG}" || true
    done
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
