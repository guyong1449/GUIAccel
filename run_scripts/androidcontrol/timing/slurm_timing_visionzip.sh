#!/usr/bin/env bash
# Timing full eval — wrapper around eval_visionzip_slurm.sh (DivPrune-style).
set -Eeuo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SKILLREUSE_SEARCH="${_SCRIPT_DIR}"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    REPO_ROOT="$(cd "${_SKILLREUSE_SEARCH}" && pwd)"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done
exec "${_SCRIPT_DIR}/_exec_timing_eval.sh" eval_visionzip_slurm.sh \
  --job-name full-eval-visionzip-timing \
  --config "${REPO_ROOT}/configs/androidcontrol/visionzip/timing.json" \
  --output-dir "${REPO_ROOT}/outputs/timing_visionzip_eager" \
  --measure-e2e-latency \
  "$@"
