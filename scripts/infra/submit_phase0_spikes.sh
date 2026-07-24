#!/usr/bin/env bash
# Submit Phase 0 runtime spikes via SLURM.

set -euo pipefail

# shellcheck disable=SC1091
_SKILLREUSE_SEARCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    # shellcheck disable=SC1090
    source "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh"
    skillreuse_init_from "${BASH_SOURCE[0]}"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done
PYTHON="${SKILLREUSE_CONDA_PREFIX:?Set SKILLREUSE_CONDA_PREFIX in .env}/bin/python"

mkdir -p "${REPO_ROOT}/logs"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_lib/submit_gpu_job.sh"

submit_gpu_job "phase0-spikes" "${REPO_ROOT}/logs" -- \
    "${PYTHON}" "${REPO_ROOT}/scripts/infra/phase0_runtime_spikes.py"
