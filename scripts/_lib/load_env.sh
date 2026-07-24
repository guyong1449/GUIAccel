#!/usr/bin/env bash
# Source GUIAccel .env from the repository root.
#
# Usage (from any repo script):
#   guiaccel_source_init_from_caller "${BASH_SOURCE[0]}"
#   # sets REPO_ROOT, SCRIPT_DIR, GUIACCEL_CONDA_PREFIX, etc.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_LIB_DIR}/repo_root.sh"

guiaccel_load_env() {
  local repo_root="${1:-}"
  if [[ -z "${repo_root}" ]]; then
    repo_root="$(guiaccel_repo_root_from "${_LIB_DIR}")"
  fi

  export GUIACCEL_REPO_ROOT="${GUIACCEL_REPO_ROOT:-${repo_root}}"

  if [[ -f "${GUIACCEL_REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${GUIACCEL_REPO_ROOT}/.env"
    set +a
  fi

  export GUIACCEL_REPO_ROOT="${GUIACCEL_REPO_ROOT:-${repo_root}}"
  export GUIACCEL_CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX:-/dkucc/home/rw335/.conda/envs/skillreuse-fa2}"

  # Backward-compat aliases for scripts still referencing SKILLREUSE_*
  export SKILLREUSE_REPO_ROOT="${GUIACCEL_REPO_ROOT}"
  export SKILLREUSE_CONDA_PREFIX="${GUIACCEL_CONDA_PREFIX}"
}

# Backward-compat alias
skillreuse_load_env() { guiaccel_load_env "$@"; }
