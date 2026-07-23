#!/usr/bin/env bash
# Source SkillReuse .env from the repository root.
#
# Usage (from any repo script):
#   skillreuse_source_init_from_caller "${BASH_SOURCE[0]}"
#   # sets REPO_ROOT, SCRIPT_DIR, SKILLREUSE_CONDA_PREFIX, etc.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_LIB_DIR}/repo_root.sh"

skillreuse_load_env() {
  local repo_root="${1:-}"
  if [[ -z "${repo_root}" ]]; then
    repo_root="$(skillreuse_repo_root_from "${_LIB_DIR}")"
  fi

  export SKILLREUSE_REPO_ROOT="${SKILLREUSE_REPO_ROOT:-${repo_root}}"

  if [[ -f "${SKILLREUSE_REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${SKILLREUSE_REPO_ROOT}/.env"
    set +a
  fi

  export SKILLREUSE_REPO_ROOT="${SKILLREUSE_REPO_ROOT:-${repo_root}}"
  export SKILLREUSE_CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:-${SKILLREUSE_REPO_ROOT}/.conda/envs/skillreuse}"
}
