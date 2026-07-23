#!/usr/bin/env bash
# Shared bootstrap for SkillReuse shell scripts at any directory depth.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_LIB_DIR}/repo_root.sh"

skillreuse_init_from() {
  local caller="${1:?caller path required}"
  SCRIPT_DIR="$(cd "$(dirname "${caller}")" && pwd)"
  REPO_ROOT="$(skillreuse_repo_root_from "${SCRIPT_DIR}")"
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/_lib/load_env.sh"
  skillreuse_load_env "${REPO_ROOT}"
}

skillreuse_source_init_from_caller() {
  local caller="${1:?caller path required}"
  local search
  search="$(cd "$(dirname "${caller}")" && pwd)"
  while [[ "${search}" != "/" ]]; do
    if [[ -f "${search}/scripts/_lib/init.sh" ]]; then
      # shellcheck disable=SC1090
      source "${search}/scripts/_lib/init.sh"
      skillreuse_init_from "${caller}"
      return 0
    fi
    search="$(dirname "${search}")"
  done
  echo "Could not locate scripts/_lib/init.sh from ${caller}" >&2
  return 1
}
