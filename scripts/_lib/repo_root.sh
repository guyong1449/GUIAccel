#!/usr/bin/env bash
# Resolve GUIAccel repository root by walking up to pyproject.toml + guiaccel/.

guiaccel_repo_root_from() {
  local dir="${1:-}"
  if [[ -z "${dir}" ]]; then
    echo "guiaccel_repo_root_from: missing start directory" >&2
    return 1
  fi
  dir="$(cd "${dir}" && pwd)"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/pyproject.toml" && -d "${dir}/guiaccel" ]]; then
      printf '%s' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  echo "Could not find GUIAccel repo root from ${1}" >&2
  return 1
}

# Backward-compat alias
skillreuse_repo_root_from() { guiaccel_repo_root_from "$@"; }

guiaccel_repo_root_from_script() {
  guiaccel_repo_root_from "$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
}
