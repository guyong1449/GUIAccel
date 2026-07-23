#!/usr/bin/env bash
# Resolve SkillReuse repository root by walking up to pyproject.toml + skillreuse/.

skillreuse_repo_root_from() {
  local dir="${1:-}"
  if [[ -z "${dir}" ]]; then
    echo "skillreuse_repo_root_from: missing start directory" >&2
    return 1
  fi
  dir="$(cd "${dir}" && pwd)"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/pyproject.toml" && -d "${dir}/skillreuse" ]]; then
      printf '%s' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  echo "Could not find SkillReuse repo root from ${1}" >&2
  return 1
}

skillreuse_repo_root_from_script() {
  skillreuse_repo_root_from "$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
}
