#!/usr/bin/env bash
# Shared launcher: preset timing eval via androidcontrol/eval/*_slurm.sh (DivPrune-style).
set -Eeuo pipefail

EVAL_SCRIPT="${1:?eval script name required}"
shift

_SKILLREUSE_SEARCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${_SKILLREUSE_SEARCH}" != "/" ]]; do
  if [[ -f "${_SKILLREUSE_SEARCH}/scripts/_lib/init.sh" ]]; then
    REPO_ROOT="$(cd "${_SKILLREUSE_SEARCH}" && pwd)"
    break
  fi
  _SKILLREUSE_SEARCH="$(dirname "${_SKILLREUSE_SEARCH}")"
done

exec "${REPO_ROOT}/run_scripts/androidcontrol/eval/${EVAL_SCRIPT}" "$@"
