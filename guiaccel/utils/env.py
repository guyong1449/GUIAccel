"""Load repository-local .env overrides into os.environ."""

from __future__ import annotations

import os
from pathlib import Path

from guiaccel.utils.project_memory import get_repo_root


def load_repo_env(repo_root: Path | None = None) -> None:
    """Parse ``.env`` at the repo root and apply unset environment variables."""

    root = (repo_root or get_repo_root()).resolve()
    env_file = root / ".env"
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        value = os.path.expandvars(value)
        os.environ.setdefault(key, value)


__all__ = ["load_repo_env"]
