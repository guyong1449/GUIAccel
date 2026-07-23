"""Resolve SkillReuse repository root from any script location."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    anchor = (start or Path(__file__)).resolve()
    for candidate in (anchor, *anchor.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "guiaccel").is_dir():
            return candidate
    raise RuntimeError(f"Could not find SkillReuse repo root from {anchor}")


def ensure_repo_on_path(start: Path | None = None) -> Path:
    root = repo_root(start)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def bootstrap(start: Path | None = None) -> Path:
    """Import helper: add repo root to sys.path for guiaccel imports."""
    return ensure_repo_on_path(start)
