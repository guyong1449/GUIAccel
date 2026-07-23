"""Repository path helpers plus environment-based local asset resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_KEY_BY_LABEL = {
    "androidcontrol_dataset_manifest": "SKILLREUSE_ANDROIDCONTROL_DATASET_MANIFEST",
    "learngui_dataset_manifest": "SKILLREUSE_LEARNGUI_DATASET_MANIFEST",
    "base_model_checkpoint": "SKILLREUSE_BASE_MODEL_PATH",
}


def get_repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root by locating pyproject.toml and the package dir."""

    base = (start or Path(__file__)).resolve()
    search_start = base if base.is_dir() else base.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "skillreuse").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def slugify_label(label: str) -> str:
    """Normalize human-readable labels into lookup keys."""

    slug = label.strip().lower()
    slug = slug.replace("+", " plus ")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def load_project_memory_paths(memory_path: Path | None = None) -> dict[str, str]:
    """Load absolute local asset paths from environment variables."""

    del memory_path
    paths: dict[str, str] = {}
    for key, env_var in _ENV_KEY_BY_LABEL.items():
        value = os.environ.get(env_var)
        if not value:
            continue
        paths[key] = str(Path(value).expanduser().resolve())
    return paths


def get_project_memory_path(key: str, memory_path: Path | None = None) -> Path:
    """Resolve a required absolute local path by normalized key."""

    lookup = load_project_memory_paths(memory_path)
    normalized_key = slugify_label(key)
    if normalized_key not in lookup:
        env_var = _ENV_KEY_BY_LABEL.get(normalized_key)
        if env_var is None:
            raise KeyError(f"Missing configured path for key: {key}")
        raise KeyError(
            f"Missing configured path for key: {key}. "
            f"Set environment variable {env_var} or pass the path explicitly."
        )
    return Path(lookup[normalized_key])
