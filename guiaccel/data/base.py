"""Shared dataset loader helpers."""

from __future__ import annotations

import json
from pathlib import Path

from guiaccel.utils.project_memory import get_project_memory_path


class BaseDatasetLoader:
    """Base helper for manifest-backed dataset loaders."""

    default_manifest_key: str

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        if manifest_path is not None:
            resolved_manifest_path = Path(manifest_path)
        else:
            try:
                resolved_manifest_path = get_project_memory_path(self.default_manifest_key)
            except KeyError as exc:
                raise ValueError(
                    f"A dataset manifest path is required for {type(self).__name__}. "
                    "Pass manifest_path explicitly or set the corresponding "
                    "SKILLREUSE_*_DATASET_MANIFEST environment variable."
                ) from exc
        self.manifest_path = resolved_manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.dataset_root = Path(self.manifest["prepared_path"]).resolve()
