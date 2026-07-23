"""Shared data structures for SkillReuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass
class ScreenshotAsset:
    """Lazy screenshot reference."""

    path: Path | None = None
    png_bytes: bytes | None = None
    width: int | None = None
    height: int | None = None
    source: str = "path"

    def read_bytes(self) -> bytes:
        if self.png_bytes is not None:
            return self.png_bytes
        if self.path is None:
            raise ValueError("ScreenshotAsset has neither path nor bytes.")
        return self.path.read_bytes()


@dataclass
class DatasetStep:
    """Benchmark-native step record."""

    dataset: str
    episode_id: str
    step_index: int
    goal: str
    screenshot: ScreenshotAsset
    raw_action: dict[str, Any]
    metadata: dict[str, Any]
    next_screenshot: ScreenshotAsset | None = None
    next_metadata: dict[str, Any] | None = None


@dataclass
class DatasetEpisode:
    """Benchmark-native episode record."""

    dataset: str
    episode_id: str
    goal: str
    steps: tuple[DatasetStep, ...]
    split: str | None = None
    app: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalAction:
    """Canonical action tuple."""

    action_type: str
    argument: str | None
    bbox: BBox | None
    app: str | None
    direction: str | None

    def as_tuple(self) -> tuple[str, str | None, BBox | None, str | None, str | None]:
        return (self.action_type, self.argument, self.bbox, self.app, self.direction)


@dataclass
class CanonicalStepRecord:
    """Canonical step record."""

    goal: str
    screenshot: ScreenshotAsset
    normalized_metadata: dict[str, Any]
    canonical_action: CanonicalAction
    effect_delta: dict[str, Any]
    bound_target: dict[str, Any] | None
    lexical_alignment: dict[str, Any]
