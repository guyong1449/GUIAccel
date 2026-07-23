"""Shared routing-time context and accounting objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping, Sequence

from guiaccel.types import CanonicalStepRecord


@dataclass(frozen=True)
class TokenUsage:
    """Token and call accounting for one model invocation or routed step."""

    prompt_tokens: int = 0
    visual_tokens: int = 0
    generated_tokens: int = 0
    full_model_calls: int = 0
    local_model_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens + self.visual_tokens + self.generated_tokens)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            visual_tokens=self.visual_tokens + other.visual_tokens,
            generated_tokens=self.generated_tokens + other.generated_tokens,
            full_model_calls=self.full_model_calls + other.full_model_calls,
            local_model_calls=self.local_model_calls + other.local_model_calls,
        )


@dataclass(frozen=True)
class StepContext:
    """Runtime-observable context for one benchmark step."""

    observation_id: str
    record: CanonicalStepRecord
    history: tuple[CanonicalStepRecord, ...] = ()
    support_context: Mapping[str, Any] = field(default_factory=dict)
    screen_candidates: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def estimate_visual_tokens(width: int | None, height: int | None) -> int:
    """Approximate Qwen-style visual tokens from the post-resize image shape."""

    safe_width = max(1, int(width or 1))
    safe_height = max(1, int(height or 1))
    return int(ceil(safe_width / 28.0) * ceil(safe_height / 28.0))
