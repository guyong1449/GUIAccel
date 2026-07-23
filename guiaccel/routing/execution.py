"""Compatibility stubs for disambiguation types used by model backends.

These were originally part of the Memoir routing pipeline but the type
definitions are needed by qwen_backend.py and service_backend.py for the
BatchLocalDisambiguationModel protocol.  We keep only the dataclass
definitions here — no quotient / reducer logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from guiaccel.routing.common import TokenUsage


@dataclass(frozen=True)
class PreparedVisualInput:
    """Compatibility wrapper for local image inputs."""
    screenshot: Any
    crop_bbox: tuple[int, int, int, int] | None
    width: int
    height: int
    crop_applied: bool
    pixel_budget: int | None
    estimated_visual_tokens: int


@dataclass(frozen=True)
class PromptPlan:
    """Compatibility wrapper for local prompt plans."""
    text: str
    token_count: int
    kept_history_count: int = 0
    removed_history_count: int = 0
    fixed_prefix_key: str = ""
    support_summary: str | None = None


@dataclass(frozen=True)
class DisambiguationRequest:
    """Compatibility request type for bounded local solver calls."""
    observation_id: str
    template_id: str
    state_id: int
    controller_action_type: str
    prompt: PromptPlan
    visual: PreparedVisualInput
    valid_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    slot_values: Mapping[str, Any]
    max_new_tokens: int
    max_text_tokens: int


@dataclass(frozen=True)
class DisambiguationResponse:
    """Compatibility response type for bounded local solver calls."""
    selected_label: str | None
    valid: bool = True
    raw_output: str | None = None
    token_usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0


@runtime_checkable
class LocalDisambiguationModel(Protocol):
    """Protocol for constrained-label local solves."""
    def disambiguate(self, request: DisambiguationRequest) -> DisambiguationResponse:
        raise NotImplementedError


@runtime_checkable
class BatchLocalDisambiguationModel(LocalDisambiguationModel, Protocol):
    """Optional batch protocol for constrained-label local solves."""
    def disambiguate_batch(
        self,
        requests: Sequence[DisambiguationRequest],
    ) -> Sequence[DisambiguationResponse]:
        raise NotImplementedError
