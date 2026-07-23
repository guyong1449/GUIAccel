"""Evaluation data structures for GUI agent inference experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from guiaccel.routing.common import StepContext, TokenUsage
from guiaccel.types import CanonicalAction


@dataclass(frozen=True)
class EvaluationExample:
    """One benchmark step prepared for evaluation."""

    benchmark: str
    setting: str
    partition: str
    group_id: str
    episode_id: str
    step_index: int
    context: StepContext
    shot_count: int | None = None
    instruction_mode: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepJudgement:
    """Correctness labels for one action under benchmark scoring."""

    action: CanonicalAction | None
    metrics: Mapping[str, bool]
    primary_correct: bool


@dataclass(frozen=True)
class StepEvaluationResult:
    """Evaluation result for a single step."""

    example: EvaluationExample
    ground_truth_action: CanonicalAction
    judgement: StepJudgement
    token_usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    raw_output: str | None = None
    timing: dict[str, float] | None = None
    method: str = "baseline"
