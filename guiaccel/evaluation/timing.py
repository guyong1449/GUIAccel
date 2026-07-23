"""Fine-grained timing data structures and aggregation for eval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class StepTimingRecord:
    """Fine-grained latency breakdown for one generate_batch() call."""

    # Outer phases (generate_batch level)
    image_prep_ms: float = 0.0
    message_build_ms: float = 0.0
    total_generation_ms: float = 0.0
    output_parse_ms: float = 0.0
    total_e2e_step_ms: float = 0.0

    # Inner phases (_run_generation_batch level)
    tokenization_ms: float = 0.0
    gpu_transfer_ms: float = 0.0
    model_generate_ms: float = 0.0
    output_decode_ms: float = 0.0

    # VisionZip / DivPrune sub-phases
    vision_encoder_ms: float = 0.0
    attn_scoring_prune_ms: float = 0.0
    llm_forward_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_steps: float = 0.0


@dataclass
class EvalTimingSingleton:
    """One-time overheads for a complete evaluation run."""

    model_load_ms: float = 0.0
    visionzip_patch_install_ms: float = 0.0
    divprune_setup_ms: float = 0.0
    metric_computation_ms: float = 0.0


def compute_timing_summary(
    results: Sequence[Any],
) -> dict[str, Any]:
    """Aggregate per-step timing across all results.

    For each phase key present in any baseline_timing or hybrid_timing dict,
    computes mean, sum, min, max, count across all non-None values.

    Returns a dict with keys 'baseline' and 'hybrid', each mapping phase name
    to {'mean_ms', 'sum_ms', 'min_ms', 'max_ms', 'count'}.
    """
    baseline_buckets: dict[str, list[float]] = {}
    hybrid_buckets: dict[str, list[float]] = {}

    for result in results:
        baseline_timing = getattr(result, "baseline_timing", None)
        if baseline_timing is not None:
            for key, value in baseline_timing.items():
                baseline_buckets.setdefault(key, []).append(float(value))
        hybrid_timing = getattr(result, "hybrid_timing", None)
        if hybrid_timing is not None:
            for key, value in hybrid_timing.items():
                hybrid_buckets.setdefault(key, []).append(float(value))

    def _summarize(buckets: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for key, values in sorted(buckets.items()):
            if not values:
                continue
            summary[key] = {
                "mean_ms": sum(values) / len(values),
                "sum_ms": sum(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "count": float(len(values)),
            }
        return summary

    return {
        "baseline": _summarize(baseline_buckets),
        "hybrid": _summarize(hybrid_buckets),
    }


__all__ = [
    "EvalTimingSingleton",
    "StepTimingRecord",
    "compute_timing_summary",
]
