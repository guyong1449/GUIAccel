"""Bootstrap intervals for the minimal evaluation report."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

from guiaccel.evaluation.controller_metrics import compute_controller_metrics
from guiaccel.evaluation.models import StepEvaluationResult
from guiaccel.evaluation.android_eval import compute_android_metrics
from guiaccel.evaluation.learngui_eval import compute_learngui_metrics

BOOTSTRAP_RESAMPLES = 1000
TWO_SIDED_LOWER_QUANTILE = 0.025
TWO_SIDED_UPPER_QUANTILE = 0.975
ONE_SIDED_LOWER_BOUND_QUANTILE = 0.05


def compute_evaluation_statistics(results: Sequence[StepEvaluationResult]) -> dict[str, Any]:
    if not results:
        return {}
    benchmark = results[0].example.benchmark
    grouped = defaultdict(list)
    for result in results:
        grouped[result.example.group_id].append(result)
    groups = list(grouped.values())
    rng = random.Random(0)
    samples: list[dict[str, float]] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resampled = [item for _ in range(len(groups)) for item in groups[rng.randrange(len(groups))]]
        samples.append(_sample_metrics(resampled, benchmark=benchmark))
    statistics: dict[str, dict[str, float]] = {}
    for key in samples[0]:
        values = [float(sample[key]) for sample in samples]
        statistics[key] = {
            "mean": _mean(values),
            "lower": _quantile(values, TWO_SIDED_LOWER_QUANTILE),
            "upper": _quantile(values, TWO_SIDED_UPPER_QUANTILE),
            "lower_one_sided_95": _quantile(values, ONE_SIDED_LOWER_BOUND_QUANTILE),
        }
    return statistics


def _sample_metrics(results: Sequence[StepEvaluationResult], *, benchmark: str) -> dict[str, float]:
    controller = compute_controller_metrics(results)
    if benchmark == "AndroidControl":
        official = compute_android_metrics(results).get("overall", {})
        primary_delta = float(official.get("delta", {}).get("low_level_step_accuracy", 0.0))
    else:
        official = compute_learngui_metrics(results)
        primary_key = sorted(official)[0] if official else ""
        primary_delta = float(official.get(primary_key, {}).get("delta", {}).get("action_match_accuracy", 0.0))
    return {
        "primary_accuracy_delta": primary_delta,
        "full_model_call_reduction": float(controller.get("full_model_call_reduction", 0.0)),
        "total_token_reduction": float(controller.get("total_token_reduction", 0.0)),
        "latency_reduction": float(controller.get("latency_reduction", 0.0)),
        "accepted_step_accuracy_delta": float(controller.get("accepted_step_accuracy_delta_vs_baseline", 0.0)),
    }


def _mean(values) -> float:
    materialized = list(values)
    return float(sum(materialized)) / float(len(materialized)) if materialized else 0.0


def _quantile(values, level: float) -> float:
    materialized = sorted(float(value) for value in values)
    if not materialized:
        return 0.0
    index = max(0, min(len(materialized) - 1, int(round((len(materialized) - 1) * float(level)))))
    return materialized[index]
