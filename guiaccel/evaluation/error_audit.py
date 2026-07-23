"""Accepted-error audit for the minimal new experiment plan."""

from __future__ import annotations

from typing import Any, Sequence

from guiaccel.evaluation.models import StepEvaluationResult


def generate_error_audit(results: Sequence[StepEvaluationResult]) -> dict[str, Any]:
    accepted_errors = [
        result
        for result in results
        if result.final_mode == "skill" and not result.hybrid.primary_correct
    ]
    return {
        "accepted_error_count": len(accepted_errors),
        "accepted_error_examples": [
            {
                "episode_id": result.example.episode_id,
                "step_index": result.example.step_index,
                "selected_skill_id": result.selected_template_id,
                "selected_budget": result.hybrid_routed.selected_budget,
                "baseline_correct": result.baseline.primary_correct,
                "hybrid_correct": result.hybrid.primary_correct,
            }
            for result in accepted_errors[:50]
        ],
    }

