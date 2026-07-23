"""Save per-step evaluation details for export_baseline_json.py consumption.

This module is called from the evaluation orchestrator (or post-hoc from pickled
results) to produce a JSON file that export_baseline_json.py can convert into the
standard baseline output format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from guiaccel.evaluation.models import StepEvaluationResult


def extract_step_record(result: StepEvaluationResult) -> dict[str, Any]:
    """Extract JSON-serialisable fields from one StepEvaluationResult."""
    example = result.example
    repo_example = example.repository_example
    dataset_step = repo_example.dataset_step

    latency_ms: float
    if result.baseline_end_to_end_latency_ms is not None:
        latency_ms = float(result.baseline_end_to_end_latency_ms)
    else:
        latency_ms = float(result.baseline_fallback.response.latency_ms)

    token_usage = result.baseline_fallback.response.token_usage
    input_tokens: int = int(token_usage.prompt_tokens) + int(token_usage.visual_tokens)
    output_tokens: int = int(token_usage.generated_tokens)

    return {
        "episode_id": str(example.episode_id),
        "step_index": int(example.step_index),
        "instruction_mode": str(example.instruction_mode or "unknown"),
        "partition": str(example.partition or "unknown"),
        "goal": str(dataset_step.goal),
        "baseline_correct": bool(result.baseline.primary_correct),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def extract_step_records(results: Sequence[StepEvaluationResult]) -> list[dict[str, Any]]:
    return [extract_step_record(result) for result in results]


def _write_detailed_results_payload(
    output_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    resolved = Path(output_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "step_count": len(records),
                "steps": list(records),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return resolved


def save_detailed_results_from_records(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Path:
    return _write_detailed_results_payload(output_path, records)


def save_detailed_results(
    results: Sequence[StepEvaluationResult],
    output_path: Path,
) -> Path:
    """Serialise per-step evaluation details to *output_path* as JSON."""
    return _write_detailed_results_payload(output_path, extract_step_records(results))
