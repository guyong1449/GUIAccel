"""Incremental partial evaluation reports during long-running eval jobs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from guiaccel.evaluation.controller_metrics import compute_controller_metrics
from guiaccel.evaluation.detailed_results import extract_step_records
from guiaccel.evaluation.log_summary import build_structured_log_summary
from guiaccel.evaluation.models import StepEvaluationResult
from guiaccel.evaluation.timing import compute_timing_summary


@dataclass(frozen=True)
class PartialReportContext:
    benchmark: str
    evaluation_split: str
    variant_id: str
    variant_description: str = ""
    backend_config: Any | None = None
    model_spec: Any | None = None


@dataclass
class PhaseStats:
    count: float = 0.0
    sum_ms: float = 0.0
    min_ms: float = math.inf
    max_ms: float = -math.inf

    def extend(self, values: Sequence[float]) -> None:
        for raw in values:
            value = float(raw)
            self.count += 1.0
            self.sum_ms += value
            self.min_ms = min(self.min_ms, value)
            self.max_ms = max(self.max_ms, value)

    def to_summary_entry(self) -> dict[str, float]:
        if self.count <= 0.0:
            return {
                "mean_ms": 0.0,
                "sum_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "count": 0.0,
            }
        return {
            "mean_ms": self.sum_ms / self.count,
            "sum_ms": self.sum_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "count": self.count,
        }


@dataclass
class TimingSummaryAccumulator:
    baseline: dict[str, PhaseStats] = field(default_factory=dict)
    hybrid: dict[str, PhaseStats] = field(default_factory=dict)

    def extend_from_results(self, results: Sequence[StepEvaluationResult]) -> None:
        for result in results:
            if result.baseline_timing is not None:
                for key, value in result.baseline_timing.items():
                    self.baseline.setdefault(key, PhaseStats()).extend((float(value),))
            if result.hybrid_timing is not None:
                for key, value in result.hybrid_timing.items():
                    self.hybrid.setdefault(key, PhaseStats()).extend((float(value),))

    def to_summary(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            "baseline": {
                key: stats.to_summary_entry()
                for key, stats in sorted(self.baseline.items())
            },
            "hybrid": {
                key: stats.to_summary_entry()
                for key, stats in sorted(self.hybrid.items())
            },
        }


def merge_timing_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    merged_baseline: dict[str, PhaseStats] = {}
    merged_hybrid: dict[str, PhaseStats] = {}

    def _merge_side(
        target: dict[str, PhaseStats],
        side: Mapping[str, Any],
    ) -> None:
        for phase, stats in side.items():
            if not isinstance(stats, Mapping):
                continue
            bucket = target.setdefault(str(phase), PhaseStats())
            count = float(stats.get("count") or 0.0)
            if count <= 0.0:
                continue
            bucket.count += count
            bucket.sum_ms += float(stats.get("sum_ms") or 0.0)
            bucket.min_ms = min(bucket.min_ms, float(stats.get("min_ms") or 0.0))
            bucket.max_ms = max(bucket.max_ms, float(stats.get("max_ms") or 0.0))

    for summary in summaries:
        _merge_side(merged_baseline, dict(summary.get("baseline") or {}))
        _merge_side(merged_hybrid, dict(summary.get("hybrid") or {}))

    return {
        "baseline": {key: stats.to_summary_entry() for key, stats in sorted(merged_baseline.items())},
        "hybrid": {key: stats.to_summary_entry() for key, stats in sorted(merged_hybrid.items())},
    }


def merge_controller_metrics(metrics_list: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    additive_int_keys = (
        "baseline_tokens_total",
        "hybrid_tokens_total",
        "baseline_tokens_input_text",
        "hybrid_tokens_input_text",
        "baseline_tokens_visual",
        "hybrid_tokens_visual",
        "baseline_tokens_output",
        "hybrid_tokens_output",
    )
    additive_float_keys = (
        "baseline_end_to_end_latency_ms_total",
        "hybrid_end_to_end_latency_ms_total",
    )
    merged: dict[str, Any] = {}
    for key in additive_int_keys:
        merged[key] = int(sum(int(item.get(key) or 0) for item in metrics_list))
    for key in additive_float_keys:
        merged[key] = float(sum(float(item.get(key) or 0.0) for item in metrics_list))

    baseline_tokens = float(merged["baseline_tokens_total"])
    hybrid_tokens = float(merged["hybrid_tokens_total"])
    baseline_latency = float(merged["baseline_end_to_end_latency_ms_total"])
    hybrid_latency = float(merged["hybrid_end_to_end_latency_ms_total"])

    def _relative_reduction(baseline: float, hybrid: float) -> float:
        if baseline <= 0.0:
            return 0.0
        return 1.0 - float(hybrid) / float(baseline)

    merged["total_token_reduction"] = _relative_reduction(baseline_tokens, hybrid_tokens)
    merged["input_text_token_reduction"] = _relative_reduction(
        float(merged["baseline_tokens_input_text"]),
        float(merged["hybrid_tokens_input_text"]),
    )
    merged["visual_token_reduction"] = _relative_reduction(
        float(merged["baseline_tokens_visual"]),
        float(merged["hybrid_tokens_visual"]),
    )
    merged["output_token_reduction"] = _relative_reduction(
        float(merged["baseline_tokens_output"]),
        float(merged["hybrid_tokens_output"]),
    )
    merged["latency_reduction"] = _relative_reduction(baseline_latency, hybrid_latency)
    return merged


def build_worker_partial_report(
    results: Sequence[StepEvaluationResult],
    *,
    timing_accumulator: TimingSummaryAccumulator | None = None,
) -> dict[str, Any]:
    timing_summary = (
        timing_accumulator.to_summary()
        if timing_accumulator is not None
        else compute_timing_summary(results)
    )
    controller_metrics = compute_controller_metrics(results)
    return {
        "result_count": len(results),
        "timing_summary": timing_summary,
        "controller_metrics": controller_metrics,
        "detailed_steps": extract_step_records(results),
    }


def write_worker_partial_report(
    progress_dir: Path,
    *,
    gpu_id: int,
    report: Mapping[str, Any],
) -> Path:
    progress_dir.mkdir(parents=True, exist_ok=True)
    out_path = progress_dir / f"partial_report_gpu{int(gpu_id)}.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_id": int(gpu_id),
        **dict(report),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def write_serial_partial_report(
    output_dir: Path,
    report: Mapping[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "partial_report.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **dict(report),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _load_partial_report_paths(progress_dir: Path) -> list[Path]:
    return sorted(progress_dir.glob("partial_report_gpu*.json"))


def write_aggregated_partial_reports(
    *,
    progress_dir: Path | None,
    output_dir: Path,
    benchmark: str,
    evaluation_split: str,
    variant_id: str,
    variant_description: str = "",
    backend_config: Any | None = None,
    model_spec: Any | None = None,
    serial_partial_report_path: Path | None = None,
) -> dict[str, Path]:
    """Merge worker partial reports and write aggregated JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    partial_reports: list[dict[str, Any]] = []
    if serial_partial_report_path is not None and serial_partial_report_path.exists():
        partial_reports.append(json.loads(serial_partial_report_path.read_text(encoding="utf-8")))
    elif progress_dir is not None and progress_dir.is_dir():
        for path in _load_partial_report_paths(progress_dir):
            try:
                partial_reports.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue

    if not partial_reports:
        return written

    timing_summary = merge_timing_summaries(
        [dict(report.get("timing_summary") or {}) for report in partial_reports]
    )
    controller_metrics = merge_controller_metrics(
        [dict(report.get("controller_metrics") or {}) for report in partial_reports]
    )
    result_count = int(sum(int(report.get("result_count") or 0) for report in partial_reports))

    detailed_steps: list[dict[str, Any]] = []
    for report in partial_reports:
        steps = report.get("detailed_steps")
        if isinstance(steps, list):
            detailed_steps.extend(dict(step) for step in steps if isinstance(step, Mapping))

    timing_path = output_dir / "partial_timing_summary.json"
    timing_path.write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "result_count": result_count,
                "timing_summary": timing_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    written["partial_timing_summary"] = timing_path

    controller_path = output_dir / "partial_controller_metrics.json"
    controller_path.write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "result_count": result_count,
                "controller_metrics": controller_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    written["partial_controller_metrics"] = controller_path

    detailed_path = output_dir / "partial_detailed_results.json"
    detailed_path.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "step_count": len(detailed_steps),
                "steps": detailed_steps,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    written["partial_detailed_results"] = detailed_path

    partial_eval_report = {
        "benchmark": benchmark,
        "evaluation_split": evaluation_split,
        "variant_id": variant_id,
        "variant_description": variant_description,
        "result_count": result_count,
        "official_metrics": {},
        "controller_metrics": controller_metrics,
        "null_action_diagnostics": {},
        "timing_summary": timing_summary,
    }
    log_summary = build_structured_log_summary(
        partial_eval_report,
        backend_config=backend_config,
        eval_json_path=None,
    )
    log_summary["metadata"]["partial"] = True
    log_summary_path = output_dir / "partial_log_summary.json"
    log_summary_path.write_text(json.dumps(log_summary, indent=2, sort_keys=True), encoding="utf-8")
    written["partial_log_summary"] = log_summary_path
    return written
