"""Build and format self-contained evaluation run summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from guiaccel.evaluation.android_eval import ANDROID_PARTITION_ORDER
from guiaccel.evaluation.models import StepEvaluationResult
from guiaccel.routing.fallback import ModelRuntimeSpec

SUMMARY_HEADER = "=== SkillReuse Evaluation Summary ==="
SUMMARY_FOOTER = "=== End SkillReuse Evaluation Summary ==="
PARTIAL_METRICS_MARKER = "[Partial metrics @"
EVAL_LOG_PARSE_PROMPT_FILENAME = "eval_log_parse_prompt.md"
_MAX_FORMAT_LINES = 80

_ANDROID_SUBSET_KEYS = (*ANDROID_PARTITION_ORDER, "overall")


# ---------------------------------------------------------------------------
# Utilities (shared)
# ---------------------------------------------------------------------------

def inspect_eval_json_path(path: str | Path | None) -> dict[str, Any]:
    """Return exists/readable status for an evaluation JSON path."""
    if path is None or str(path).strip() == "":
        return {"exists": False, "readable": False, "error": "path not set"}
    resolved = Path(path)
    if not resolved.exists():
        return {"exists": False, "readable": False, "error": "file not found"}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"exists": True, "readable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {"exists": True, "readable": False, "error": "json root is not an object"}
    return {"exists": True, "readable": True, "error": None}


def _parse_attn_implementation(
    value: str | Mapping[str, str] | None,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, str):
        return value, value
    vit = value.get("vision_config") or value.get("vit")
    llm = value.get("text_config") or value.get("llm")
    return (
        str(vit) if vit is not None else None,
        str(llm) if llm is not None else None,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def _retention(hybrid: int, baseline: int) -> float | None:
    if baseline <= 0:
        return None
    return float(hybrid) / float(baseline)


# ---------------------------------------------------------------------------
# VisionZip-style build_log_summary (flat, journal-oriented)
# Used by apply_evaluation_summary / format_log_summary
# ---------------------------------------------------------------------------

def build_log_summary(
    report: Mapping[str, Any],
    *,
    backend_config: Any | None = None,
    eval_json_path: str | Path | None = None,
    eval_json_source: str = "fresh",
) -> dict[str, Any]:
    """Build a self-contained summary dict from an in-memory evaluation report."""
    result_count = int(report.get("result_count") or 0)
    official = dict(report.get("official_metrics") or {})
    controller = dict(report.get("controller_metrics") or {})
    null_diag = dict(report.get("null_action_diagnostics") or {})
    output_path = eval_json_path or report.get("output_path")

    return {
        "header": SUMMARY_HEADER,
        "run_metadata": _build_run_metadata(report, backend_config=backend_config),
        "overall_accuracy": _build_overall_accuracy(
            str(report.get("benchmark") or ""),
            official,
            null_diag,
        ),
        "per_subset_accuracy": _build_per_subset_accuracy(
            str(report.get("benchmark") or ""),
            official,
        ),
        "tokens_compression": _build_tokens_compression(controller, result_count),
        "latency_speedup": _build_latency_speedup(controller, result_count),
        "eval_json": _resolve_eval_json_status(output_path, eval_json_source),
    }


def format_log_summary(summary: Mapping[str, Any]) -> str:
    """Format a bounded, human-readable summary block for slurm.out."""
    lines: list[str] = [str(summary.get("header") or SUMMARY_HEADER), ""]

    meta = dict(summary.get("run_metadata") or {})
    lines.append("[Run Metadata]")
    for key in (
        "benchmark",
        "evaluation_split",
        "variant_id",
        "result_count",
        "backend",
        "vit_attn",
        "llm_attn",
        "compression_target",
    ):
        if key in meta:
            lines.append(f"  {key}: {meta[key]}")

    overall = dict(summary.get("overall_accuracy") or {})
    if overall:
        lines.extend(["", "[Overall Accuracy]"])
        for key, value in overall.items():
            lines.append(f"  {key}: {value}")

    subsets = list(summary.get("per_subset_accuracy") or [])
    if subsets:
        lines.extend(["", "[Per-Subset Accuracy]"])
        for entry in subsets[:12]:
            subset = entry.get("subset", "unknown")
            lines.append(f"  {subset}:")
            for metric_key, metric_value in dict(entry.get("metrics") or {}).items():
                lines.append(f"    {metric_key}: {metric_value}")

    tokens = dict(summary.get("tokens_compression") or {})
    if tokens:
        lines.extend(["", "[Tokens & Compression]"])
        for key, value in tokens.items():
            lines.append(f"  {key}: {value}")

    latency = dict(summary.get("latency_speedup") or {})
    if latency:
        lines.extend(["", "[Latency & Speedup]"])
        for key, value in latency.items():
            lines.append(f"  {key}: {value}")

    eval_json = dict(summary.get("eval_json") or {})
    lines.extend(
        [
            "",
            "[Eval JSON Source]",
            f"  path: {eval_json.get('path') or 'N/A'}",
            f"  source: {eval_json.get('source', 'unknown')}",
            f"  exists: {eval_json.get('exists', False)}",
            f"  readable: {eval_json.get('readable', False)}",
            "",
            f"=== End {SUMMARY_HEADER[4:]} ===",
        ]
    )
    return "\n".join(lines[:_MAX_FORMAT_LINES])


def apply_evaluation_summary(
    report: dict[str, Any],
    *,
    output_path: Path | str | None = None,
    write_json_file: Any | None = None,
    backend_config: Any | None = None,
    eval_json_source: str = "fresh",
) -> dict[str, Any]:
    """Attach, print, and persist log_summary for a completed evaluation report."""
    if output_path is not None:
        report["output_path"] = str(output_path)

    existing = report.get("log_summary")
    if not isinstance(existing, Mapping):
        report["log_summary"] = build_log_summary(
            report,
            backend_config=backend_config,
            eval_json_path=report.get("output_path"),
            eval_json_source=eval_json_source,
        )
    summary = dict(report["log_summary"])

    print(format_log_summary(summary), flush=True)

    resolved_output_path = str(report.get("output_path") or "")
    if resolved_output_path and write_json_file is not None and "benchmark" in report:
        write_json_file(Path(resolved_output_path), report)

    from guiaccel.journal import persist_evaluation_summary_to_journal

    persist_evaluation_summary_to_journal(summary, report_path=resolved_output_path or None)
    return report


# ---------------------------------------------------------------------------
# DivPrune-style build_structured_log_summary (structured, metadata-rich)
# Used by attach_log_summary_to_report / format_evaluation_summary
# ---------------------------------------------------------------------------

def _extract_partition_accuracy(partition: Mapping[str, Any]) -> dict[str, Any]:
    hybrid = dict(partition.get("hybrid") or {})
    baseline = dict(partition.get("baseline") or {})
    return {
        "hybrid": {
            "high_level_step_accuracy": _safe_float(hybrid.get("high_level_step_accuracy")),
            "low_level_step_accuracy": _safe_float(hybrid.get("low_level_step_accuracy")),
            "high_level_episode_accuracy": _safe_float(hybrid.get("high_level_episode_accuracy")),
            "action_type_accuracy": _safe_float(hybrid.get("action_type_accuracy")),
            "action_match_accuracy": _safe_float(hybrid.get("action_match_accuracy")),
        },
        "baseline": {
            "high_level_step_accuracy": _safe_float(baseline.get("high_level_step_accuracy")),
            "low_level_step_accuracy": _safe_float(baseline.get("low_level_step_accuracy")),
            "high_level_episode_accuracy": _safe_float(baseline.get("high_level_episode_accuracy")),
            "action_type_accuracy": _safe_float(baseline.get("action_type_accuracy")),
            "action_match_accuracy": _safe_float(baseline.get("action_match_accuracy")),
        },
    }


def _extract_accuracy_section(
    benchmark: str,
    official_metrics: Mapping[str, Any],
    null_action_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    by_subset: dict[str, Any] = {}
    if benchmark == "AndroidControl":
        for key in _ANDROID_SUBSET_KEYS:
            if key in official_metrics:
                by_subset[key] = _extract_partition_accuracy(official_metrics[key])
        overall = official_metrics.get("overall") or {}
    else:
        for key, value in official_metrics.items():
            if isinstance(value, Mapping):
                by_subset[str(key)] = _extract_partition_accuracy(value)
        overall = next(iter(official_metrics.values()), {}) if official_metrics else {}

    overall_accuracy = _extract_partition_accuracy(overall) if isinstance(overall, Mapping) else {}
    return {
        "overall": overall_accuracy,
        "by_subset": by_subset,
        "null_actions": {
            "baseline_null_action_count": _safe_int(null_action_diagnostics.get("baseline_null_action_count")),
            "baseline_null_action_rate": _safe_float(null_action_diagnostics.get("baseline_null_action_rate")),
            "hybrid_null_action_count": _safe_int(null_action_diagnostics.get("hybrid_null_action_count")),
            "hybrid_null_action_rate": _safe_float(null_action_diagnostics.get("hybrid_null_action_rate")),
        },
    }


def _extract_tokens_section(
    controller_metrics: Mapping[str, Any],
    *,
    result_count: int,
    keep_ratio: float | None,
) -> dict[str, Any]:
    step_count = max(1, int(result_count))
    hybrid_total = _safe_int(controller_metrics.get("hybrid_tokens_total"))
    hybrid_visual = _safe_int(controller_metrics.get("hybrid_tokens_visual"))
    hybrid_input = _safe_int(controller_metrics.get("hybrid_tokens_input_text"))
    hybrid_output = _safe_int(controller_metrics.get("hybrid_tokens_output"))
    return {
        "baseline_tokens_total": _safe_int(controller_metrics.get("baseline_tokens_total")),
        "hybrid_tokens_total": hybrid_total,
        "baseline_tokens_visual": _safe_int(controller_metrics.get("baseline_tokens_visual")),
        "hybrid_tokens_visual": hybrid_visual,
        "baseline_tokens_input_text": _safe_int(controller_metrics.get("baseline_tokens_input_text")),
        "hybrid_tokens_input_text": hybrid_input,
        "baseline_tokens_output": _safe_int(controller_metrics.get("baseline_tokens_output")),
        "hybrid_tokens_output": hybrid_output,
        "mean_hybrid_tokens_total_per_step": float(hybrid_total) / float(step_count),
        "mean_hybrid_tokens_visual_per_step": float(hybrid_visual) / float(step_count),
        "mean_hybrid_tokens_input_text_per_step": float(hybrid_input) / float(step_count),
        "mean_hybrid_tokens_output_per_step": float(hybrid_output) / float(step_count),
        "total_token_reduction": _safe_float(controller_metrics.get("total_token_reduction")),
        "visual_token_reduction": _safe_float(controller_metrics.get("visual_token_reduction")),
        "input_text_token_reduction": _safe_float(controller_metrics.get("input_text_token_reduction")),
        "output_token_reduction": _safe_float(controller_metrics.get("output_token_reduction")),
        "keep_ratio": keep_ratio,
    }


def _extract_latency_section(
    controller_metrics: Mapping[str, Any],
    *,
    result_count: int,
) -> dict[str, Any]:
    step_count = max(1, int(result_count))
    baseline_total = _safe_float(controller_metrics.get("baseline_end_to_end_latency_ms_total"))
    hybrid_total = _safe_float(controller_metrics.get("hybrid_end_to_end_latency_ms_total"))
    latency_reduction = _safe_float(controller_metrics.get("latency_reduction"))
    speedup = baseline_total / hybrid_total if hybrid_total > 0.0 else None
    return {
        "baseline_end_to_end_latency_ms_total": baseline_total,
        "hybrid_end_to_end_latency_ms_total": hybrid_total,
        "mean_step_lat_ms": hybrid_total / float(step_count),
        "latency_reduction": latency_reduction,
        "speedup_vs_baseline": speedup,
    }


def build_structured_log_summary(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    eval_json_path: str | Path | None = None,
    backend_config: Any | None = None,
    model_spec: ModelRuntimeSpec | None = None,
) -> dict[str, Any]:
    """Build a structured, metadata-rich JSON-serializable summary (DivPrune-style).

    Returns a dict with keys: metadata, accuracy, tokens, latency.
    See also: build_log_summary (VisionZip-style flat summary).
    """
    benchmark = str(report.get("benchmark") or "")
    official_metrics = dict(report.get("official_metrics") or {})
    controller_metrics = dict(report.get("controller_metrics") or {})
    null_action_diagnostics = dict(report.get("null_action_diagnostics") or {})
    result_count = _safe_int(report.get("result_count"))

    resolved_output_dir = str(Path(output_dir).resolve()) if output_dir is not None else None
    resolved_eval_json_path = (
        str(Path(eval_json_path).resolve())
        if eval_json_path is not None
        else str(report.get("output_path") or "") or None
    )

    backend_kind = None
    keep_ratio = None
    vit_attn = None
    llm_attn = None
    if backend_config is not None:
        backend_kind = getattr(backend_config, "kind", None)
        keep_ratio = getattr(backend_config, "keep_ratio", None)
        vit_attn, llm_attn = _parse_attn_implementation(
            getattr(backend_config, "attn_implementation", None)
        )

    model_name = None
    model_path = None
    if model_spec is not None:
        model_name = model_spec.request_model_name
        if model_spec.base_model_path is not None:
            model_path = str(model_spec.base_model_path)

    return {
        "metadata": {
            "benchmark": benchmark,
            "evaluation_split": str(report.get("evaluation_split") or ""),
            "variant_id": str(report.get("variant_id") or ""),
            "variant_description": str(report.get("variant_description") or ""),
            "result_count": result_count,
            "backend_kind": backend_kind,
            "model_name": model_name,
            "model_path": model_path,
            "vit_attn_implementation": vit_attn,
            "llm_attn_implementation": llm_attn,
            "keep_ratio": keep_ratio,
            "output_dir": resolved_output_dir,
            "eval_json_path": resolved_eval_json_path,
            "eval_json": inspect_eval_json_path(resolved_eval_json_path),
        },
        "accuracy": _extract_accuracy_section(
            benchmark,
            official_metrics,
            null_action_diagnostics,
        ),
        "tokens": _extract_tokens_section(
            controller_metrics,
            result_count=result_count,
            keep_ratio=keep_ratio,
        ),
        "latency": _extract_latency_section(controller_metrics, result_count=result_count),
    }


def attach_log_summary_to_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    eval_json_path: str | Path | None = None,
    backend_config: Any | None = None,
    model_spec: ModelRuntimeSpec | None = None,
) -> dict[str, Any]:
    """Attach ``log_summary`` to *report* and return the summary dict."""
    summary = build_structured_log_summary(
        report,
        output_dir=output_dir,
        eval_json_path=eval_json_path,
        backend_config=backend_config,
        model_spec=model_spec,
    )
    report["log_summary"] = summary
    return summary


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _hybrid_step_latency_ms(result: StepEvaluationResult) -> float:
    if result.hybrid_end_to_end_latency_ms is not None:
        return float(result.hybrid_end_to_end_latency_ms)
    return float(result.hybrid_routed.latency_ms)


def compute_mode_benchmark_metrics(
    results: Sequence[StepEvaluationResult],
) -> dict[str, dict[str, float]]:
    """Per instruction_mode metrics aligned with AndroidControl benchmark tables."""
    episodes: dict[str, dict[str, dict[str, float]]] = {
        "high_level": {},
        "low_level": {},
    }
    step_totals: dict[str, dict[str, float]] = {
        "high_level": {"correct": 0.0, "steps": 0.0, "step_latency_ms": 0.0},
        "low_level": {"correct": 0.0, "steps": 0.0, "step_latency_ms": 0.0},
    }

    for result in results:
        mode = str(getattr(result.example, "instruction_mode", "") or "")
        if mode not in episodes:
            continue
        episode_id = str(result.example.episode_id)
        token_usage = result.hybrid_routed.token_usage
        input_tokens = float(int(token_usage.prompt_tokens) + int(token_usage.visual_tokens))
        output_tokens = float(int(token_usage.generated_tokens))
        step_latency_ms = _hybrid_step_latency_ms(result)

        bucket = step_totals[mode]
        bucket["steps"] += 1.0
        if result.hybrid.primary_correct:
            bucket["correct"] += 1.0
        bucket["step_latency_ms"] += step_latency_ms

        episode = episodes[mode].setdefault(
            episode_id,
            {"input_tokens": 0.0, "output_tokens": 0.0, "latency_ms": 0.0},
        )
        episode["input_tokens"] += input_tokens
        episode["output_tokens"] += output_tokens
        episode["latency_ms"] += step_latency_ms

    metrics: dict[str, dict[str, float]] = {}
    for mode in ("high_level", "low_level"):
        steps = step_totals[mode]["steps"]
        episode_values = list(episodes[mode].values())
        episode_count = float(len(episode_values))
        metrics[mode] = {
            "step_accuracy": (
                step_totals[mode]["correct"] / steps if steps > 0.0 else 0.0
            ),
            "avg_input_tokens_per_episode": (
                sum(item["input_tokens"] for item in episode_values) / episode_count
                if episode_count > 0.0
                else 0.0
            ),
            "avg_output_tokens_per_episode": (
                sum(item["output_tokens"] for item in episode_values) / episode_count
                if episode_count > 0.0
                else 0.0
            ),
            "step_latency_s": (
                (step_totals[mode]["step_latency_ms"] / steps) / 1000.0 if steps > 0.0 else 0.0
            ),
            "e2e_latency_s": (
                (sum(item["latency_ms"] for item in episode_values) / episode_count) / 1000.0
                if episode_count > 0.0
                else 0.0
            ),
        }
    return metrics


def format_mode_benchmark_progress_line(
    mode: str,
    stats: Mapping[str, float],
    *,
    completed: int,
    target: int,
    gpu_id: int | None = None,
    label: str | None = None,
) -> str:
    """Format one benchmark-table row for stdout (grep-friendly)."""
    if gpu_id is not None:
        prefix = f"[Partial metrics @ {completed}/{target} GPU{int(gpu_id)}]"
    elif label:
        prefix = f"[Partial metrics @ {completed}/{target} {label}]"
    else:
        prefix = f"[Partial metrics @ {completed}/{target} AGG]"

    return (
        f"{prefix} {mode} "
        f"step_acc={float(stats.get('step_accuracy') or 0.0):.4f} "
        f"avg_in_tok/ep={float(stats.get('avg_input_tokens_per_episode') or 0.0):.0f} "
        f"avg_out_tok/ep={float(stats.get('avg_output_tokens_per_episode') or 0.0):.0f} "
        f"step_lat_s={float(stats.get('step_latency_s') or 0.0):.3f} "
        f"episode_lat_s={float(stats.get('e2e_latency_s') or 0.0):.2f}"
    )


def print_mode_benchmark_progress(
    results: Sequence[StepEvaluationResult],
    *,
    completed: int,
    target: int,
    gpu_id: int | None = None,
    label: str | None = None,
) -> None:
    """Print HL/LL benchmark-table rows to stdout (slurm.out)."""
    if not results:
        return
    mode_metrics = compute_mode_benchmark_metrics(results)
    for mode in ("high_level", "low_level"):
        print(
            format_mode_benchmark_progress_line(
                mode,
                mode_metrics[mode],
                completed=completed,
                target=target,
                gpu_id=gpu_id,
                label=label,
            ),
            flush=True,
        )


def format_partial_progress_stdout(
    metrics: Mapping[str, Any],
    *,
    controller_metrics: Mapping[str, Any] | None = None,
    gpu_id: int | None = None,
    label: str | None = None,
) -> str:
    """Format a grep-friendly incremental progress line for slurm.out."""
    completed = int(metrics.get("completed") or metrics.get("tot") or 0)
    target = int(metrics.get("target") or 0)
    hl_pct = 100.0 * float(metrics.get("hl_step_accuracy") or 0.0)
    ll_pct = 100.0 * float(metrics.get("ll_step_accuracy") or 0.0)
    null_pct = 100.0 * float(metrics.get("null_action_rate") or 0.0)
    sps = float(metrics.get("steps_per_second") or 0.0)
    progress_pct = float(
        metrics.get("progress_percent")
        if metrics.get("progress_percent") is not None
        else (100.0 * completed / target if target > 0 else 0.0)
    )
    elapsed_s = metrics.get("elapsed_s")

    if gpu_id is not None:
        prefix = f"[Partial metrics @ {completed}/{target} GPU{int(gpu_id)}]"
    elif label:
        prefix = f"[Partial metrics @ {completed}/{target} {label}]"
    else:
        prefix = f"[Partial metrics @ {completed}/{target} AGG]"

    parts = [
        prefix,
        f"HL={hl_pct:.1f}% LL={ll_pct:.1f}% null={null_pct:.1f}%",
    ]

    if controller_metrics:
        hybrid_total = int(controller_metrics.get("hybrid_tokens_total") or 0)
        hybrid_visual = int(controller_metrics.get("hybrid_tokens_visual") or 0)
        hybrid_output = int(controller_metrics.get("hybrid_tokens_output") or 0)
        hybrid_latency_total = float(controller_metrics.get("hybrid_end_to_end_latency_ms_total") or 0.0)
        step_count = max(1, completed)
        parts.append(
            f"tokens_total={hybrid_total} visual={hybrid_visual} output={hybrid_output}"
        )
        parts.append(f"mean_step_lat_ms={hybrid_latency_total / float(step_count):.1f}")

    if elapsed_s is not None:
        parts.append(f"elapsed={_format_elapsed(float(elapsed_s))}")

    parts.append(f"({sps:.2f} step/s {progress_pct:.1f}%)")
    return " ".join(parts)


def resolve_eval_log_path() -> Path | None:
    """Return the Slurm stdout log path when running under a journal launcher."""
    from guiaccel.journal import resolve_journal_run_dir

    run_dir = resolve_journal_run_dir()
    if run_dir is None:
        return None
    return run_dir / "slurm.out"


def _partial_metrics_example_line() -> str:
    """Example incremental line generated by the same formatter used at runtime."""
    return format_partial_progress_stdout(
        {
            "completed": 1000,
            "target": 18412,
            "hl_step_accuracy": 0.444,
            "ll_step_accuracy": 0.547,
            "null_action_rate": 0.038,
            "steps_per_second": 0.05,
            "progress_percent": 5.4,
            "elapsed_s": 19953.0,
        },
        controller_metrics={
            "hybrid_tokens_total": 1_894_900,
            "hybrid_tokens_visual": 689_700,
            "hybrid_tokens_output": 214_900,
            "hybrid_end_to_end_latency_ms_total": 3_828_100.0,
        },
        gpu_id=0,
    )


def build_eval_log_parse_prompt(*, log_path: str | Path | None = None) -> str:
    """Build an agent-facing prompt that describes how to parse eval stdout logs."""
    resolved = Path(log_path).resolve() if log_path is not None else resolve_eval_log_path()
    log_ref = str(resolved) if resolved is not None else "<SLURM_LOG>"
    example = _partial_metrics_example_line()
    return "\n".join(
        [
            "Parse a SkillReuse AndroidControl evaluation log and emit markdown tables "
            "(format aligned with docs/plans/visionzip_four_way_eval_comparison.md).",
            "",
            f"Log file: {log_ref}",
            "",
            "## Output sink",
            "- Python `print(..., flush=True)` from evaluation → process stdout",
            "- Slurm sbatch `--output` → `<journal_run_dir>/slurm.out`",
            "- Smoke runs: `slurm_logs/<backend>/smoke/run_<stamp>/slurm.out`",
            "- Full eval: `training_journal/<YYYY_MM_DD>/run_<stamp>/slurm.out`",
            "- `terminal.log` is a symlink to `slurm.out` in the same run dir",
            "",
            "## Incremental lines (during eval)",
            f"Marker: `{PARTIAL_METRICS_MARKER}`",
            "Scopes: `GPU{n}` (per worker), `AGG` (aggregated), `Serial`",
            "Summary line (format_partial_progress_stdout):",
            example,
            "Benchmark-table rows (one per instruction_mode, high_level / low_level):",
            (
                "[Partial metrics @ 1000/18412 AGG] high_level "
                "step_acc=0.5527 avg_in_tok/ep=16182 avg_out_tok/ep=652 step_lat_s=4.963 episode_lat_s=27.88"
            ),
            (
                "[Partial metrics @ 1000/18412 AGG] low_level "
                "step_acc=0.6621 avg_in_tok/ep=16095 avg_out_tok/ep=706 step_lat_s=5.287 episode_lat_s=29.65"
            ),
            "",
            "Summary fields:",
            "- `HL` / `LL`: hybrid high/low level step accuracy (percent)",
            "- `null`: null-action rate (percent)",
            "- `tokens_total` / `visual` / `output`: hybrid cumulative token counts",
            "- `mean_step_lat_ms`: hybrid end-to-end latency mean per completed step (ms)",
            "- `elapsed`: wall time since worker start (HhMMmSSs)",
            "- trailing `(step/s progress%)`: throughput and completion percent",
            "",
            "Benchmark-table row fields (per instruction_mode):",
            "- `step_acc`: hybrid step accuracy (0–1)",
            "- `avg_in_tok/ep`: mean hybrid input tokens (prompt+visual) per episode",
            "- `avg_out_tok/ep`: mean hybrid output tokens per episode",
            "- `step_lat_s`: mean hybrid step latency (seconds)",
            "- `episode_lat_s`: mean episode end-to-end latency (seconds)",
            "",
            "## Final summary block (eval complete)",
            f"Marker: `{SUMMARY_HEADER}` … `{SUMMARY_FOOTER}`",
            "Sections: `[accuracy.overall.hybrid]`, `[accuracy.by_subset.hybrid]`, `[tokens]`, `[latency]`",
            "",
            "## Extraction rules",
            "1. Running snapshot: last line matching `Partial metrics @` with `AGG`",
            "2. Final metrics: prefer the summary block; else `evaluation_summary.json` in journal dir",
            "3. Table columns: HL/LL step accuracy, HL episode accuracy, null rate, visual/total/output tokens "
            "(total + per-step mean), E2E latency total + mean/step, elapsed wall time, speedup vs baseline",
            "",
            "## Grep",
            f"grep '{PARTIAL_METRICS_MARKER}' {log_ref} | tail -3",
            f"grep -A40 '{SUMMARY_HEADER}' {log_ref} | tail -45",
        ]
    )


def print_eval_log_banner(*, log_path: str | Path | None = None) -> None:
    """Print once at eval start: where logs go and how to parse them."""
    resolved = Path(log_path).resolve() if log_path is not None else resolve_eval_log_path()
    if resolved is not None:
        sink = str(resolved)
        prompt_hint = f"{resolved.parent / EVAL_LOG_PARSE_PROMPT_FILENAME}"
    else:
        sink = "stdout (Slurm → training_journal/.../slurm.out)"
        prompt_hint = EVAL_LOG_PARSE_PROMPT_FILENAME
    print(
        f"[Eval log] metrics → {sink} | "
        f"incremental: grep '{PARTIAL_METRICS_MARKER}' | "
        f"final: grep '{SUMMARY_HEADER}' | "
        f"parse prompt: {prompt_hint}",
        flush=True,
    )


def format_evaluation_summary(log_summary: Mapping[str, Any]) -> str:
    """Render a bounded, grep-friendly summary block for stdout/journal (structured-summary format)."""
    metadata = dict(log_summary.get("metadata") or {})
    accuracy = dict(log_summary.get("accuracy") or {})
    tokens = dict(log_summary.get("tokens") or {})
    latency = dict(log_summary.get("latency") or {})
    eval_json = dict(metadata.get("eval_json") or {})
    overall = dict(accuracy.get("overall") or {})
    overall_hybrid = dict(overall.get("hybrid") or {})
    null_actions = dict(accuracy.get("null_actions") or {})
    by_subset = dict(accuracy.get("by_subset") or {})

    lines = [
        SUMMARY_HEADER,
        (
            f"benchmark={metadata.get('benchmark', '')} "
            f"split={metadata.get('evaluation_split', '')} "
            f"variant={metadata.get('variant_id', '')} "
            f"results={metadata.get('result_count', 0)}"
        ),
        (
            f"backend={metadata.get('backend_kind') or 'n/a'} "
            f"vit_attn={metadata.get('vit_attn_implementation') or 'n/a'} "
            f"llm_attn={metadata.get('llm_attn_implementation') or 'n/a'} "
            f"keep_ratio={metadata.get('keep_ratio') if metadata.get('keep_ratio') is not None else 'n/a'}"
        ),
        f"output_dir={metadata.get('output_dir') or 'n/a'}",
        f"eval_json_path={metadata.get('eval_json_path') or 'n/a'}",
        (
            f"eval_json_exists={str(bool(eval_json.get('exists'))).lower()} "
            f"eval_json_readable={str(bool(eval_json.get('readable'))).lower()}"
        ),
        "",
        "[accuracy.overall.hybrid]",
        f"high_level_step_accuracy={_format_rate(overall_hybrid.get('high_level_step_accuracy'))}",
        f"low_level_step_accuracy={_format_rate(overall_hybrid.get('low_level_step_accuracy'))}",
        f"high_level_episode_accuracy={_format_rate(overall_hybrid.get('high_level_episode_accuracy'))}",
        f"action_match_accuracy={_format_rate(overall_hybrid.get('action_match_accuracy'))}",
        (
            f"hybrid_null_action_count={null_actions.get('hybrid_null_action_count', 0)} "
            f"hybrid_null_action_rate={_format_rate(null_actions.get('hybrid_null_action_rate'))}"
        ),
        "",
        "[accuracy.by_subset.hybrid]",
    ]

    benchmark = str(metadata.get("benchmark") or "")
    for subset_name, subset_payload in by_subset.items():
        hybrid = dict(subset_payload.get("hybrid") or {})
        if benchmark == "AndroidControl":
            lines.append(
                f"{subset_name}: "
                f"hl_step={_format_rate(hybrid.get('high_level_step_accuracy'))} "
                f"ll_step={_format_rate(hybrid.get('low_level_step_accuracy'))} "
                f"hl_episode={_format_rate(hybrid.get('high_level_episode_accuracy'))}"
            )
        else:
            lines.append(
                f"{subset_name}: "
                f"action_match={_format_rate(hybrid.get('action_match_accuracy'))} "
                f"action_type={_format_rate(hybrid.get('action_type_accuracy'))}"
            )

    lines.extend(
        [
            "",
            "[tokens]",
            (
                f"hybrid_total={tokens.get('hybrid_tokens_total', 0)} "
                f"hybrid_visual={tokens.get('hybrid_tokens_visual', 0)} "
                f"hybrid_input_text={tokens.get('hybrid_tokens_input_text', 0)} "
                f"hybrid_output={tokens.get('hybrid_tokens_output', 0)}"
            ),
            (
                f"mean_per_step_total={tokens.get('mean_hybrid_tokens_total_per_step', 0.0):.2f} "
                f"visual={tokens.get('mean_hybrid_tokens_visual_per_step', 0.0):.2f} "
                f"input_text={tokens.get('mean_hybrid_tokens_input_text_per_step', 0.0):.2f} "
                f"output={tokens.get('mean_hybrid_tokens_output_per_step', 0.0):.2f}"
            ),
            (
                f"total_token_reduction={_format_rate(tokens.get('total_token_reduction'))} "
                f"visual_token_reduction={_format_rate(tokens.get('visual_token_reduction'))}"
            ),
            "",
            "[latency]",
            (
                f"hybrid_step_lat_ms_total={_format_ms(latency.get('hybrid_end_to_end_latency_ms_total'))} "
                f"mean_step_lat_ms={_format_ms(latency.get('mean_step_lat_ms'))}"
            ),
            (
                f"latency_reduction={_format_rate(latency.get('latency_reduction'))} "
                f"speedup_vs_baseline={_format_rate(latency.get('speedup_vs_baseline'))}"
            ),
            SUMMARY_FOOTER,
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers for VisionZip-style build_log_summary
# ---------------------------------------------------------------------------

def _build_run_metadata(
    report: Mapping[str, Any],
    *,
    backend_config: Any | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "benchmark": report.get("benchmark"),
        "evaluation_split": report.get("evaluation_split"),
        "variant_id": report.get("variant_id"),
        "variant_description": report.get("variant_description"),
        "result_count": int(report.get("result_count") or 0),
    }
    metadata.update(_extract_backend_metadata(backend_config))
    if report.get("dataset_manifest"):
        metadata["dataset_manifest"] = report.get("dataset_manifest")
    return metadata


def _extract_backend_metadata(backend_config: Any | None) -> dict[str, Any]:
    if backend_config is None:
        return {}

    if is_dataclass(backend_config):
        payload = asdict(backend_config)
    elif isinstance(backend_config, Mapping):
        payload = dict(backend_config)
    else:
        return {}

    extra = dict(payload.get("extra") or {})
    compression_bits: list[str] = []
    for key in ("dominant_ratio", "contextual_ratio", "keep_ratio"):
        if key in extra:
            compression_bits.append(f"{key}={extra[key]}")
    vit_attn = extra.get("vit_attn") or extra.get("vision_attn")

    metadata: dict[str, Any] = {
        "backend": payload.get("kind"),
        "llm_attn": payload.get("attn_implementation"),
    }
    if vit_attn is not None:
        metadata["vit_attn"] = vit_attn
    if compression_bits:
        metadata["compression_target"] = ", ".join(compression_bits)
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _build_overall_accuracy(
    benchmark: str,
    official: Mapping[str, Any],
    null_diag: Mapping[str, Any],
) -> dict[str, Any]:
    if benchmark == "AndroidControl":
        overall = dict(official.get("overall") or {})
        hybrid = dict(overall.get("hybrid") or {})
        baseline = dict(overall.get("baseline") or {})
        return {
            "hl_step_accuracy_hybrid": _pct(hybrid.get("high_level_step_accuracy")),
            "ll_step_accuracy_hybrid": _pct(hybrid.get("low_level_step_accuracy")),
            "hl_step_accuracy_baseline": _pct(baseline.get("high_level_step_accuracy")),
            "ll_step_accuracy_baseline": _pct(baseline.get("low_level_step_accuracy")),
            "hl_episode_accuracy_hybrid": _pct(hybrid.get("high_level_episode_accuracy")),
            "null_action_rate_hybrid": _pct(null_diag.get("hybrid_null_action_rate")),
            "null_action_count_hybrid": null_diag.get("hybrid_null_action_count", "N/A"),
        }

    if benchmark == "LearnGUI":
        summary: dict[str, Any] = {}
        for shot_key in sorted(official):
            shot = dict(official.get(shot_key) or {})
            hybrid = dict(shot.get("hybrid") or {})
            summary[f"{shot_key}_action_match_hybrid"] = _pct(hybrid.get("action_match_accuracy"))
            summary[f"{shot_key}_action_type_hybrid"] = _pct(hybrid.get("action_type_accuracy"))
        return summary

    return {}


def _build_per_subset_accuracy(benchmark: str, official: Mapping[str, Any]) -> list[dict[str, Any]]:
    if benchmark == "AndroidControl":
        entries: list[dict[str, Any]] = []
        for subset, payload in official.items():
            if subset == "overall":
                continue
            hybrid = dict(dict(payload).get("hybrid") or {})
            baseline = dict(dict(payload).get("baseline") or {})
            entries.append(
                {
                    "subset": subset,
                    "metrics": {
                        "hl_step_hybrid": _pct(hybrid.get("high_level_step_accuracy")),
                        "ll_step_hybrid": _pct(hybrid.get("low_level_step_accuracy")),
                        "hl_step_baseline": _pct(baseline.get("high_level_step_accuracy")),
                        "ll_step_baseline": _pct(baseline.get("low_level_step_accuracy")),
                    },
                }
            )
        return entries

    if benchmark == "LearnGUI":
        entries = []
        for shot_key in sorted(official):
            hybrid = dict(dict(official.get(shot_key) or {}).get("hybrid") or {})
            entries.append(
                {
                    "subset": shot_key,
                    "metrics": {
                        "action_match_hybrid": _pct(hybrid.get("action_match_accuracy")),
                        "action_type_hybrid": _pct(hybrid.get("action_type_accuracy")),
                    },
                }
            )
        return entries

    return []


def _build_tokens_compression(controller: Mapping[str, Any], result_count: int) -> dict[str, Any]:
    baseline_visual = int(controller.get("baseline_tokens_visual") or 0)
    hybrid_visual = int(controller.get("hybrid_tokens_visual") or 0)
    baseline_total = int(controller.get("baseline_tokens_total") or 0)
    hybrid_total = int(controller.get("hybrid_tokens_total") or 0)

    def _mean_per_step(total: int) -> str:
        if result_count <= 0:
            return "N/A"
        return f"{float(total) / float(result_count):.1f}"

    visual_retention = _retention(hybrid_visual, baseline_visual)
    total_retention = _retention(hybrid_total, baseline_total)

    return {
        "visual_tokens_total_baseline": baseline_visual,
        "visual_tokens_total_hybrid": hybrid_visual,
        "visual_tokens_per_step_hybrid": _mean_per_step(hybrid_visual),
        "visual_token_retention": _pct(visual_retention),
        "visual_token_compression": _pct(1.0 - visual_retention if visual_retention is not None else None),
        "total_tokens_baseline": baseline_total,
        "total_tokens_hybrid": hybrid_total,
        "total_tokens_per_step_hybrid": _mean_per_step(hybrid_total),
        "total_token_retention": _pct(total_retention),
        "total_token_compression": _pct(1.0 - total_retention if total_retention is not None else None),
        "input_text_tokens_baseline": int(controller.get("baseline_tokens_input_text") or 0),
        "input_text_tokens_hybrid": int(controller.get("hybrid_tokens_input_text") or 0),
        "output_tokens_baseline": int(controller.get("baseline_tokens_output") or 0),
        "output_tokens_hybrid": int(controller.get("hybrid_tokens_output") or 0),
    }


def _build_latency_speedup(controller: Mapping[str, Any], result_count: int) -> dict[str, Any]:
    baseline_ms = float(controller.get("baseline_end_to_end_latency_ms_total") or 0.0)
    hybrid_ms = float(controller.get("hybrid_end_to_end_latency_ms_total") or 0.0)

    def _mean_per_step(total_ms: float) -> str:
        if result_count <= 0:
            return "N/A"
        return f"{total_ms / float(result_count):.1f}"

    speedup = baseline_ms / hybrid_ms if hybrid_ms > 0.0 else None
    return {
        "baseline_e2e_latency_ms_total": f"{baseline_ms:.0f}",
        "hybrid_e2e_latency_ms_total": f"{hybrid_ms:.0f}",
        "baseline_mean_step_lat_ms": _mean_per_step(baseline_ms),
        "hybrid_mean_step_lat_ms": _mean_per_step(hybrid_ms),
        "hybrid_speedup_vs_baseline": f"{speedup:.2f}x" if speedup is not None else "N/A",
        "latency_reduction": _pct(controller.get("latency_reduction")),
        "model_only_latency_reduction": _pct(controller.get("model_only_latency_reduction")),
    }


def _resolve_eval_json_status(path: str | Path | None, source: str) -> dict[str, Any]:
    if path is None:
        return {"path": None, "source": source, "exists": False, "readable": False}
    resolved = Path(path)
    exists = resolved.exists()
    readable = False
    if exists:
        try:
            json.loads(resolved.read_text(encoding="utf-8"))
            readable = True
        except (OSError, json.JSONDecodeError):
            readable = False
    return {
        "path": str(resolved),
        "source": source,
        "exists": exists,
        "readable": readable,
    }
