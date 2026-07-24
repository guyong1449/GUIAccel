#!/usr/bin/env python3
"""Convert SkillReuse orchestrator output to the standard baseline JSON format.

The script accepts the summary evaluation JSON produced by ``run_evaluation.py``
(via ``save_evaluation_report()``) and an optional detailed-results JSON produced
by ``skillreuse.evaluation.detailed_results.save_detailed_results()``.

* With detailed results → full output including per-task and per-step fields.
* Without detailed results → summary-only output (tasks array omitted); a
  WARNING is printed because the output cannot be independently validated.

Usage examples
--------------
Full export (recommended):

    python scripts/infra/export_baseline_json.py \\
        --input  outputs/divprune_keep0.098/androidcontrol_test_v0_evaluation.json \\
        --method divprune \\
        --backbone "Qwen3-VL-8B-Instruct" \\
        --output  outputs/divprune_keep0.098/baseline_output.json \\
        --notes   "DivPrune (CVPR 2025) MMDP greedy selection, keep_ratio=0.098" \\
        --detailed-results outputs/divprune_keep0.098/detailed_results.json

Summary-only export (no detailed results available):

    python scripts/infra/export_baseline_json.py \\
        --input  outputs/baseline_v0/androidcontrol_test_v0_evaluation.json \\
        --method baseline_transformers \\
        --backbone "Qwen3-VL-8B-Instruct" \\
        --output  outputs/baseline_v0/baseline_output.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Summary extraction from orchestrator report
# ---------------------------------------------------------------------------

def _extract_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Pull step_accuracy and task_success_rate per instruction_mode from the report."""
    official = report.get("official_metrics", {})
    overall = official.get("overall", {})
    baseline_overall = overall.get("baseline", {})

    return {
        "high_level": {
            "step_accuracy": _safe_float(baseline_overall.get("high_level_step_accuracy")),
            "task_success_rate": _safe_float(baseline_overall.get("high_level_episode_accuracy")),
        },
        "low_level": {
            "step_accuracy": _safe_float(baseline_overall.get("low_level_step_accuracy")),
        },
    }


# ---------------------------------------------------------------------------
# Detailed-results processing
# ---------------------------------------------------------------------------

def _build_tasks_from_detailed(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group flat step records by (episode_id, instruction_mode) → task dicts."""
    # Key: (episode_id, instruction_mode) to keep high-level and low-level separate.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        key = (str(step["episode_id"]), str(step.get("instruction_mode", "unknown")))
        grouped[key].append(step)

    tasks: list[dict[str, Any]] = []
    for (episode_id, instruction_mode), episode_steps in sorted(grouped.items()):
        # Sort by step_index within the episode.
        episode_steps.sort(key=lambda s: _safe_int(s["step_index"]))

        step_records: list[dict[str, Any]] = []
        for raw in episode_steps:
            step_records.append(
                {
                    "step_index": _safe_int(raw["step_index"]),
                    "correct": bool(raw["baseline_correct"]),
                    "latency": {
                        "step_latency_s": _safe_float(raw["latency_ms"]) / 1000.0,
                    },
                    "token": {
                        "input_tokens": _safe_int(raw["input_tokens"]),
                        "output_tokens": _safe_int(raw["output_tokens"]),
                    },
                }
            )

        n_steps = len(step_records)
        task_success = all(s["correct"] for s in step_records)
        step_accuracy = (
            sum(1 for s in step_records if s["correct"]) / n_steps
            if n_steps > 0
            else 0.0
        )
        task_latency_s = sum(
            s["latency"]["step_latency_s"] for s in step_records
        )
        task_input_tokens = sum(s["token"]["input_tokens"] for s in step_records)
        task_output_tokens = sum(s["token"]["output_tokens"] for s in step_records)

        # goal: use the first step's goal (consistent for the whole episode).
        goal = str(episode_steps[0].get("goal", ""))

        tasks.append(
            {
                "task_id": episode_id,
                "instruction_mode": instruction_mode,
                "goal": goal,
                "n_steps": n_steps,
                "task_success": task_success,
                "step_accuracy": step_accuracy,
                "latency": {"task_latency_s": task_latency_s},
                "token": {
                    "task_input_tokens": task_input_tokens,
                    "task_output_tokens": task_output_tokens,
                },
                "steps": step_records,
            }
        )

    return tasks


def _aggregate_summary_from_tasks(
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive all summary metrics from the tasks array."""
    n_tasks = len(tasks)
    if n_tasks == 0:
        return {
            "step_accuracy": 0.0,
            "task_success_rate": 0.0,
            "latency": {"avg_task_latency_s": 0.0, "avg_step_latency_s": 0.0},
            "token": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "avg_task_input_tokens": 0.0,
                "avg_task_output_tokens": 0.0,
                "avg_step_input_tokens": 0.0,
                "avg_step_output_tokens": 0.0,
            },
        }

    total_steps = sum(t["n_steps"] for t in tasks)
    total_correct = sum(
        sum(1 for s in t["steps"] if s["correct"]) for t in tasks
    )
    total_successful_tasks = sum(1 for t in tasks if t["task_success"])

    total_input_tokens = sum(t["token"]["task_input_tokens"] for t in tasks)
    total_output_tokens = sum(t["token"]["task_output_tokens"] for t in tasks)

    total_task_latency = sum(t["latency"]["task_latency_s"] for t in tasks)
    total_step_latency = sum(
        s["latency"]["step_latency_s"] for t in tasks for s in t["steps"]
    )

    step_accuracy = total_correct / total_steps if total_steps > 0 else 0.0
    task_success_rate = total_successful_tasks / n_tasks

    avg_task_latency_s = total_task_latency / n_tasks
    avg_step_latency_s = total_step_latency / total_steps if total_steps > 0 else 0.0

    avg_task_input_tokens = total_input_tokens / n_tasks
    avg_task_output_tokens = total_output_tokens / n_tasks
    avg_step_input_tokens = total_input_tokens / total_steps if total_steps > 0 else 0.0
    avg_step_output_tokens = total_output_tokens / total_steps if total_steps > 0 else 0.0

    return {
        "step_accuracy": step_accuracy,
        "task_success_rate": task_success_rate,
        "latency": {
            "avg_task_latency_s": avg_task_latency_s,
            "avg_step_latency_s": avg_step_latency_s,
        },
        "token": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_task_input_tokens": avg_task_input_tokens,
            "avg_task_output_tokens": avg_task_output_tokens,
            "avg_step_input_tokens": avg_step_input_tokens,
            "avg_step_output_tokens": avg_step_output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cross_check(
    summary_from_report: dict[str, Any],
    tasks: list[dict[str, Any]],
    tolerance: float = 0.002,
) -> None:
    """Warn when per-mode metrics from tasks deviate from the report."""
    for mode in ("high_level", "low_level"):
        report_mode = summary_from_report.get(mode, {})
        mode_tasks = [t for t in tasks if t.get("instruction_mode") == mode]
        if not mode_tasks or not report_mode:
            continue

        mode_steps = [s for t in mode_tasks for s in t["steps"]]
        if mode_steps:
            tasks_step_acc = sum(1 for s in mode_steps if s["correct"]) / len(mode_steps)
            report_step_acc = _safe_float(report_mode.get("step_accuracy"))
            if abs(report_step_acc - tasks_step_acc) > tolerance:
                warnings.warn(
                    f"{mode} step_accuracy mismatch: report={report_step_acc:.4f}, "
                    f"tasks={tasks_step_acc:.4f}  "
                    f"(delta={abs(report_step_acc - tasks_step_acc):.4f})",
                    stacklevel=2,
                )

        if "task_success_rate" in report_mode:
            tasks_tsr = sum(1 for t in mode_tasks if t["task_success"]) / len(mode_tasks)
            report_tsr = _safe_float(report_mode.get("task_success_rate"))
            if abs(report_tsr - tasks_tsr) > tolerance:
                warnings.warn(
                    f"{mode} task_success_rate mismatch: report={report_tsr:.4f}, "
                    f"tasks={tasks_tsr:.4f}  "
                    f"(delta={abs(report_tsr - tasks_tsr):.4f})",
                    stacklevel=2,
                )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    *,
    input_path: Path,
    output_path: Path,
    method: str,
    backbone: str,
    notes: str = "",
    detailed_results_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full conversion and write output_path.

    Returns the produced output dict.
    """
    report = _load_json(input_path)

    benchmark = str(report.get("benchmark", "AndroidControl"))
    split = str(report.get("evaluation_split", "test"))
    result_count = _safe_int(report.get("result_count"))

    summary_from_report = _extract_summary_from_report(report)

    tasks: list[dict[str, Any]] | None = None
    summary: dict[str, Any]

    if detailed_results_path is not None:
        detailed = _load_json(detailed_results_path)
        steps: list[dict[str, Any]] = detailed.get("steps", [])
        if not steps:
            warnings.warn(
                f"Detailed results file '{detailed_results_path}' contains no steps. "
                "Falling back to summary-only output.",
                stacklevel=2,
            )
            summary = _build_summary_only(summary_from_report)
        else:
            tasks = _build_tasks_from_detailed(steps)
            summary_from_tasks = _aggregate_summary_from_tasks(tasks)
            _cross_check(summary_from_report, tasks)
            # Use the tasks-derived summary as ground truth (it is per-step
            # granular and directly auditable from the tasks array).
            summary = summary_from_tasks
    else:
        warnings.warn(
            "No --detailed-results provided. The output will lack the 'tasks' array "
            "and per-step metrics.  Latency and token fields will be null.",
            stacklevel=2,
        )
        summary = _build_summary_only(summary_from_report)

    n_tasks: int
    if tasks is not None:
        # Each (episode_id, instruction_mode) pair is one evaluation unit.
        n_tasks = len(tasks)
    else:
        # Best effort from the official metric counts (summary-only path).
        official_overall_counts = (
            report.get("official_metrics", {})
            .get("overall", {})
            .get("counts", {})
        )
        n_tasks = _safe_int(
            official_overall_counts.get("high_level_episodes", result_count)
        )

    output: dict[str, Any] = {
        "meta": {
            "benchmark": benchmark,
            "method": method,
            "backbone": backbone,
            "split": split,
            "n_tasks": n_tasks,
            "notes": notes,
        },
        "summary": summary,
    }
    if tasks is not None:
        output["tasks"] = tasks

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def _build_summary_only(summary_from_report: dict[str, Any]) -> dict[str, Any]:
    """Produce a summary dict with accuracy from the report and null latency/token."""
    hl = summary_from_report.get("high_level", {})
    return {
        "step_accuracy": _safe_float(hl.get("step_accuracy")),
        "task_success_rate": _safe_float(hl.get("task_success_rate")),
        "latency": {
            "avg_task_latency_s": None,
            "avg_step_latency_s": None,
        },
        "token": {
            "total_input_tokens": None,
            "total_output_tokens": None,
            "avg_task_input_tokens": None,
            "avg_task_output_tokens": None,
            "avg_step_input_tokens": None,
            "avg_step_output_tokens": None,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_baseline_json.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help=(
            "Path to the orchestrator evaluation JSON "
            "(e.g. outputs/divprune_keep0.098/androidcontrol_test_v0_evaluation.json)"
        ),
    )
    parser.add_argument(
        "--method",
        required=True,
        metavar="NAME",
        help=(
            "Method identifier, e.g. 'divprune', 'baseline_transformers', "
            "'similarity-pruning'."
        ),
    )
    parser.add_argument(
        "--backbone",
        required=True,
        metavar="NAME",
        help="Base model name, e.g. 'Qwen3-VL-8B-Instruct'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the standard baseline JSON.",
    )
    parser.add_argument(
        "--notes",
        default="",
        metavar="TEXT",
        help=(
            "Optional free-text notes stored in meta.notes, "
            "e.g. \"DivPrune (CVPR 2025) MMDP greedy selection, keep_ratio=0.098\"."
        ),
    )
    parser.add_argument(
        "--detailed-results",
        default=None,
        metavar="PATH",
        dest="detailed_results",
        help=(
            "Path to the per-step detailed results JSON produced by "
            "skillreuse.evaluation.detailed_results.save_detailed_results(). "
            "When omitted, the output will lack the 'tasks' array and "
            "latency/token fields will be null."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    detailed_results_path = (
        Path(args.detailed_results) if args.detailed_results else None
    )

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    if detailed_results_path is not None and not detailed_results_path.exists():
        print(
            f"ERROR: detailed-results file not found: {detailed_results_path}",
            file=sys.stderr,
        )
        return 1

    result = convert(
        input_path=input_path,
        output_path=output_path,
        method=args.method,
        backbone=args.backbone,
        notes=args.notes,
        detailed_results_path=detailed_results_path,
    )

    summary = result["summary"]
    tasks = result.get("tasks")
    print(f"Written: {output_path}")
    print(f"  benchmark      : {result['meta']['benchmark']}")
    print(f"  method         : {result['meta']['method']}")
    print(f"  backbone        : {result['meta']['backbone']}")
    print(f"  split          : {result['meta']['split']}")
    print(f"  n_tasks        : {result['meta']['n_tasks']}")
    print(f"  step_accuracy  : {summary['step_accuracy']:.4f}")
    print(f"  task_success   : {summary['task_success_rate']:.4f}")
    if tasks is not None:
        print(f"  tasks written  : {len(tasks)} (episode × instruction_mode)")
    else:
        print("  tasks written  : (none — no --detailed-results provided)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
