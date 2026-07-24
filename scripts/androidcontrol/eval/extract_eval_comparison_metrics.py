#!/usr/bin/env python3
"""Extract AndroidControl V0 eval metrics for four-way comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def _pct_pp(delta: float | None) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else "−"
    return f"{sign}{abs(delta * 100):.1f} pp"


def _rel_pct(delta: float | None) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else "−"
    return f"{sign}{abs(delta * 100):.1f}%"


def _int_fmt(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def _float1(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}"


def load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    result_count = int(report.get("result_count") or 0)
    official = dict(report.get("official_metrics") or {})
    controller = dict(report.get("controller_metrics") or {})
    null_diag = dict(report.get("null_action_diagnostics") or {})
    log_summary = dict(report.get("log_summary") or {})
    accuracy = dict(log_summary.get("accuracy") or {})
    by_subset = dict(accuracy.get("by_subset") or {})

    def subset_hl_ll(name: str) -> tuple[float | None, float | None]:
        block = by_subset.get(name) or official.get(name) or {}
        if isinstance(block, dict) and "baseline" in block:
            block = block["baseline"]
        if not isinstance(block, dict):
            return None, None
        return block.get("high_level_step_accuracy"), block.get("low_level_step_accuracy")

    overall_block = by_subset.get("overall") or official.get("overall") or {}
    if isinstance(overall_block, dict) and "baseline" in overall_block:
        overall_block = overall_block["baseline"]
    hl_overall = overall_block.get("high_level_step_accuracy") if isinstance(overall_block, dict) else None
    ll_overall = overall_block.get("low_level_step_accuracy") if isinstance(overall_block, dict) else None
    hl_ep = overall_block.get("high_level_episode_accuracy") if isinstance(overall_block, dict) else None

    null_rate = null_diag.get("baseline_null_action_rate")
    null_count = null_diag.get("baseline_null_action_count")

    hybrid_visual = int(controller.get("hybrid_tokens_visual") or controller.get("baseline_tokens_visual") or 0)
    hybrid_total = int(controller.get("hybrid_tokens_total") or controller.get("baseline_tokens_total") or 0)
    hybrid_input = int(controller.get("hybrid_tokens_input_text") or controller.get("baseline_tokens_input_text") or 0)
    hybrid_output = int(controller.get("hybrid_tokens_output") or controller.get("baseline_tokens_output") or 0)
    steps = max(1, result_count)

    subsets: dict[str, dict[str, float | None]] = {}
    for name in ("in_distribution", "app_unseen", "category_unseen", "task_unseen", "overall"):
        hl, ll = subset_hl_ll(name)
        subsets[name] = {"hl": hl, "ll": ll}

    return {
        "result_count": result_count,
        "hl_step_accuracy": hl_overall,
        "ll_step_accuracy": ll_overall,
        "hl_episode_accuracy": hl_ep,
        "null_action_rate": null_rate,
        "null_action_count": null_count,
        "visual_tokens_total": hybrid_visual,
        "visual_tokens_per_step": hybrid_visual / steps,
        "total_tokens": hybrid_total,
        "total_tokens_per_step": hybrid_total / steps,
        "input_text_tokens": hybrid_input,
        "output_tokens": hybrid_output,
        "subsets": subsets,
    }


def delta_pp(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def delta_rel(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return (float(b) - float(a)) / float(a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_json", type=Path, help="Path to androidcontrol_test_v0_evaluation.json")
    parser.add_argument("--label", default="", help="Optional label for stdout")
    args = parser.parse_args()

    metrics = extract_metrics(load_eval(args.eval_json))
    label = args.label or args.eval_json.parent.name
    print(f"# {label} ({metrics['result_count']} steps)")
    print(f"HL step: {_pct(metrics['hl_step_accuracy'])}")
    print(f"LL step: {_pct(metrics['ll_step_accuracy'])}")
    print(f"Null: {_pct(metrics['null_action_rate'])} ({metrics['null_action_count']})")
    print(f"Visual/step: {_float1(metrics['visual_tokens_per_step'])}")
    print(f"Total/step: {_float1(metrics['total_tokens_per_step'])}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
