#!/usr/bin/env python3
"""Compare per-step latency between local_transformers baseline and VisionZip eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} does not contain a JSON object")
    return dict(payload)


def _extract_latency(report: Mapping[str, Any]) -> dict[str, float | int]:
    controller = dict(report.get("controller_metrics") or {})
    result_count = int(report.get("result_count") or 0)
    baseline_total_ms = float(controller.get("baseline_end_to_end_latency_ms_total") or 0.0)
    hybrid_total_ms = float(controller.get("hybrid_end_to_end_latency_ms_total") or 0.0)
    avg_baseline_ms = baseline_total_ms / result_count if result_count else 0.0
    avg_hybrid_ms = hybrid_total_ms / result_count if result_count else 0.0
    return {
        "result_count": result_count,
        "baseline_end_to_end_latency_ms_total": baseline_total_ms,
        "hybrid_end_to_end_latency_ms_total": hybrid_total_ms,
        "avg_step_latency_ms": avg_baseline_ms,
        "avg_step_latency_s": avg_baseline_ms / 1000.0,
        "latency_reduction": float(controller.get("latency_reduction") or 0.0),
    }


def _resolve_report_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidates = sorted(path.glob("androidcontrol_*_v0.json"))
        if not candidates:
            candidates = sorted(path.glob("*.json"))
            candidates = [
                candidate
                for candidate in candidates
                if candidate.name not in {"parallel_evaluation_progress.json", "config_snapshot.json"}
            ]
        if not candidates:
            raise FileNotFoundError(f"No evaluation JSON found under {path}")
        return candidates[-1].resolve()
    raise FileNotFoundError(f"Path does not exist: {path}")


def _format_markdown_table(
    *,
    baseline_label: str,
    visionzip_label: str,
    baseline: Mapping[str, float | int],
    visionzip: Mapping[str, float | int],
    speedup: float,
) -> str:
    lines = [
        "| Metric | Baseline (local_transformers) | VisionZip | Speedup |",
        "| --- | ---: | ---: | ---: |",
        f"| Steps evaluated | {baseline['result_count']} | {visionzip['result_count']} | — |",
        (
            f"| Total e2e latency (s) | "
            f"{baseline['baseline_end_to_end_latency_ms_total'] / 1000.0:.1f} | "
            f"{visionzip['baseline_end_to_end_latency_ms_total'] / 1000.0:.1f} | "
            f"{speedup:.2f}x |"
        ),
        (
            f"| Avg step latency (ms) | "
            f"{baseline['avg_step_latency_ms']:.1f} | "
            f"{visionzip['avg_step_latency_ms']:.1f} | "
            f"{speedup:.2f}x |"
        ),
        (
            f"| Avg step latency (s) | "
            f"{baseline['avg_step_latency_s']:.3f} | "
            f"{visionzip['avg_step_latency_s']:.3f} | "
            f"{speedup:.2f}x |"
        ),
        "",
        f"**Speedup** = `{baseline_label}` avg step latency / `{visionzip_label}` avg step latency = **{speedup:.2f}x**",
        "",
        "_Latency measured via `FullModelResponse.latency_ms` aggregated in `controller_metrics.baseline_end_to_end_latency_ms_total` (V0 eval)._",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute VisionZip speedup vs local_transformers baseline from evaluation JSON outputs."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline_transformers evaluation JSON or output directory",
    )
    parser.add_argument(
        "--visionzip",
        type=Path,
        required=True,
        help="Path to visionzip evaluation JSON or output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = _resolve_report_path(args.baseline)
    visionzip_path = _resolve_report_path(args.visionzip)

    baseline_report = _load_report(baseline_path)
    visionzip_report = _load_report(visionzip_path)

    baseline = _extract_latency(baseline_report)
    visionzip = _extract_latency(visionzip_report)

    if baseline["avg_step_latency_ms"] <= 0.0:
        raise SystemExit(f"Baseline report has zero latency: {baseline_path}")
    if visionzip["avg_step_latency_ms"] <= 0.0:
        raise SystemExit(f"VisionZip report has zero latency: {visionzip_path}")

    speedup = float(baseline["avg_step_latency_ms"]) / float(visionzip["avg_step_latency_ms"])

    print(_format_markdown_table(
        baseline_label=str(baseline_path),
        visionzip_label=str(visionzip_path),
        baseline=baseline,
        visionzip=visionzip,
        speedup=speedup,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
