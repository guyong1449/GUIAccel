#!/usr/bin/env python3
"""Compare prefill-related timing between baseline and VisionZip eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PREFILL_PHASES = (
    "prefill_ms",
    "llm_forward_ms",
    "model_generate_ms",
    "vision_encoder_ms",
    "attn_scoring_prune_ms",
    "decode_ms",
    "total_e2e_step_ms",
)


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} does not contain a JSON object")
    return dict(payload)


def _resolve_report_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidates = sorted(path.glob("androidcontrol_*_evaluation.json"))
        if not candidates:
            candidates = sorted(path.glob("androidcontrol_*_v0.json"))
        if not candidates:
            candidates = [
                candidate
                for candidate in sorted(path.glob("*.json"))
                if candidate.name
                not in {
                    "parallel_evaluation_progress.json",
                    "config_snapshot.json",
                    "detailed_results.json",
                }
            ]
        if not candidates:
            raise FileNotFoundError(f"No evaluation JSON found under {path}")
        return candidates[-1].resolve()
    raise FileNotFoundError(f"Path does not exist: {path}")


def _phase_mean(timing_summary: Mapping[str, Any], phase: str) -> float | None:
    baseline = dict(timing_summary.get("baseline") or {})
    entry = baseline.get(phase)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("mean_ms")
    return float(value) if value is not None else None


def _format_row(label: str, baseline: float | None, visionzip: float | None) -> str:
    if baseline is None and visionzip is None:
        return f"| {label} | — | — | — |"
    base_str = f"{baseline:.2f}" if baseline is not None else "—"
    vz_str = f"{visionzip:.2f}" if visionzip is not None else "—"
    if baseline is not None and visionzip is not None and baseline > 0:
        speedup = baseline / visionzip
        delta_pct = (1.0 - visionzip / baseline) * 100.0
        delta_str = f"{speedup:.2f}x ({delta_pct:+.1f}%)"
    else:
        delta_str = "—"
    return f"| {label} | {base_str} | {vz_str} | {delta_str} |"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare prefill-related timing summary between eval outputs.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline eval JSON or output directory")
    parser.add_argument("--visionzip", type=Path, required=True, help="VisionZip eval JSON or output directory")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path")
    args = parser.parse_args()

    baseline_report = _load_report(_resolve_report_path(args.baseline))
    visionzip_report = _load_report(_resolve_report_path(args.visionzip))
    baseline_timing = dict(baseline_report.get("timing_summary") or {})
    visionzip_timing = dict(visionzip_report.get("timing_summary") or {})

    lines = [
        "# Prefill-related timing comparison",
        "",
        f"- Baseline steps: {baseline_report.get('result_count', '?')}",
        f"- VisionZip steps: {visionzip_report.get('result_count', '?')}",
        "",
        "| Phase (mean ms / step) | Baseline | VisionZip | Speedup / savings |",
        "| --- | ---: | ---: | ---: |",
    ]
    for phase in PREFILL_PHASES:
        lines.append(
            _format_row(
                phase,
                _phase_mean(baseline_timing, phase),
                _phase_mean(visionzip_timing, phase),
            )
        )

    baseline_prefill = _phase_mean(baseline_timing, "prefill_ms")
    visionzip_prefill = _phase_mean(visionzip_timing, "prefill_ms")
    baseline_model_gen = _phase_mean(baseline_timing, "model_generate_ms")
    visionzip_model_gen = _phase_mean(visionzip_timing, "model_generate_ms")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- VisionZip should reduce LLM prefill cost by feeding fewer visual tokens into the text model.",
            "- Both backends now record `prefill_ms` / `decode_ms` on `language_model.forward`.",
        ]
    )
    if baseline_prefill is not None and visionzip_prefill is not None and visionzip_prefill > 0:
        speedup = baseline_prefill / visionzip_prefill
        delta_pct = (1.0 - visionzip_prefill / baseline_prefill) * 100.0 if baseline_prefill > 0 else 0.0
        lines.append(
            f"- Direct prefill comparison: baseline {baseline_prefill:.2f} ms vs "
            f"VisionZip {visionzip_prefill:.2f} ms → {speedup:.2f}x ({delta_pct:+.1f}%)."
        )
    elif baseline_prefill is None and baseline_model_gen is not None and visionzip_prefill is not None:
        speedup = baseline_model_gen / visionzip_prefill if visionzip_prefill > 0 else 0.0
        lines.append(
            f"- Proxy: baseline `model_generate_ms` ({baseline_model_gen:.2f} ms) "
            f"vs VisionZip `prefill_ms` ({visionzip_prefill:.2f} ms) ≈ {speedup:.2f}x."
        )

    markdown = "\n".join(lines) + "\n"
    print(markdown)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
