#!/usr/bin/env python3
"""Compare per-phase timing breakdowns from baseline vs VisionZip evaluation runs.

Usage:
  python scripts/androidcontrol/timing/compare_timing_breakdown.py \\
      --baseline outputs/timing_baseline_eager/ \\
      --visionzip outputs/timing_visionzip_eager/ \\
      [--csv outputs/timing_comparison.csv]

Reads the `timing_summary` field from `*_evaluation.json` files in each directory,
then prints a Markdown comparison table and optionally writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Phase display configuration
# ---------------------------------------------------------------------------

PHASES = [
    ("image_prep_ms",        "Image preprocessing"),
    ("message_build_ms",     "Message build"),
    ("tokenization_ms",      "Tokenization"),
    ("gpu_transfer_ms",      "GPU transfer"),
    ("model_generate_ms",    "Model generate (total)"),
    ("vision_encoder_ms",    "  Vision encoder"),
    ("attn_scoring_prune_ms","  Attn scoring+prune"),
    ("llm_forward_ms",       "  LLM forward"),
    ("output_decode_ms",     "Output decode"),
    ("output_parse_ms",      "Output parsing"),
    ("total_e2e_step_ms",    "**Total e2e / step**"),
]

# Phases that only exist for VisionZip
VISIONZIP_ONLY = {"vision_encoder_ms", "attn_scoring_prune_ms", "llm_forward_ms"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-phase timing breakdown between baseline and VisionZip eval runs."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        metavar="DIR",
        help="Directory containing the baseline evaluation JSON.",
    )
    parser.add_argument(
        "--visionzip",
        required=True,
        metavar="DIR",
        help="Directory containing the VisionZip evaluation JSON.",
    )
    parser.add_argument(
        "--csv",
        default="",
        metavar="PATH",
        help="Optional path to write CSV output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_evaluation_json(directory: str) -> Path:
    """Find the first *_evaluation.json in a directory."""
    pattern = str(Path(directory) / "*_evaluation.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # Also try top-level evaluation.json
        alt = Path(directory) / "evaluation.json"
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"No *_evaluation.json found in {directory!r}. "
            "Make sure the directory contains a completed evaluation output."
        )
    return Path(matches[0])


def load_timing_summary(json_path: Path) -> dict[str, dict[str, float]]:
    """Load and return the timing_summary section from an evaluation JSON."""
    data = json.loads(json_path.read_text())
    summary = data.get("timing_summary")
    if not summary:
        raise ValueError(
            f"No 'timing_summary' key found in {json_path}. "
            "Was the evaluation run with --measure-e2e-latency?"
        )
    return summary


def fmt_ms(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}"


def fmt_speedup(baseline: float | None, vz: float | None) -> str:
    if baseline is None or vz is None or vz == 0.0:
        return "—"
    return f"{baseline / vz:.2f}x"


def print_markdown_table(rows: list[tuple[str, str, str, str]]) -> None:
    """Print a Markdown table from (phase_label, baseline_ms, vz_ms, speedup) rows."""
    col_widths = [
        max(len(r[0]) for r in rows) + 1,
        max(len(r[1]) for r in rows) + 2,
        max(len(r[2]) for r in rows) + 2,
        max(len(r[3]) for r in rows) + 2,
    ]
    col_widths = [max(w, 20) for w in col_widths]
    col_widths[0] = max(col_widths[0], 22)

    header = (
        f"| {'Phase':<{col_widths[0]}} "
        f"| {'Baseline (ms)':>{col_widths[1]}} "
        f"| {'VisionZip (ms)':>{col_widths[2]}} "
        f"| {'Speedup':>{col_widths[3]}} |"
    )
    separator = (
        f"|{'-' * (col_widths[0] + 2)}"
        f"|{'-' * (col_widths[1] + 2)}"
        f"|{'-' * (col_widths[2] + 2)}"
        f"|{'-' * (col_widths[3] + 2)}|"
    )
    print(header)
    print(separator)
    for label, baseline, vz, speedup in rows:
        print(
            f"| {label:<{col_widths[0]}} "
            f"| {baseline:>{col_widths[1]}} "
            f"| {vz:>{col_widths[2]}} "
            f"| {speedup:>{col_widths[3]}} |"
        )


def write_csv(path: str, rows: list[tuple[str, str, str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Baseline_ms", "VisionZip_ms", "Speedup"])
        writer.writerows(rows)
    print(f"\nCSV written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # ---- Locate evaluation JSONs ----
    try:
        baseline_path = find_evaluation_json(args.baseline)
        vz_path = find_evaluation_json(args.visionzip)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Baseline JSON : {baseline_path}")
    print(f"VisionZip JSON: {vz_path}")

    # ---- Load timing summaries ----
    try:
        baseline_summary = load_timing_summary(baseline_path)
        vz_summary = load_timing_summary(vz_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ---- Build table rows ----
    rows: list[tuple[str, str, str, str]] = []
    for phase_key, phase_label in PHASES:
        baseline_entry = baseline_summary.get(phase_key)
        vz_entry = vz_summary.get(phase_key)

        baseline_mean = baseline_entry["mean_ms"] if baseline_entry else None
        vz_mean = vz_entry["mean_ms"] if vz_entry else None

        # VisionZip-only sub-phases show "—" for baseline
        if phase_key in VISIONZIP_ONLY:
            baseline_mean = None

        rows.append((
            phase_label,
            fmt_ms(baseline_mean),
            fmt_ms(vz_mean),
            fmt_speedup(baseline_mean, vz_mean),
        ))

    # ---- Print Markdown table ----
    print()
    print_markdown_table(rows)

    # ---- Count info ----
    baseline_count = 0
    vz_count = 0
    for phase_key, _ in PHASES:
        if phase_key not in VISIONZIP_ONLY:
            entry = baseline_summary.get(phase_key)
            if entry:
                baseline_count = int(entry.get("count", 0))
                break
    for phase_key, _ in PHASES:
        entry = vz_summary.get(phase_key)
        if entry:
            vz_count = int(entry.get("count", 0))
            break

    print(f"\n  Baseline: {baseline_count} steps | VisionZip: {vz_count} steps")

    # ---- Overall speedup ----
    baseline_e2e = baseline_summary.get("total_e2e_step_ms", {}).get("mean_ms")
    vz_e2e = vz_summary.get("total_e2e_step_ms", {}).get("mean_ms")
    if baseline_e2e and vz_e2e:
        overall = baseline_e2e / vz_e2e
        print(f"  Overall speedup (total_e2e_step_ms): {overall:.2f}x")

    # ---- Optional CSV ----
    if args.csv:
        try:
            write_csv(args.csv, rows)
        except OSError as exc:
            print(f"WARNING: Could not write CSV — {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
