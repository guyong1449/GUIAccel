#!/usr/bin/env python3
"""Dry-run partial report aggregation from existing eval checkpoint pkls.

Usage:
    python scripts/androidcontrol/timing/dryrun_partial_report.py \\
        --output-dir outputs/divprune_eager_timing_smoke1pct_20260629_025925
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    _lib = _parent / "_lib"
    if (_lib / "repo_path.py").is_file():
        sys.path.insert(0, str(_lib))
        break
else:
    raise RuntimeError(f"Could not locate scripts/_lib from {__file__}")
from repo_path import bootstrap

REPO_ROOT = bootstrap(Path(__file__))
sys.path.insert(0, str(REPO_ROOT))

from skillreuse.evaluation.controller_metrics import compute_controller_metrics
from skillreuse.evaluation.partial_report import (
    PartialReportContext,
    build_worker_partial_report,
    write_aggregated_partial_reports,
)
from skillreuse.evaluation.timing import compute_timing_summary


def _load_all_results(output_dir: Path):
    chunk_dir = output_dir / "parallel_evaluation_chunks"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Missing chunk dir: {chunk_dir}")
    results = []
    for path in sorted(chunk_dir.glob("*.pkl")):
        payload = pickle.loads(path.read_bytes())
        results.extend(payload.get("results") or ())
    if not results:
        raise ValueError(f"No results found under {chunk_dir}")
    return results


def _simulate_gpu_reports(results, chunk_dir: Path):
    by_gpu: dict[int, list] = {}
    for path in sorted(chunk_dir.glob("*.pkl")):
        name = path.name
        if "_gpu" not in name:
            continue
        gpu_token = name.split("_gpu", 1)[1]
        gpu_id = int(gpu_token.split("_chunk", 1)[0])
        payload = pickle.loads(path.read_bytes())
        by_gpu.setdefault(gpu_id, []).extend(payload.get("results") or ())
    if by_gpu:
        return {gpu_id: build_worker_partial_report(group) for gpu_id, group in sorted(by_gpu.items())}
    midpoint = len(results) // 2 or len(results)
    return {
        0: build_worker_partial_report(list(results[:midpoint])),
        1: build_worker_partial_report(list(results[midpoint:])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run partial report aggregation from checkpoint pkls")
    parser.add_argument("--output-dir", required=True, help="Eval output directory with parallel_evaluation_chunks/")
    parser.add_argument("--benchmark", default="AndroidControl")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--variant-id", default="V0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"ERROR: output dir not found: {output_dir}", file=sys.stderr)
        return 1

    chunk_dir = output_dir / "parallel_evaluation_chunks"
    results = _load_all_results(output_dir)
    progress_dir = output_dir / "parallel_evaluation_progress_dryrun"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for old in progress_dir.glob("partial_report_gpu*.json"):
        old.unlink()

    gpu_reports = _simulate_gpu_reports(results, chunk_dir)
    for gpu_id, report in gpu_reports.items():
        (progress_dir / f"partial_report_gpu{gpu_id}.json").write_text(
            json.dumps({"gpu_id": gpu_id, **report}, indent=2),
            encoding="utf-8",
        )

    context = PartialReportContext(
        benchmark=args.benchmark,
        evaluation_split=args.evaluation_split,
        variant_id=args.variant_id,
    )
    written = write_aggregated_partial_reports(
        progress_dir=progress_dir,
        output_dir=output_dir,
        benchmark=context.benchmark,
        evaluation_split=context.evaluation_split,
        variant_id=context.variant_id,
        variant_description="dryrun",
    )
    expected_timing = compute_timing_summary(results)
    expected_controller = compute_controller_metrics(results)
    partial_timing = json.loads(written["partial_timing_summary"].read_text(encoding="utf-8"))
    partial_controller = json.loads(written["partial_controller_metrics"].read_text(encoding="utf-8"))

    got_mean = partial_timing["timing_summary"]["baseline"]["total_e2e_step_ms"]["mean_ms"]
    exp_mean = expected_timing["baseline"]["total_e2e_step_ms"]["mean_ms"]
    got_tokens = partial_controller["controller_metrics"]["baseline_tokens_total"]
    exp_tokens = expected_controller["baseline_tokens_total"]

    print(f"[dryrun_partial_report] results={len(results)} gpu_reports={len(gpu_reports)}")
    print(f"[dryrun_partial_report] total_e2e mean_ms: partial={got_mean:.4f} expected={exp_mean:.4f}")
    print(f"[dryrun_partial_report] baseline_tokens_total: partial={got_tokens} expected={exp_tokens}")
    for key, path in written.items():
        print(f"[dryrun_partial_report] wrote {key}: {path}")

    if abs(got_mean - exp_mean) > 1e-6 or got_tokens != exp_tokens:
        print("FAIL: partial aggregation mismatch", file=sys.stderr)
        return 1
    print("PASS: partial aggregation matches full recompute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
