#!/usr/bin/env python3
"""Run VisionZip V0 baseline evaluation or a single-GPU dry-run smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
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
SUPPORTED_BENCHMARKS = ("AndroidControl", "LearnGUI")
DEFAULT_CONFIGS: dict[str, Path] = {
    "AndroidControl": REPO_ROOT / "configs/androidcontrol/visionzip/default.json",
    "LearnGUI": REPO_ROOT / "configs/learngui/visionzip/default.json",
}
DEFAULT_MODEL_PATH = REPO_ROOT / "models/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VisionZip baseline evaluation (V0, base Qwen3-VL, no LoRA) "
            "or a single-GPU dry-run smoke test."
        )
    )
    parser.add_argument(
        "--benchmark",
        choices=list(SUPPORTED_BENCHMARKS),
        default=None,
        help="Benchmark to evaluate (default: inferred from config's 'benchmark' field; "
        "fallback AndroidControl for backward compatibility)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Benchmark JSON config (default: per-benchmark default config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation output directory "
            "(default: outputs/visionzip_{benchmark_lower}_v0_eval)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run single-GPU smoke test via scripts/visionzip/dryrun_visionzip.py",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU index for --dry-run (default: 0)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Base model directory for --dry-run",
    )
    parser.add_argument(
        "--dominant-ratio",
        type=float,
        default=None,
        help="Override VisionZip dominant_ratio (dry-run only)",
    )
    parser.add_argument(
        "--contextual-ratio",
        type=float,
        default=None,
        help="Override VisionZip contextual_ratio (dry-run only)",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Optional split override (e.g. validation, test)",
    )
    parser.add_argument(
        "--measure-e2e-latency",
        action="store_true",
        help="Measure per-step end-to-end latency during evaluation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing output_dir checkpoints instead of starting over.",
    )
    return parser.parse_args()


def _load_config_summary(config_path: Path) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    service = payload.get("service") or {}
    extra = service.get("extra") or {}
    evaluation = payload.get("evaluation") or {}
    model = payload.get("model") or {}
    paths = payload.get("paths") or {}
    return {
        "benchmark": payload.get("benchmark"),
        "backend": model.get("backend"),
        "model_name": model.get("model_name"),
        "base_model_path": paths.get("base_model_path"),
        "worker_gpus": evaluation.get("worker_gpus", []),
        "dominant_ratio": extra.get("dominant_ratio"),
        "contextual_ratio": extra.get("contextual_ratio"),
    }


def _run_dryrun(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/visionzip/dryrun_visionzip.py"),
        "--gpu",
        str(args.gpu),
        "--model-path",
        str(args.model_path),
    ]
    if args.dominant_ratio is not None:
        cmd.extend(["--dominant-ratio", str(args.dominant_ratio)])
    if args.contextual_ratio is not None:
        cmd.extend(["--contextual-ratio", str(args.contextual_ratio)])
    print("Launching dry-run:")
    print(" ", " ".join(cmd))
    return subprocess.call(cmd)


def _run_evaluation_for_benchmark(args: argparse.Namespace, benchmark: str) -> int:
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/core/run_evaluation.py"),
        "--benchmark",
        benchmark,
        "--config",
        str(config_path),
        "--variant",
        "V0",
        "--output-dir",
        str(output_dir),
    ]
    if args.split:
        cmd.extend(["--split", args.split])
    if args.measure_e2e_latency:
        cmd.append("--measure-e2e-latency")
    if args.resume:
        cmd.append("--resume")

    print(f"Launching VisionZip V0 evaluation ({benchmark}):")
    print(" ", " ".join(cmd))
    return subprocess.call(cmd)


def _run_evaluation(args: argparse.Namespace) -> int:
    """Run evaluation, inferring benchmark from args.benchmark or config if not set."""
    benchmark = getattr(args, "benchmark", None)
    if not benchmark:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        benchmark = cfg.get("benchmark") or "AndroidControl"
    return _run_evaluation_for_benchmark(args, benchmark)


def _run_learngui_evaluation(args: argparse.Namespace) -> int:
    """Run evaluation targeting the LearnGUI benchmark."""
    return _run_evaluation_for_benchmark(args, "LearnGUI")


def main() -> int:
    args = parse_args()

    # Resolve config: explicit --config takes priority; else use per-benchmark default.
    # If neither --benchmark nor --config is given, fall back to AndroidControl for
    # backward compatibility and read benchmark from the config file below.
    if args.config is None:
        if args.benchmark is not None:
            args.config = DEFAULT_CONFIGS[args.benchmark]
        else:
            args.config = DEFAULT_CONFIGS["AndroidControl"]

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    summary = _load_config_summary(config_path)

    # Resolve benchmark: CLI flag > config field > "AndroidControl" fallback
    config_benchmark = summary.get("benchmark")
    if args.benchmark is None:
        args.benchmark = config_benchmark or "AndroidControl"
    elif config_benchmark and config_benchmark != args.benchmark:
        print(
            f"WARNING: --benchmark={args.benchmark!r} but config has "
            f"benchmark={config_benchmark!r}",
            file=sys.stderr,
        )

    benchmark = args.benchmark

    # Resolve output_dir
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"outputs/visionzip_{benchmark.lower()}_v0_eval"

    print("VisionZip run configuration:")
    print(json.dumps(summary, indent=2))
    print(f"Resolved benchmark: {benchmark}")

    if summary.get("backend") != "visionzip":
        print(
            f"WARNING: config model.backend={summary.get('backend')!r}, expected 'visionzip'",
            file=sys.stderr,
        )

    if args.dry_run:
        return _run_dryrun(args)
    return _run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
