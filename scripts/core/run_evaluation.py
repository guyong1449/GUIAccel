"""CLI entrypoint for V0-V3 evaluation under the experiment plan."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from guiaccel.calibration import CalibrationRunResult
    from guiaccel.configuration import (
        apply_runtime_overrides,
        load_benchmark_config,
        resolve_backend_config,
        resolve_evaluation_config,
        resolve_model_spec,
    )
    from guiaccel.evaluation import run_benchmark_evaluation
    from guiaccel.evaluation.log_summary import format_evaluation_summary
    from guiaccel.journal import persist_evaluation_summary_artifacts
    from guiaccel.model import build_backend

    parser = argparse.ArgumentParser(description="Run V0-V3 evaluation for the quotient controller.")
    parser.add_argument("--benchmark", required=True, choices=("LearnGUI", "AndroidControl"))
    parser.add_argument("--config", help="Optional JSON config path.")
    parser.add_argument(
        "--skill-library-path",
        default="",
        help="Pickled CalibrationRunResult from discovery. "
             "Required for V1/V2/V3. Optional for V0 (no routing — skill library is not used).",
    )
    parser.add_argument("--variant", default="V1", choices=("V0", "V1", "V2", "V3"))
    parser.add_argument("--split", help="Optional benchmark split override, e.g. validation or test.")
    parser.add_argument("--output-dir", required=True, help="Directory for the evaluation JSON output.")
    parser.add_argument("--model-path", help="Optional model path override.")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter override.")
    parser.add_argument("--api-base", help="Optional vLLM API base override.")
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--measure-e2e-latency",
        action="store_true",
        help="Use microbatch size 1 for per-step timing measurement.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing output_dir checkpoints instead of starting over.",
    )
    args = parser.parse_args()

    # V0 (baseline only) does not use the skill library at all; other variants require it.
    needs_skill_library = args.variant != "V0"
    if needs_skill_library and not args.skill_library_path:
        parser.error(f"--skill-library-path is required for variant {args.variant}")

    config = load_benchmark_config(benchmark=args.benchmark, config_path=args.config)
    config = apply_runtime_overrides(
        config,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        api_base=args.api_base,
    )
    if args.split is not None:
        evaluation = dict(config.get("evaluation") or {})
        if args.benchmark == "AndroidControl":
            evaluation["android_split"] = str(args.split)
        else:
            evaluation["learn_gui_split"] = str(args.split)
        config["evaluation"] = evaluation

    model_spec = resolve_model_spec(config, benchmark=args.benchmark)
    backend_config = resolve_backend_config(config)
    evaluation_config = resolve_evaluation_config(config)
    if args.task_limit is not None:
        evaluation_config = replace(evaluation_config, task_limit=args.task_limit)
    if args.measure_e2e_latency:
        evaluation_config = replace(
            evaluation_config,
            measure_end_to_end_latency=True,
            microbatch=replace(
                evaluation_config.microbatch,
                sample_cap_per_gpu=1,
            ),
        )

    backend = None
    skill_library: tuple = tuple()
    skill_library_path = str(Path(args.skill_library_path).resolve()) if args.skill_library_path else ""

    if not evaluation_config.worker_gpus:
        backend = build_backend(model_spec, config=backend_config, eager_load=True)
        if needs_skill_library and skill_library_path:
            payload = pickle.loads(Path(skill_library_path).read_bytes())
            if not isinstance(payload, CalibrationRunResult):
                raise TypeError(f"{args.skill_library_path} does not contain a CalibrationRunResult.")
            skill_library = payload.accepted_skills

    report = run_benchmark_evaluation(
        args.benchmark,
        fallback_model=backend,
        skill_library=skill_library,
        variant_id=args.variant,
        config=evaluation_config,
        model_spec=model_spec,
        backend_config=backend_config,
        output_dir=args.output_dir,
        skill_library_path=skill_library_path or None,
        resume_from_output_dir=bool(args.resume),
    )
    log_summary = dict(report.get("log_summary") or {})
    if log_summary:
        print(format_evaluation_summary(log_summary), flush=True)
        journal_run_dir = os.environ.get("DIVPRUNE_JOURNAL_RUN_DIR") or os.environ.get(
            "SKILLREUSE_JOURNAL_RUN_DIR"
        )
        if journal_run_dir:
            persist_evaluation_summary_artifacts(Path(journal_run_dir), report)
    print(json.dumps({"output_path": report.get("output_path")}, indent=2))


if __name__ == "__main__":
    main()
