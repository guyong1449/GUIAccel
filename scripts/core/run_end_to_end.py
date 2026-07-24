"""Convenience wrapper: run discovery then V1 evaluation."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skillreuse.calibration import CalibrationRunResult
from skillreuse.configuration import (
    apply_runtime_overrides,
    load_benchmark_config,
    resolve_backend_config,
    resolve_discovery_config,
    resolve_evaluation_config,
    resolve_model_spec,
)
from skillreuse.discovery import run_discovery_pipeline
from skillreuse.evaluation import run_benchmark_evaluation
from skillreuse.model import build_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discovery and then V1 evaluation.")
    parser.add_argument("--benchmark", required=True, choices=("LearnGUI", "AndroidControl"))
    parser.add_argument("--config", help="Optional JSON config path.")
    parser.add_argument("--discovery-output-dir", required=True)
    parser.add_argument("--evaluation-output-dir", required=True)
    parser.add_argument("--model-path", help="Optional model path override.")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter override.")
    parser.add_argument("--api-base", help="Optional vLLM API base override.")
    args = parser.parse_args()

    config = load_benchmark_config(benchmark=args.benchmark, config_path=args.config)
    config = apply_runtime_overrides(
        config,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        api_base=args.api_base,
    )

    model_spec = resolve_model_spec(config, benchmark=args.benchmark)
    backend_config = resolve_backend_config(config)
    discovery_config = resolve_discovery_config(config)
    evaluation_config = resolve_evaluation_config(config)

    discovery_backend = None
    if not discovery_config["worker_gpus"]:
        discovery_backend = build_backend(model_spec, config=backend_config, eager_load=True)

    discovery = run_discovery_pipeline(
        args.benchmark,
        config=config,
        fallback_model=discovery_backend,
        model_spec=model_spec,
        backend_config=backend_config,
        output_dir=args.discovery_output_dir,
    )

    payload = discovery.skill_library
    if not isinstance(payload, CalibrationRunResult):
        raise TypeError("Discovery did not return a CalibrationRunResult.")

    evaluation_backend = discovery_backend
    if not evaluation_config.worker_gpus and evaluation_backend is None:
        evaluation_backend = build_backend(model_spec, config=backend_config, eager_load=True)

    report = run_benchmark_evaluation(
        args.benchmark,
        fallback_model=evaluation_backend,
        skill_library=payload.accepted_skills,
        variant_id="V1",
        config=evaluation_config,
        model_spec=model_spec,
        backend_config=backend_config,
        output_dir=args.evaluation_output_dir,
    )

    print(
        json.dumps(
            {
                "skill_library_path": discovery.skill_library_path,
                "evaluation_output_path": report.get("output_path"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
