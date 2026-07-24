"""CLI entrypoint for the new quotient-skill discovery pipeline."""

from __future__ import annotations

import argparse
import json
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

from skillreuse.configuration import (
    apply_runtime_overrides,
    load_benchmark_config,
    resolve_backend_config,
    resolve_discovery_config,
    resolve_model_spec,
)
from skillreuse.discovery import run_discovery_pipeline
from skillreuse.model import build_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quotient-skill discovery and calibration.")
    parser.add_argument("--benchmark", required=True, choices=("LearnGUI", "AndroidControl"))
    parser.add_argument("--config", help="Optional JSON config path.")
    parser.add_argument("--output-dir", required=True, help="Directory for the calibrated quotient-skill library.")
    parser.add_argument("--model-path", help="Optional model path override (local transformers or vLLM model dir).")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter override.")
    parser.add_argument("--api-base", help="Optional vLLM service API base override, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--resume", action="store_true", help="Resume discovery from an interrupted output directory.")
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
    # Ensure output directory exists before run_discovery_pipeline tries to write
    # progress files (discovery_postprocess_progress.json) directly inside it.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    backend = None
    if not discovery_config["worker_gpus"]:
        backend = build_backend(model_spec, config=backend_config, eager_load=True)
    result = run_discovery_pipeline(
        args.benchmark,
        config=config,
        fallback_model=backend,
        model_spec=model_spec,
        backend_config=backend_config,
        output_dir=args.output_dir,
        resume_from_output_dir=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "skill_library_path": result.skill_library_path,
                "summary_path": result.summary_path,
                "calibrated_skill_count": result.calibrated_skill_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
