"""Merge partial witness outputs from distributed discovery nodes and run postprocess.

Used after shard-parallel discovery (run_discovery.py --skip-postprocess) where
each node processed a subset of shards and saved witness chunks independently.

Usage:
  python scripts/core/run_discovery_postprocess.py \\
      --benchmark AndroidControl \\
      --config configs/androidcontrol_64gpu.json \\
      --partial-output-dirs \\
          outputs/disc_partial_node0 \\
          outputs/disc_partial_node1 \\
          outputs/disc_partial_node2 \\
          outputs/disc_partial_node3 \\
      --output-dir outputs/androidcontrol_discovery_v3 \\
      --model-path outputs/sft_v3_merged
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

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

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_progress(partial_dir: Path) -> dict:
    """Load serial_discovery_progress.json from a partial output directory."""
    progress_file = partial_dir / "serial_discovery_progress.json"
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge partial discovery outputs and run skill induction + calibration."
    )
    parser.add_argument("--benchmark", required=True, choices=("AndroidControl", "LearnGUI"))
    parser.add_argument("--config", help="JSON config path (uses default if omitted).")
    parser.add_argument(
        "--partial-output-dirs", nargs="+", required=True, metavar="DIR",
        help="Partial discovery output directories from each node (witness chunks must be present).",
    )
    parser.add_argument("--output-dir", required=True, help="Final output directory for skill library.")
    parser.add_argument("--model-path", help="Model source override.")
    parser.add_argument("--model-name", help="Model name override.")
    parser.add_argument("--served-model-name", help="Served model name override.")
    parser.add_argument("--api-base", help="API base override (default: http://127.0.0.1:8000/v1).")
    parser.add_argument("--resume", action="store_true", help="Resume postprocess from existing output.")
    args = parser.parse_args()

    from skillreuse.configuration import (
        apply_runtime_overrides,
        load_benchmark_config,
        resolve_backend_config,
        resolve_model_spec,
    )
    from skillreuse.discovery import run_discovery_postprocess_from_witnesses, save_discovery_run
    from skillreuse.model import build_backend

    config = apply_runtime_overrides(
        load_benchmark_config(benchmark=args.benchmark, config_path=args.config),
        model_path=args.model_path,
        model_name=args.model_name,
        served_model_name=args.served_model_name,
        api_base=args.api_base,
    )
    model_spec = resolve_model_spec(config, benchmark=args.benchmark)
    backend_config = resolve_backend_config(config)
    backend = build_backend(model_spec, config=backend_config, eager_load=True)

    # ── Load and merge witness chunks from all partial output directories ──────
    _log(f"loading witnesses from {len(args.partial_output_dirs)} partial directories...")
    all_witnesses: list = []
    total_discovery_examples = 0
    total_witness_count = 0

    for i, partial_dir_str in enumerate(args.partial_output_dirs):
        partial_dir = Path(partial_dir_str)
        if not partial_dir.exists():
            raise FileNotFoundError(f"Partial output directory not found: {partial_dir}")

        # Load progress metadata for discovery_examples and witness_count
        progress = _load_progress(partial_dir)
        node_discovery_examples = int(progress.get("discovery_examples") or 0)
        node_witness_count = int(progress.get("witness_count") or 0)
        total_discovery_examples += node_discovery_examples
        total_witness_count += node_witness_count

        # Witness chunks live in the serial_discovery_witnesses/ subdirectory
        # with names like train_retained_witness_chunk*.pkl.
        # The progress JSON has the authoritative list (use it when available).
        progress_chunk_paths = (
            progress.get("witness_chunk_paths")
            or progress.get("chunk_paths")
            or []
        )

        if progress_chunk_paths:
            chunk_files = [Path(p) for p in sorted(progress_chunk_paths)]
        else:
            # Fallback search in common locations
            chunk_files = sorted(
                list(partial_dir.glob("serial_discovery_witnesses/train_retained_witness_chunk*.pkl"))
                or list(partial_dir.glob("serial_discovery_witnesses/*.pkl"))
                or list(partial_dir.glob("witness_chunk*.pkl"))
                or list(partial_dir.glob("**/*witness_chunk*.pkl"))
            )

        node_witness_count_from_chunks = 0
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as fh:
                chunk_witnesses = pickle.load(fh)
            all_witnesses.extend(chunk_witnesses)
            node_witness_count_from_chunks += len(chunk_witnesses)

        _log(
            f"  node {i} ({partial_dir.name}): "
            f"{node_discovery_examples} examples, "
            f"{node_witness_count_from_chunks} retained witnesses "
            f"({len(chunk_files)} chunks)"
        )

    _log(
        f"merged: {total_discovery_examples} total train examples, "
        f"{len(all_witnesses)} retained witnesses, "
        f"{total_witness_count} total attempted"
    )

    # ── Run induction + calibration ───────────────────────────────────────────
    _log("running postprocess (induction + calibration)...")
    result = run_discovery_postprocess_from_witnesses(
        args.benchmark,
        config=config,
        retained_witnesses=all_witnesses,
        discovery_examples_count=total_discovery_examples,
        witness_count=total_witness_count,
        fallback_model=backend,
        model_spec=model_spec,
        backend_config=backend_config,
        output_dir=args.output_dir,
    )

    if result.skill_library is not None:
        result = save_discovery_run(result, output_dir=args.output_dir)
        _log(f"skill library saved: {result.skill_library_path}")
        _log(f"  skills: {result.skill_count}  calibrated: {result.calibrated_skill_count}")
    else:
        _log("WARNING: postprocess returned no skill library")

    print(
        json.dumps({
            "benchmark": result.benchmark,
            "discovery_examples": result.discovery_examples,
            "witness_count": result.witness_count,
            "skill_count": result.skill_count,
            "calibrated_skill_count": result.calibrated_skill_count,
            "skill_library_path": result.skill_library_path,
        }, indent=2)
    )


if __name__ == "__main__":
    main()
