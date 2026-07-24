#!/usr/bin/env python3
"""Convert AndroidControl training split to Swift sharegpt JSONL format (Qwen3-VL version).

Parallel processing: one worker per TFRecord shard.
With 20 shards and --workers 20, wall-clock time drops from ~2h to ~5-10min.

Output format (one JSON per line):
  {
    "messages": [
      {"role": "system", "content": "<qwen3vl_system_prompt>"},
      {"role": "user",   "content": "<image><inference-matching-prompt>"},
      {"role": "assistant", "content": "{\"action_type\": ..., \"bbox\": [...], ...}"}
    ],
    "images": ["/abs/path/to/screenshot.png"]
  }

Key differences from SkillReuse (MAI-UI) version:
  - Response format: plain JSON (not <thinking>...<tool_call>)
  - Coordinates: absolute pixel coords (not 0-999 scaled)
  - System prompt: Qwen3-VL JSON-output prompt
  - Both high_level and low_level modes supported

Usage:
  python scripts/convert_androidcontrol_to_swift.py \\
      --config configs/androidcontrol/default.json \\
      --output-dir outputs/swift_data \\
      [--workers 20] \\
      [--instruction-modes high_level low_level] \\
      [--limit 5]    # per-shard episode limit for smoke test
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

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

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ── System prompt must match VLLMOpenAIBackend._build_full_messages() exactly ──
QWEN3VL_SYSTEM_PROMPT = (
    "You are a benchmark-faithful GUI action model. "
    "Return exactly one JSON object with keys: action_type, argument, bbox, app, direction. "
    "Use null for fields not needed by the action. "
    "bbox=[left,top,right,bottom] must use ABSOLUTE pixel coordinates from the screenshot "
    "(x: 0=left edge, y: 0=top edge of screen). Do NOT use normalized or 0-999 scale. "
    "action_type values: "
    "CLICK (bbox=target element); "
    "LONG_PRESS (bbox=target element); "
    "TYPE (argument=text string); "
    "SCROLL (direction=up/down/left/right); "
    "NAV (argument=back/home/enter or app name); "
    "TERMINATE; "
    "WAIT."
)


# ---------------------------------------------------------------------------
# Helpers (pure functions, safe for multiprocessing)
# ---------------------------------------------------------------------------

def _log(msg: str, worker: int | None = None) -> None:
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    if worker is not None:
        prefix += f" [shard {worker}]"
    print(f"{prefix} {msg}", flush=True)


def _action_to_response_json(action: Any) -> str:
    """Convert CanonicalAction to Qwen3-VL JSON response.

    Uses absolute pixel coordinates (Qwen3-VL native — no 0-999 scaling).
    Matches the format parsed by _parse_action_output() in service_backend.py.
    """
    return json.dumps(
        {
            "action_type": action.action_type,
            "argument": action.argument,
            "bbox": list(action.bbox) if action.bbox is not None else None,
            "app": action.app,
            "direction": action.direction,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _save_screenshot(screenshot: Any, out_path: Path) -> bool:
    try:
        from PIL import Image as PILImage
        if screenshot.path is not None and Path(screenshot.path).exists():
            img = PILImage.open(screenshot.path).convert("RGB")
        elif screenshot.read_bytes():
            img = PILImage.open(io.BytesIO(screenshot.read_bytes())).convert("RGB")
        else:
            return False
        img.save(str(out_path), format="PNG", optimize=False)
        return True
    except Exception as exc:
        _log(f"WARNING: failed to save screenshot {out_path.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Worker: process one shard
# ---------------------------------------------------------------------------

def _process_shard(args_tuple: tuple) -> dict:
    """Worker: process a single TFRecord shard → partial JSONL."""
    (
        shard_idx,
        shard_path_str,
        partial_jsonl_path_str,
        images_dir_str,
        manifest_path_str,
        instruction_modes,
        episode_limit,
    ) = args_tuple

    images_dir = Path(images_dir_str)

    from skillreuse.data import AndroidControlDataset, canonicalize_step
    from skillreuse.data.android_control import AndroidControlDataset as _ACD
    from skillreuse.routing.execution import StepContext
    from skillreuse.routing.fallback import build_full_prompt

    # Try to import project_android_runtime_record for HL mode (clear step_instruction)
    try:
        from skillreuse.android_runtime import project_android_runtime_record
        _HAS_PROJECT = True
    except ImportError:
        _HAS_PROJECT = False

    dataset = _ACD(manifest_path=manifest_path_str)

    t0 = time.time()
    episode_count = 0
    example_count = 0
    skip_count = 0

    shard_path = Path(shard_path_str)

    # iter_episodes with shard_paths restricts to this shard for parallel processing
    iter_kwargs: dict = {"split": "train"}
    try:
        # Try the shard_paths + limit interface (SkillReuse-style)
        episodes_iter = dataset.iter_episodes(
            split="train", shard_paths=[shard_path], limit=episode_limit
        )
    except TypeError:
        # Fallback: no shard_paths support — iterate all episodes (less parallel)
        episodes_iter = dataset.iter_episodes(split="train")

    with open(partial_jsonl_path_str, "w", encoding="utf-8") as fout:
        for episode in episodes_iter:
            if episode_limit is not None and episode_count >= episode_limit:
                break
            episode_count += 1
            canonical_records = tuple(canonicalize_step(step) for step in episode.steps)

            for step_index, step in enumerate(episode.steps):
                # Save screenshot once per step (shared across modes)
                img_name = f"{episode.episode_id}_{step_index:04d}.png"
                img_path = images_dir / img_name
                if not img_path.exists():
                    if not _save_screenshot(step.screenshot, img_path):
                        skip_count += len(instruction_modes)
                        continue

                base_record = canonical_records[step_index]
                base_history = tuple(canonical_records[:step_index])

                for mode in instruction_modes:
                    # Skip LL steps that have no step_instruction
                    if mode == "low_level" and not base_record.normalized_metadata.get("step_instruction"):
                        skip_count += 1
                        continue

                    # For HL mode, clear step_instruction via project_android_runtime_record
                    # to match evaluation behavior (high_level inference doesn't see step_instruction).
                    if mode == "high_level" and _HAS_PROJECT:
                        record = project_android_runtime_record(base_record, mode="high_level")
                        # project_android_runtime_record sets canonical_action=UNOBSERVED;
                        # always use the original base_record action as the label.
                        action_for_response = base_record.canonical_action
                        history = tuple(
                            project_android_runtime_record(r, mode="high_level", preserve_action=True)
                            for r in base_history
                        )
                    else:
                        record = base_record
                        action_for_response = base_record.canonical_action
                        history = base_history

                    if action_for_response is None:
                        skip_count += 1
                        continue

                    obs_id = f"androidcontrol:train:{episode.episode_id}:{step_index}:{mode}"
                    ctx = StepContext(
                        observation_id=obs_id,
                        record=record,
                        history=history,
                        support_context={"instruction_mode": mode},
                    )
                    prompt_text = build_full_prompt(ctx)

                    # User content format must match VLLMOpenAIBackend._build_full_messages():
                    #   "Benchmark: {benchmark}\nReason: {reason}\n{prompt_text}\n\n{response_instruction}"
                    user_content = (
                        "Benchmark: AndroidControl\n"
                        "Reason: baseline_eval\n"
                        f"{prompt_text}"
                        "\n\nReturn exactly one JSON object."
                    )

                    record_out = {
                        "messages": [
                            {"role": "system", "content": QWEN3VL_SYSTEM_PROMPT},
                            # <image> token tells Swift where to insert image tokens
                            {"role": "user", "content": f"<image>{user_content}"},
                            {"role": "assistant", "content": _action_to_response_json(action_for_response)},
                        ],
                        "images": [str(img_path)],
                    }
                    fout.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                    example_count += 1

    elapsed = time.time() - t0
    _log(
        f"done: episodes={episode_count} examples={example_count} "
        f"skipped={skip_count} elapsed={elapsed:.0f}s",
        worker=shard_idx,
    )
    return {
        "shard_idx": shard_idx,
        "partial_jsonl": partial_jsonl_path_str,
        "episode_count": episode_count,
        "example_count": example_count,
        "skip_count": skip_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AndroidControl train split → Swift JSONL (Qwen3-VL, parallel)"
    )
    parser.add_argument("--config", default="configs/androidcontrol/default.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--workers", type=int, default=20,
        help="Parallel workers (one per shard). Default: 20.",
    )
    parser.add_argument(
        "--instruction-modes", nargs="+",
        default=["high_level", "low_level"],
        choices=["high_level", "low_level"],
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Per-shard episode limit for smoke testing (e.g. --limit 5).",
    )
    args = parser.parse_args()

    from skillreuse.configuration import load_benchmark_config, resolve_dataset_manifest_path
    from skillreuse.data import AndroidControlDataset

    config = load_benchmark_config(benchmark="AndroidControl", config_path=args.config)
    manifest = resolve_dataset_manifest_path(config)
    if not manifest or not Path(manifest).exists():
        print(f"ERROR: manifest not found: {manifest!r}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    partial_dir = output_dir / "_partial"
    images_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    final_jsonl = output_dir / "androidcontrol_train.jsonl"

    dataset = AndroidControlDataset(manifest_path=manifest)
    shard_paths = dataset.shard_paths
    num_workers = min(args.workers, len(shard_paths))

    _log(f"shards={len(shard_paths)}  workers={num_workers}  modes={args.instruction_modes}")
    _log(f"limit_per_shard={args.limit}  output={final_jsonl}")

    worker_args = [
        (
            i,
            str(shard_path),
            str(partial_dir / f"shard_{i:02d}.jsonl"),
            str(images_dir),
            str(manifest),
            args.instruction_modes,
            args.limit,
        )
        for i, shard_path in enumerate(shard_paths)
    ]

    t0 = time.time()
    _log(f"starting {num_workers} parallel workers …")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_process_shard, worker_args)

    _log("merging partial JSONL files …")
    total_episodes = sum(r["episode_count"] for r in results)
    total_examples = sum(r["example_count"] for r in results)
    total_skipped = sum(r["skip_count"] for r in results)

    with open(final_jsonl, "w", encoding="utf-8") as fout:
        for r in sorted(results, key=lambda x: x["shard_idx"]):
            p = Path(r["partial_jsonl"])
            if p.exists():
                fout.write(p.read_text(encoding="utf-8"))
                p.unlink()

    try:
        partial_dir.rmdir()
    except OSError:
        pass

    elapsed = time.time() - t0
    _log(f"ALL DONE  episodes={total_episodes}  examples={total_examples}  "
         f"skipped={total_skipped}  elapsed={elapsed:.0f}s")
    _log(f"JSONL: {final_jsonl}  ({total_examples} lines)")

    info = {
        "benchmark": "AndroidControl",
        "split": "train",
        "instruction_modes": args.instruction_modes,
        "episode_count": total_episodes,
        "example_count": total_examples,
        "skip_count": total_skipped,
        "shard_count": len(shard_paths),
        "workers_used": num_workers,
        "limit_per_shard": args.limit,
        "elapsed_seconds": round(elapsed),
        "jsonl_path": str(final_jsonl),
        "images_dir": str(images_dir),
        "format": "Qwen3-VL: plain JSON response, absolute pixel coords, no 0-999 scaling",
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    _log(f"info: {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
