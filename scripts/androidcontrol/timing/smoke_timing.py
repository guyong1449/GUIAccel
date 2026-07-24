#!/usr/bin/env python3
"""Smoke test for timing hooks: run N real AndroidControl examples and verify timing.

Timing eval configs must use attn_implementation with vision_config/text_config only:
  {"vision_config": "eager", "text_config": "sdpa"}
  {"vision_config": "sdpa", "text_config": "sdpa"}
flash_attention_2 and legacy visual/model keys are not allowed.

Usage:
  python scripts/androidcontrol/timing/smoke_timing.py --gpu 0 --model-path models/Qwen3-VL-8B-Instruct --samples 10
  python scripts/androidcontrol/timing/smoke_timing.py --gpu 0 --model-path models/Qwen3-VL-8B-Instruct \\
      --samples 10 --visionzip --output outputs/smoke_timing.json

Exits 0 if all samples have populated timing fields, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from skillreuse.utils.env import load_repo_env  # noqa: E402

load_repo_env(REPO_ROOT)

DEFAULT_MODEL_PATH = os.environ.get(
    "SKILLREUSE_BASE_MODEL_PATH",
    str(REPO_ROOT / "models/Qwen3-VL-8B-Instruct"),
)

TIMING_PHASES = [
    "image_prep_ms",
    "message_build_ms",
    "tokenization_ms",
    "gpu_transfer_ms",
    "model_generate_ms",
    "prefill_ms",
    "decode_ms",
    "decode_steps",
    "vision_encoder_ms",
    "attn_scoring_prune_ms",
    "llm_forward_ms",
    "output_decode_ms",
    "output_parse_ms",
    "total_e2e_step_ms",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: verify timing across N real AndroidControl examples.")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index (default: 0)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of examples to run (default: 10)")
    parser.add_argument("--visionzip", action="store_true", help="Use VisionZipQwenBackend")
    parser.add_argument("--dominant-ratio", type=float, default=0.65)
    parser.add_argument("--contextual-ratio", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="",
        help="Path to write JSON output with per-step timing + summary (optional)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(variance)


def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Backend              = {'visionzip' if args.visionzip else 'baseline (local_transformers)'}")
    print(f"Model path           = {args.model_path}")
    print(f"Samples              = {args.samples}")

    # ---- Imports ----
    try:
        import torch
        from skillreuse.model.qwen_backend import QwenBackendConfig
        from skillreuse.routing.fallback import (
            FallbackModelConfig,
            QwenLoRAModelSpec,
            build_fallback_request,
        )
        from skillreuse.routing.execution import context_from_repository_example
        from skillreuse.evaluation.orchestrator import build_android_examples
        from skillreuse.evaluation.timing import compute_timing_summary
    except ImportError as exc:
        print(f"\nERROR: Import failed — {exc}")
        print("Make sure the SkillReuse conda/venv environment is activated.")
        return 1

    # -----------------------------------------------------------------------
    # Build backend
    # -----------------------------------------------------------------------
    section("Build Backend")

    attn_impl: dict[str, str] | str = {"visual": "eager", "model": "flash_attention_2"}
    backend_config = QwenBackendConfig(device_map=0, attn_implementation=attn_impl)

    if args.visionzip:
        try:
            from skillreuse.model.visionzip_backend import VisionZipConfig, VisionZipQwenBackend
        except ImportError as exc:
            print(f"\nERROR: VisionZip import failed — {exc}")
            return 1
        vz_config = VisionZipConfig(
            dominant_ratio=args.dominant_ratio,
            contextual_ratio=args.contextual_ratio,
        )
        model_spec = QwenLoRAModelSpec(
            benchmark="AndroidControl",
            backend="visionzip",
            base_model_path=args.model_path,
            lora_adapter_path=None,
            model_name="Qwen3-VL-8B-VisionZip",
        )
        backend = VisionZipQwenBackend(
            model_spec,
            config=backend_config,
            visionzip_config=vz_config,
            eager_load=False,
        )
    else:
        from skillreuse.model.qwen_backend import QwenLoRABackend
        model_spec = QwenLoRAModelSpec(
            benchmark="AndroidControl",
            backend="local",
            base_model_path=args.model_path,
            lora_adapter_path=None,
            model_name="Qwen3-VL-8B-Instruct",
        )
        backend = QwenLoRABackend(model_spec, config=backend_config, eager_load=False)

    # -----------------------------------------------------------------------
    # Load examples
    # -----------------------------------------------------------------------
    section(f"Load AndroidControl Examples (first {args.samples} from test split)")

    print(f"  Loading examples from test split …")
    try:
        all_examples = build_android_examples(
            split="test",
            instruction_modes=["high_level", "low_level"],
            episode_limit=None,
        )
    except Exception as exc:
        print(f"  ERROR loading examples: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    examples = list(all_examples[: args.samples])
    print(f"  Loaded {len(examples)} examples (requested {args.samples}).")
    if not examples:
        print("  ERROR: No examples loaded.")
        return 1

    # -----------------------------------------------------------------------
    # Warmup (1 untimed call to pre-compile CUDA kernels)
    # -----------------------------------------------------------------------
    section("Warmup (1 untimed step)")

    fallback_config = FallbackModelConfig()
    warmup_example = examples[0]
    try:
        warmup_request = build_fallback_request(
            context_from_repository_example(warmup_example.repository_example),
            reason="warmup",
            config=fallback_config,
            model_spec=model_spec,
        )
        with torch.no_grad():
            _ = backend.generate_batch([warmup_request])
        print("  Warmup complete.")
    except Exception as exc:
        print(f"  WARNING: warmup failed ({exc}); continuing anyway.")

    # -----------------------------------------------------------------------
    # Timed inference loop
    # -----------------------------------------------------------------------
    section(f"Timed Inference ({len(examples)} examples)")

    per_step_timing: list[dict[str, float]] = []
    errors = 0
    for i, eval_example in enumerate(examples, start=1):
        try:
            request = build_fallback_request(
                context_from_repository_example(eval_example.repository_example),
                reason="smoke_timing",
                config=fallback_config,
                model_spec=model_spec,
            )
            with torch.no_grad():
                responses = backend.generate_batch([request])
            response = responses[0]
            timing = response.timing or {}
            per_step_timing.append(timing)
            populated = len([v for v in timing.values() if v > 0.0])
            print(f"  [{i:3d}/{len(examples)}] total_e2e_step_ms={timing.get('total_e2e_step_ms', 0.0):.1f}  "
                  f"model_generate_ms={timing.get('model_generate_ms', 0.0):.1f}  "
                  f"populated_fields={populated}")
        except Exception as exc:
            print(f"  [{i:3d}/{len(examples)}] ERROR: {exc}")
            errors += 1
            per_step_timing.append({})

    # -----------------------------------------------------------------------
    # Compute and print summary
    # -----------------------------------------------------------------------
    section("Timing Summary (mean ± std)")

    # Build pseudo StepEvaluationResult objects that compute_timing_summary expects
    class _FakeResult:
        def __init__(self, t: dict[str, float]) -> None:
            self.baseline_timing = t if t else None

    fake_results = [_FakeResult(t) for t in per_step_timing]
    summary = compute_timing_summary(fake_results)  # type: ignore[arg-type]

    print(f"\n  {'Phase':<32}  {'Mean (ms)':>10}  {'Std (ms)':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for phase in TIMING_PHASES:
        vals = [t.get(phase, 0.0) for t in per_step_timing if t]
        if not vals:
            continue
        m, s = mean_std(vals)
        mn = min(vals)
        mx = max(vals)
        indent = "    " if phase in ("vision_encoder_ms", "attn_scoring_prune_ms", "llm_forward_ms", "prefill_ms", "decode_ms", "decode_steps") else ""
        print(f"  {indent}{phase:<28}  {m:>10.2f}  {s:>10.2f}  {mn:>8.2f}  {mx:>8.2f}")

    # -----------------------------------------------------------------------
    # Validation: all samples must have populated timing
    # -----------------------------------------------------------------------
    section("Validation")

    samples_with_timing = sum(1 for t in per_step_timing if t and t.get("total_e2e_step_ms", 0.0) > 0.0)
    all_populated = samples_with_timing == len(examples)
    status = "PASS" if all_populated else "FAIL"
    print(f"  [{status}] {samples_with_timing}/{len(examples)} samples have populated timing")
    if errors > 0:
        print(f"  [FAIL] {errors} samples raised exceptions")

    # -----------------------------------------------------------------------
    # Write JSON output
    # -----------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "visionzip" if args.visionzip else "baseline",
            "model_path": str(args.model_path),
            "samples": len(examples),
            "per_step_timing": per_step_timing,
            "timing_summary": summary,
            "validation": {
                "samples_with_timing": samples_with_timing,
                "all_populated": all_populated,
                "errors": errors,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\n  Output written to: {output_path}")

    section("Result")
    if all_populated and errors == 0:
        print("  ALL SMOKE CHECKS PASSED.\n")
        return 0
    else:
        print("  SMOKE CHECKS FAILED — review output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
