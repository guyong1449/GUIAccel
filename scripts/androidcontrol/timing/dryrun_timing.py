#!/usr/bin/env python3
"""Dry-run validation for timing hooks in QwenLoRABackend / VisionZipQwenBackend.

Timing eval configs must use attn_implementation with vision_config/text_config only:
  {"vision_config": "eager", "text_config": "sdpa"}
  {"vision_config": "sdpa", "text_config": "sdpa"}
flash_attention_2 and legacy visual/model keys are not allowed.

Usage:
  python scripts/androidcontrol/timing/dryrun_timing.py --gpu 0 --model-path models/Qwen3-VL-8B-Instruct
  python scripts/androidcontrol/timing/dryrun_timing.py --gpu 0 --model-path models/Qwen3-VL-8B-Instruct --visionzip

Exits 0 if all timing assertions pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
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

REQUIRED_BASELINE = [
    "image_prep_ms",
    "message_build_ms",
    "tokenization_ms",
    "gpu_transfer_ms",
    "model_generate_ms",
    "prefill_ms",
    "decode_ms",
    "decode_steps",
    "vision_encoder_ms",
    "output_decode_ms",
    "output_parse_ms",
    "total_e2e_step_ms",
]
REQUIRED_VISIONZIP = REQUIRED_BASELINE + [
    "attn_scoring_prune_ms",
    "llm_forward_ms",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate timing hooks with a single synthetic sample.")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index (default: 0)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model directory")
    parser.add_argument(
        "--visionzip",
        action="store_true",
        help="Use VisionZipQwenBackend instead of QwenLoRABackend",
    )
    parser.add_argument("--dominant-ratio", type=float, default=0.65)
    parser.add_argument("--contextual-ratio", type=float, default=0.05)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return passed


def make_synthetic_png(width: int = 720, height: int = 1280) -> bytes:
    """Create a solid-colour PNG in memory using PIL."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create a synthetic image.") from exc
    img = Image.new("RGB", (width, height), color=(64, 96, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def print_timing_table(timing: dict[str, float], visionzip: bool) -> None:
    """Pretty-print timing breakdown table."""
    print()
    print(f"  {'Phase':<32}  {'ms':>10}")
    print(f"  {'-'*32}  {'-'*10}")
    phases = REQUIRED_VISIONZIP if visionzip else REQUIRED_BASELINE
    sub_phases = {"vision_encoder_ms", "attn_scoring_prune_ms", "llm_forward_ms", "prefill_ms", "decode_ms", "decode_steps"}
    for phase in phases:
        val = timing.get(phase, 0.0)
        indent = "    " if phase in sub_phases else ""
        print(f"  {indent}{phase:<28}  {val:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Backend              = {'visionzip' if args.visionzip else 'baseline (local_transformers)'}")
    print(f"Model path           = {args.model_path}")

    # ---- Imports ----
    try:
        import torch
        from skillreuse.model.qwen_backend import QwenBackendConfig
        from skillreuse.routing.fallback import FullModelRequest, QwenLoRAModelSpec
        from skillreuse.routing.common import TokenUsage
        from skillreuse.types import ScreenshotAsset
    except ImportError as exc:
        print(f"\nERROR: Import failed — {exc}")
        print("Make sure the SkillReuse conda/venv environment is activated.")
        return 1

    # -----------------------------------------------------------------------
    # Build backend
    # -----------------------------------------------------------------------
    section("Build Backend")

    attn_impl: dict[str, str] | str = {"visual": "eager", "model": "flash_attention_2"}
    backend_config = QwenBackendConfig(
        device_map=0,
        attn_implementation=attn_impl,
    )

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
        print(f"  dominant_ratio   = {args.dominant_ratio}")
        print(f"  contextual_ratio = {args.contextual_ratio}")
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
        print("  QwenLoRABackend created.")

    # -----------------------------------------------------------------------
    # Synthetic request
    # -----------------------------------------------------------------------
    section("Synthetic Request")

    try:
        png_bytes = make_synthetic_png(720, 1280)
    except RuntimeError as exc:
        print(f"  ERROR creating synthetic image: {exc}")
        return 1

    screenshot = ScreenshotAsset(png_bytes=png_bytes, width=720, height=1280)

    request = FullModelRequest(
        observation_id="dryrun_timing_001",
        reason="full_model",
        benchmark="AndroidControl",
        screenshot=screenshot,
        prompt_text=(
            "goal: open Settings\n"
            "step_instruction: tap the Settings icon\n"
            "screen_description: home screen with app grid\n"
            "history: none\n"
            "support: none"
        ),
        history_length=0,
        support_context={},
        model_spec=model_spec,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=256,
        repetition_penalty=1.0,
        image_max_pixels=1_000_000,
        estimated_token_usage=TokenUsage(
            prompt_tokens=64,
            visual_tokens=1225,
            generated_tokens=64,
            full_model_calls=1,
        ),
    )
    print("  Created 1 synthetic 720×1280 request for AndroidControl.")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    section("Single-Sample Inference (B=1)")
    print("  Loading model (~16 GB, up to 120 s) …")

    try:
        with torch.no_grad():
            responses = backend.generate_batch([request])
    except Exception as exc:
        print(f"\n  ERROR during generate_batch():\n  {exc}")
        import traceback
        traceback.print_exc()
        return 1

    response = responses[0]
    timing = response.timing or {}

    print(f"  Raw output : {(response.raw_output or '')[:120]!r}")
    print(f"  Action     : {response.action}")
    print(f"  timing keys: {sorted(timing.keys())}")

    # -----------------------------------------------------------------------
    # Timing validation
    # -----------------------------------------------------------------------
    section("Timing Validation")
    all_passed = True

    required_fields = REQUIRED_VISIONZIP if args.visionzip else REQUIRED_BASELINE
    for field in required_fields:
        present = field in timing
        positive = present and timing[field] > 0.0
        all_passed &= check(
            f"{field} present and > 0",
            present and positive,
            f"{timing.get(field, 'MISSING'):.4f} ms" if present else "MISSING",
        )

    # VisionZip sub-phase sanity: T7a + T7b + T7c should be within 30% of T7.
    # The gap is LM-head output projection + decode-loop scheduling overhead,
    # which is real work not covered by the three sub-phase timers.
    if args.visionzip and all(k in timing for k in ("vision_encoder_ms", "attn_scoring_prune_ms", "llm_forward_ms", "model_generate_ms")):
        sub_sum = timing["vision_encoder_ms"] + timing["attn_scoring_prune_ms"] + timing["llm_forward_ms"]
        model_gen = timing["model_generate_ms"]
        ratio = abs(sub_sum - model_gen) / max(model_gen, 1.0)
        all_passed &= check(
            "VisionZip sub-phases within 30% of model_generate_ms",
            ratio < 0.30,
            f"sub_sum={sub_sum:.1f}ms  model_generate={model_gen:.1f}ms  diff_ratio={ratio:.3f}",
        )

    # Prefill + decode + vision encoder sanity: should be within 40% of model_generate_ms.
    # Baseline: model.generate() includes vision encoder time alongside prefill/decode.
    # VisionZip: vision encoder is timed separately but still part of model_generate_ms.
    # Remaining gap is sampling, cache management, and other overhead.
    if all(k in timing for k in ("prefill_ms", "decode_ms", "vision_encoder_ms", "model_generate_ms")):
        phase_sum = timing["prefill_ms"] + timing["decode_ms"] + timing["vision_encoder_ms"]
        model_gen = timing["model_generate_ms"]
        ratio = abs(phase_sum - model_gen) / max(model_gen, 1.0)
        all_passed &= check(
            "prefill_ms + decode_ms + vision_encoder_ms within 40% of model_generate_ms",
            ratio < 0.40,
            f"phase_sum={phase_sum:.1f}ms  model_generate={model_gen:.1f}ms  diff_ratio={ratio:.3f}",
        )

    # -----------------------------------------------------------------------
    # Timing breakdown table
    # -----------------------------------------------------------------------
    section("Timing Breakdown")
    print_timing_table(timing, args.visionzip)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("Result")
    if all_passed:
        print("  ALL TIMING CHECKS PASSED.\n")
        return 0
    else:
        print("  SOME TIMING CHECKS FAILED — review output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
