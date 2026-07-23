#!/usr/bin/env python3
"""Baseline evaluation — run Qwen3-VL-8B with standard CoT on AndroidControl.

This establishes the baseline numbers (accuracy, latency, token counts)
that all acceleration experiments compare against.

Usage:
    python experiments/baseline_eval.py \
        --api-base http://127.0.0.1:8000/v1 \
        --output-dir outputs/baseline \
        --split test \
        --task-limit 100  # optional: subset for quick iteration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline GUI agent evaluation")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="Qwen3-VL-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--instruction-mode", default="low_level",
                        choices=("low_level", "high_level"))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    from guiaccel.data import AndroidControlDataset
    from guiaccel.evaluation import score_android_prediction

    # Load dataset
    dataset = AndroidControlDataset()
    episodes = dataset.load_episodes(split=args.split)
    if args.task_limit:
        episodes = episodes[:args.task_limit]

    print(f"Loaded {len(episodes)} episodes from {args.split} split")
    print(f"Model: {args.model_name} at {args.api_base}")
    print(f"Instruction mode: {args.instruction_mode}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement baseline evaluation loop
    # This is the entry point — wire up:
    # 1. VLLMOpenAIBackend for model inference
    # 2. AndroidControl step iteration
    # 3. Scoring via score_android_prediction
    # 4. Latency and token usage tracking
    print("Baseline evaluation entry point ready.")
    print(f"Output will be saved to: {output_dir}")


if __name__ == "__main__":
    main()
