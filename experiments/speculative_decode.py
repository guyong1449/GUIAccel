#!/usr/bin/env python3
"""Experiment: Speculative decoding for GUI agent VLM inference.

Hypothesis: A lightweight draft model can predict GUI action tokens with
high acceptance rate, since action outputs follow repetitive patterns
(JSON format, limited coordinate vocabulary).

Approach:
1. Use vLLM's built-in speculative decoding (EAGLE / draft model)
2. Measure acceptance rate on GUI action outputs specifically
3. Compare end-to-end latency vs standard autoregressive decode

Key questions:
- What acceptance rate do we get on GUI action tokens?
- Does the draft model handle coordinate tokens well?
- What's the optimal num_speculative_tokens for ~150 token outputs?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPECULATIVE_CONFIGS = {
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 4,
    },
    "eagle": {
        "method": "eagle",
        "num_speculative_tokens": 5,
        # Requires EAGLE head trained for Qwen3-VL
    },
    "draft_model": {
        "method": "draft_model",
        "draft_model": "Qwen/Qwen3-VL-2B-Instruct",  # smaller Qwen as draft
        "num_speculative_tokens": 5,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speculative decoding experiment for GUI agents"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method",
        default="ngram",
        choices=list(SPECULATIVE_CONFIGS.keys()),
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    args = parser.parse_args()

    config = SPECULATIVE_CONFIGS[args.method].copy()
    config["num_speculative_tokens"] = args.num_speculative_tokens

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Speculative decoding experiment")
    print(f"Method: {args.method}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print()
    print("To run, start vLLM with speculative decoding:")
    print(f"  vllm serve models/Qwen3-VL-8B-Instruct \\")
    print(f"    --speculative-config '{json.dumps(config)}'")

    # TODO: Implement evaluation loop with spec-decode vLLM backend
    # Key metrics: acceptance rate, tokens/second, latency, accuracy
    print("\nExperiment entry point ready.")


if __name__ == "__main__":
    main()
