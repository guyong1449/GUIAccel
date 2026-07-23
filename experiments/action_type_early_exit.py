#!/usr/bin/env python3
"""Experiment: Action-type conditioned early exit.

Hypothesis: Different GUI action types need vastly different decode
effort. A lightweight classifier on prefill hidden states can predict
the action type before decode begins, enabling type-specific decode
strategies:
  - WAIT/TERMINATE: 0 decode tokens needed
  - SCROLL: 1 token (direction only)
  - NAV: 1-2 tokens (back/home/enter)
  - TYPE: text from instruction extraction (rule-based, 0 decode tokens)
  - CLICK/LONG_PRESS: full coordinate decode needed

Approach:
1. Train a linear probe on prefill output to predict action_type
2. Route to type-specific decode paths
3. Measure latency savings from skipping unnecessary decode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Action types and their decode requirements
ACTION_DECODE_MAP = {
    "WAIT":       {"tokens_needed": 0, "strategy": "skip_decode"},
    "TERMINATE":  {"tokens_needed": 0, "strategy": "skip_decode"},
    "SCROLL":     {"tokens_needed": 1, "strategy": "direction_only"},
    "NAV":        {"tokens_needed": 2, "strategy": "nav_type_only"},
    "TYPE":       {"tokens_needed": 0, "strategy": "rule_extract"},
    "CLICK":      {"tokens_needed": 15, "strategy": "full_decode"},
    "LONG_PRESS": {"tokens_needed": 15, "strategy": "full_decode"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action-type early exit experiment"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="analyze",
                        choices=("analyze", "train_probe", "eval"),
                        help="analyze: action type distribution; "
                             "train_probe: train classifier; eval: compare")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "analyze":
        print("Phase 1: Analyze action type distribution")
        print("  - Count action types in AndroidControl")
        print("  - Compute potential token savings per type")
        print(f"\n  Action decode requirements:")
        for atype, info in ACTION_DECODE_MAP.items():
            print(f"    {atype:12s}: {info['tokens_needed']:2d} tokens → {info['strategy']}")

        # TODO: Load dataset and count action type distribution
        # Then compute: sum(frequency[type] * tokens_saved[type])

    elif args.mode == "train_probe":
        print("Phase 2: Train action-type probe on prefill hidden states")
        print("  - Linear probe: hidden_dim → 7 (action types)")
        print("  - Train on prefill-stage hidden states")
        # TODO: Extract hidden states from prefill, train linear classifier

    elif args.mode == "eval":
        print("Phase 3: Evaluate with action-type early exit")
        print("  - Predict action type → route to efficient decode path")
        print("  - Measure: latency, accuracy, probe accuracy")
        # TODO: Full evaluation pipeline with type-based routing

    print(f"\nExperiment entry point ready. Output: {output_dir}")


if __name__ == "__main__":
    main()
