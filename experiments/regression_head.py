#!/usr/bin/env python3
"""Experiment: Regression head for coordinate prediction.

Hypothesis: GUI agent bbox coordinates are continuous values being
generated as discrete text tokens (15-20 decode steps for 4 integers).
A regression head on the VLM's last hidden state can output coordinates
in a single forward pass, eliminating sequential coordinate decode.

Approach:
1. Add a lightweight MLP head on Qwen3-VL's last hidden state
2. Train it to predict [x1, y1, x2, y2] from the hidden state at
   the action_type token position
3. Compare latency: text decode vs single-pass regression

Architecture:
  Qwen3-VL hidden state at [action_type] token
      ↓
  MLP (hidden_dim → 256 → 4)
      ↓
  [x1, y1, x2, y2] normalized coordinates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coordinate regression head experiment"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="extract",
                        choices=("extract", "train", "eval"),
                        help="extract: collect hidden states; train: fit MLP; eval: compare")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="MLP intermediate dimension")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "extract":
        print("Phase 1: Extract hidden states at action_type token positions")
        print("  - Run Qwen3-VL on training data")
        print("  - Save (hidden_state, ground_truth_bbox) pairs")
        # TODO: Hook into Qwen3-VL forward pass to extract hidden states

    elif args.mode == "train":
        print("Phase 2: Train regression MLP")
        print(f"  - Architecture: {4096} → {args.hidden_dim} → 4")
        print("  - Loss: smooth L1 on normalized coordinates")
        # TODO: Simple PyTorch training loop

    elif args.mode == "eval":
        print("Phase 3: Compare regression head vs text decode")
        print("  - Metrics: coordinate MAE, latency, action match rate")
        # TODO: Evaluate with and without regression head

    print(f"\nExperiment entry point ready. Output: {output_dir}")


if __name__ == "__main__":
    main()
