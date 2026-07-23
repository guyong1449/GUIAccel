#!/usr/bin/env python3
"""Experiment: Grammar-constrained decoding for GUI agent actions.

Hypothesis: GUI actions follow a fixed JSON schema. Constraining the decode
search space via structured output (grammar-guided decoding) can reduce
per-token decode cost by eliminating invalid branches, without changing
output quality.

Approach:
1. Define JSON schema for GUI actions (action_type + bbox + argument)
2. Use vLLM's structured output / guided decoding support
3. Compare latency and accuracy vs unconstrained baseline

Key variables:
- Schema strictness (loose JSON vs exact action grammar)
- Whether to constrain thinking tokens or only action tokens
- Impact on coordinate precision
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# JSON schema for GUI actions
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["CLICK", "TYPE", "SCROLL", "LONG_PRESS", "WAIT", "TERMINATE", "NAV"]
        },
        "bbox": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
        "argument": {"type": "string"},
        "direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"]
        },
    },
    "required": ["action_type"],
}

# Variant: action-only schema (no thinking allowed)
ACTION_ONLY_SCHEMA = {
    **ACTION_SCHEMA,
    "additionalProperties": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrained decoding experiment for GUI actions"
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="Qwen3-VL-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--schema-mode",
        default="action_only",
        choices=("action_only", "thinking_plus_action", "unconstrained"),
        help="How to apply the JSON schema constraint",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Constrained decoding experiment")
    print(f"Schema mode: {args.schema_mode}")
    print(f"Schema: {json.dumps(ACTION_SCHEMA, indent=2)}")

    # TODO: Implement constrained decoding via vLLM guided_json parameter
    # vLLM supports: extra_body={"guided_json": ACTION_SCHEMA}
    # Compare: latency, accuracy, token count vs baseline
    print("Experiment entry point ready.")


if __name__ == "__main__":
    main()
