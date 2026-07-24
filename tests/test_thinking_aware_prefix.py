"""CPU unit tests for E1-A thinking-aware GT-Forcing prefix + meta merge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guiaccel.model.hidden_state_extractor import (  # noqa: E402
    TOOL_CALL_UP_TO_ACTION,
    build_gt_prefix,
    build_template_thinking,
    worker_meta_list,
)


def _fake_step(instruction: str = "Open Settings") -> SimpleNamespace:
    return SimpleNamespace(metadata={"step_instruction": instruction})


class TestThinkingAwarePrefix(unittest.TestCase):
    def test_template_prefix_contains_thinking_then_tool_call(self) -> None:
        prefix = build_gt_prefix(
            _fake_step("Tap Connected devices"),
            action_type="click",
            thinking_mode="template",
            extract_point="action",
        )
        self.assertIn("<thinking>", prefix)
        self.assertIn("</thinking>", prefix)
        self.assertIn("Tap Connected devices", prefix)
        self.assertIn(TOOL_CALL_UP_TO_ACTION["click"], prefix)
        # Ordering: thinking closes before tool_call
        self.assertLess(prefix.index("</thinking>"), prefix.index("<tool_call>"))
        self.assertTrue(prefix.rstrip().endswith('"click"'))

    def test_long_press_action_keyword(self) -> None:
        prefix = build_gt_prefix(
            _fake_step(),
            action_type="long_press",
            thinking_mode="template",
            extract_point="action",
        )
        self.assertIn("<thinking>", prefix)
        self.assertTrue(prefix.rstrip().endswith('"long_press"'))

    def test_none_mode_matches_legacy_tool_only(self) -> None:
        prefix = build_gt_prefix(
            _fake_step(),
            action_type="click",
            thinking_mode="none",
            extract_point="action",
        )
        self.assertNotIn("<thinking>", prefix)
        self.assertEqual(prefix, TOOL_CALL_UP_TO_ACTION["click"])

    def test_thinking_end_stops_before_tool_call(self) -> None:
        prefix = build_gt_prefix(
            _fake_step("Scroll down"),
            action_type="click",
            thinking_mode="template",
            extract_point="thinking_end",
        )
        self.assertIn("<thinking>", prefix)
        self.assertTrue(prefix.rstrip().endswith("</thinking>"))
        self.assertNotIn("<tool_call>", prefix)

    def test_coord_bracket_extends_past_action(self) -> None:
        prefix = build_gt_prefix(
            _fake_step(),
            action_type="click",
            thinking_mode="template",
            extract_point="coord_bracket",
        )
        self.assertIn("<thinking>", prefix)
        self.assertIn('"coordinate": [', prefix)
        self.assertGreater(
            len(prefix),
            len(build_gt_prefix(
                _fake_step(),
                action_type="click",
                thinking_mode="template",
                extract_point="action",
            )),
        )

    def test_thinking_end_rejects_none_mode(self) -> None:
        with self.assertRaises(ValueError):
            build_gt_prefix(
                _fake_step(),
                action_type="click",
                thinking_mode="none",
                extract_point="thinking_end",
            )

    def test_ar_cache_requires_thinking_text(self) -> None:
        with self.assertRaises(ValueError):
            build_gt_prefix(
                _fake_step(),
                action_type="click",
                thinking_mode="ar_cache",
                extract_point="action",
            )
        prefix = build_gt_prefix(
            _fake_step(),
            action_type="click",
            thinking_mode="ar_cache",
            extract_point="action",
            thinking_text="I will tap the settings icon.",
        )
        self.assertIn("<thinking>", prefix)
        self.assertIn("settings icon", prefix)
        self.assertIn("<tool_call>", prefix)

    def test_template_thinking_without_instruction(self) -> None:
        text = build_template_thinking(_fake_step(""), "click")
        self.assertIn("<thinking>", text)
        self.assertIn("click", text.lower())

    def test_worker_meta_list_prefers_metadata_falls_back_to_meta(self) -> None:
        # Simulates the pre-fix bug: workers wrote ``meta``, merge read ``metadata``.
        worker_shard = {
            "hidden_states": "tensor_placeholder",
            "meta": [{"episode_id": "a"}, {"episode_id": "b"}],
        }
        merged = {"metadata": []}
        merged["metadata"].extend(worker_meta_list(worker_shard))
        self.assertEqual(len(merged["metadata"]), 2)
        self.assertEqual(merged["metadata"][0]["episode_id"], "a")

        both = {"meta": [{"x": 1}], "metadata": [{"x": 2}]}
        # Prefer explicit metadata when present
        self.assertEqual(worker_meta_list(both), [{"x": 2}])

        empty = {}
        self.assertEqual(worker_meta_list(empty), [])


if __name__ == "__main__":
    unittest.main()
