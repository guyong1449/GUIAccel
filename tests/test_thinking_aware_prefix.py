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
        self.assertIn("</thinking>", prefix)
        self.assertTrue(
            prefix.rstrip().endswith("</thinking>") or prefix.endswith("</thinking>\n"),
        )
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


    def test_multi_rejected_by_build_gt_prefix(self) -> None:
        with self.assertRaises(ValueError):
            build_gt_prefix(
                _fake_step(),
                action_type="click",
                thinking_mode="template",
                extract_point="multi",
            )

    def test_multi_point_prefix_token_lengths_nested(self) -> None:
        from guiaccel.model.hidden_state_extractor import _multi_point_prefix_token_lengths

        class _Tok:
            """Minimal tokenizer stub with offset_mapping (prefix-unstable encode)."""

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                # Deliberately non-prefix-stable: append a sentinel that depends on
                # whether the string ends with newline (mirrors Qwen BPE behaviour).
                ids = list(text.encode("utf-8"))
                if text.endswith("\n"):
                    ids.append(999)
                return ids

            def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
                del add_special_tokens
                # Char-level tokens with true offsets (stable indexing path).
                ids = list(range(len(text)))
                offs = [(i, i + 1) for i in range(len(text))]
                out = {"input_ids": ids}
                if return_offsets_mapping:
                    out["offset_mapping"] = offs
                return out

        full, lengths = _multi_point_prefix_token_lengths(
            _Tok(),
            _fake_step("Open WiFi"),
            action_type="click",
            thinking_mode="template",
        )
        self.assertIn('"coordinate": [', full)
        self.assertLessEqual(lengths["thinking_end"], lengths["action"])
        self.assertLessEqual(lengths["action"], lengths["coord_bracket"])
        self.assertGreater(lengths["thinking_end"], 0)
        self.assertEqual(lengths["coord_bracket"], len(full))


if __name__ == "__main__":
    unittest.main()
