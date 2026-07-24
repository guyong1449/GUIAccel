"""Unit tests for Qwen action-output parsing helpers."""

from __future__ import annotations

from skillreuse.model.qwen_backend import _coerce_bbox, _payload_to_action


def test_coerce_bbox_rejects_none_entries() -> None:
    assert _coerce_bbox([10, 20, None, 40]) is None
    assert _coerce_bbox((1, 2, 3, None)) is None


def test_coerce_bbox_accepts_valid_coordinates() -> None:
    assert _coerce_bbox([10, 20, 30, 40]) == (10, 20, 30, 40)
    assert _coerce_bbox((1.0, 2.0, 3.0, 4.0)) == (1, 2, 3, 4)


def test_payload_to_action_survives_malformed_bbox() -> None:
    action = _payload_to_action(
        {
            "action_type": "CLICK",
            "bbox": [100, 200, None, 400],
        }
    )
    assert action.action_type == "CLICK"
    assert action.bbox is None
