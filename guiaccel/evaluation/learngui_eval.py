"""LearnGUI benchmark-faithful scoring."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from guiaccel.evaluation.models import StepEvaluationResult
from guiaccel.types import CanonicalAction, DatasetStep
from guiaccel.utils.text import normalize_text, tokenize

_ACTION_RE = re.compile(r"^(?P<name>[A-Z_]+)(?:\[(?P<argument>.*)\])?$")


def score_learngui_prediction(
    predicted_action: CanonicalAction | None,
    step: DatasetStep,
) -> dict[str, bool]:
    ground_truth = _parse_ground_truth(step)
    predicted = _canonical_to_official(predicted_action)
    type_correct = predicted["action_type"] == ground_truth["action_type"]
    action_match = _is_action_match(predicted, ground_truth)
    return {"action_type_correct": type_correct, "action_match": action_match}


def compute_learngui_metrics(results: Sequence[StepEvaluationResult]) -> dict[str, Any]:
    grouped: dict[int, list[StepEvaluationResult]] = defaultdict(list)
    for result in results:
        if result.example.benchmark != "LearnGUI" or result.example.shot_count is None:
            continue
        grouped[int(result.example.shot_count)].append(result)
    report: dict[str, Any] = {}
    for shot_count in sorted(grouped):
        shot_results = grouped[shot_count]
        baseline_type = [result.baseline.metrics.get("action_type_correct", False) for result in shot_results]
        baseline_match = [result.baseline.metrics.get("action_match", False) for result in shot_results]
        hybrid_type = [result.hybrid.metrics.get("action_type_correct", False) for result in shot_results]
        hybrid_match = [result.hybrid.metrics.get("action_match", False) for result in shot_results]
        key = f"{shot_count}_shot"
        report[key] = {
            "baseline": {
                "action_type_accuracy": _fraction(baseline_type),
                "action_match_accuracy": _fraction(baseline_match),
            },
            "hybrid": {
                "action_type_accuracy": _fraction(hybrid_type),
                "action_match_accuracy": _fraction(hybrid_match),
            },
            "delta": {
                "action_type_accuracy": _fraction(hybrid_type) - _fraction(baseline_type),
                "action_match_accuracy": _fraction(hybrid_match) - _fraction(baseline_match),
            },
        }
    return report


def _parse_ground_truth(step: DatasetStep) -> dict[str, Any]:
    raw = str(step.raw_action.get("low_level_action") or "")
    match = _ACTION_RE.match(raw.strip())
    action_name = (match.group("name") if match else raw).upper()
    argument = match.group("argument") if match else None
    touch_coord = step.raw_action.get("touch_coord") or ()
    device_dim = step.metadata.get("device_dim") or ()
    screen_width = int(device_dim[0]) if isinstance(device_dim, (list, tuple)) and len(device_dim) == 2 else int(step.screenshot.width or 1)
    return {
        "action_type": action_name,
        "argument": normalize_text(argument),
        "text": normalize_text(step.raw_action.get("type_text") or argument),
        "touch_coord": tuple(int(value) for value in touch_coord) if len(touch_coord) == 2 else None,
        "screen_width": max(1, screen_width),
    }


def _canonical_to_official(action: CanonicalAction | None) -> dict[str, Any]:
    if action is None:
        return {"action_type": None}
    action_type = action.action_type.upper()
    if action_type == "CLICK":
        return {"action_type": "CLICK", "touch_coord": _center_of_bbox(action.bbox)}
    if action_type == "TYPE":
        return {"action_type": "TYPE", "text": normalize_text(action.argument)}
    if action_type == "SCROLL":
        return {"action_type": "SWIPE", "direction": normalize_text(action.direction)}
    if action_type == "NAV":
        mapping = {"back": "PRESS_BACK", "home": "PRESS_HOME", "enter": "PRESS_ENTER"}
        return {"action_type": mapping.get(normalize_text(action.argument), "NAV")}
    if action_type == "TERMINATE":
        return {"action_type": "TASK_COMPLETE"}
    return {"action_type": action_type}


def _is_action_match(predicted: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> bool:
    if predicted.get("action_type") != ground_truth.get("action_type"):
        return False
    action_type = ground_truth.get("action_type")
    if action_type == "CLICK":
        if predicted.get("touch_coord") is None or ground_truth.get("touch_coord") is None:
            return False
        tolerance_pixels = int(float(ground_truth["screen_width"]) * 0.14)
        return math.dist(predicted["touch_coord"], ground_truth["touch_coord"]) <= tolerance_pixels
    if action_type == "TYPE":
        return _f1_overlap(predicted.get("text"), ground_truth.get("text")) >= 0.5
    if action_type == "SWIPE":
        # ground_truth direction is stored in "argument" (the bracketed part of SWIPE[dir]).
        # "direction" is stored in predicted (from _canonical_to_official SCROLL → SWIPE).
        # When ground_truth has no bracketed argument, direction is unspecified → accept any.
        gt_direction = normalize_text(ground_truth.get("argument"))
        if not gt_direction:
            return True
        return normalize_text(predicted.get("direction")) == gt_direction
    return True


def _center_of_bbox(bbox: tuple[int, int, int, int] | None) -> tuple[int, int] | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (int(round((left + right) / 2.0)), int(round((top + bottom) / 2.0)))


def _f1_overlap(predicted_text: str | None, ground_truth_text: str | None) -> float:
    predicted_tokens = set(tokenize(predicted_text))
    ground_truth_tokens = set(tokenize(ground_truth_text))
    if not predicted_tokens and not ground_truth_tokens:
        return 1.0
    if not predicted_tokens or not ground_truth_tokens:
        return 0.0
    overlap = predicted_tokens & ground_truth_tokens
    precision = float(len(overlap)) / float(len(predicted_tokens))
    recall = float(len(overlap)) / float(len(ground_truth_tokens))
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _fraction(values: Sequence[bool]) -> float:
    return float(sum(bool(value) for value in values)) / float(len(values)) if values else 0.0

