"""AndroidControl benchmark-faithful scoring."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from guiaccel.evaluation.models import StepEvaluationResult
from guiaccel.types import CanonicalAction, DatasetStep
from guiaccel.utils.text import normalize_text, tokenize

ANDROID_PARTITION_NAMES = {
    "IDD": "in_distribution",
    "app_unseen": "app_unseen",
    "task_unseen": "task_unseen",
    "category_unseen": "category_unseen",
}
ANDROID_PARTITION_ORDER = (
    "in_distribution",
    "app_unseen",
    "task_unseen",
    "category_unseen",
)


def score_android_prediction(
    predicted_action: CanonicalAction | None,
    step: DatasetStep,
) -> dict[str, bool]:
    ground_truth = _parse_android_ground_truth(step)
    predicted = _canonical_to_android(predicted_action, step)
    return {"step_match": _actions_match(predicted, ground_truth)}


def compute_android_metrics(results: Sequence[StepEvaluationResult]) -> dict[str, Any]:
    android_results = [result for result in results if result.example.benchmark == "AndroidControl"]
    partitioned: dict[str, list[StepEvaluationResult]] = defaultdict(list)
    for result in android_results:
        for partition in _result_partitions(result):
            partitioned[partition].append(result)

    report: dict[str, Any] = {}
    ordered_partitions = [
        partition for partition in ANDROID_PARTITION_ORDER if partition in partitioned
    ] + sorted(partition for partition in partitioned if partition not in ANDROID_PARTITION_ORDER)
    for partition in ordered_partitions:
        report[partition] = _summarize_partition(partitioned[partition])
    if android_results:
        report["overall"] = _summarize_partition(android_results)
    return report


def _summarize_partition(results: Sequence[StepEvaluationResult]) -> dict[str, Any]:
    high_level = [result for result in results if result.example.instruction_mode == "high_level"]
    low_level = [result for result in results if result.example.instruction_mode == "low_level"]
    return {
        "counts": {
            "high_level_steps": len(high_level),
            "low_level_steps": len(low_level),
            "high_level_episodes": len({result.example.episode_id for result in high_level}),
        },
        "baseline": {
            "high_level_step_accuracy": _fraction(result.baseline.primary_correct for result in high_level),
            "low_level_step_accuracy": _fraction(result.baseline.primary_correct for result in low_level),
            "high_level_episode_accuracy": _episode_accuracy(high_level, use_hybrid=False),
        },
        "hybrid": {
            "high_level_step_accuracy": _fraction(result.hybrid.primary_correct for result in high_level),
            "low_level_step_accuracy": _fraction(result.hybrid.primary_correct for result in low_level),
            "high_level_episode_accuracy": _episode_accuracy(high_level, use_hybrid=True),
        },
        "delta": {
            "high_level_step_accuracy": _fraction(result.hybrid.primary_correct for result in high_level)
            - _fraction(result.baseline.primary_correct for result in high_level),
            "low_level_step_accuracy": _fraction(result.hybrid.primary_correct for result in low_level)
            - _fraction(result.baseline.primary_correct for result in low_level),
            "high_level_episode_accuracy": _episode_accuracy(high_level, use_hybrid=True)
            - _episode_accuracy(high_level, use_hybrid=False),
        },
    }


def _episode_accuracy(results: Sequence[StepEvaluationResult], *, use_hybrid: bool) -> float:
    by_episode: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_episode[result.example.episode_id].append(
            result.hybrid.primary_correct if use_hybrid else result.baseline.primary_correct
        )
    if not by_episode:
        return 0.0
    return _fraction(all(values) for values in by_episode.values())


def _parse_android_ground_truth(step: DatasetStep) -> dict[str, Any]:
    action_payload = dict(step.raw_action)
    target = _extract_target(step, action_payload)
    return {
        "action_type": normalize_text(action_payload.get("action_type")),
        "x": action_payload.get("x"),
        "y": action_payload.get("y"),
        "text": normalize_text(action_payload.get("text")),
        "app_name": normalize_text(action_payload.get("app_name")),
        "direction": normalize_text(action_payload.get("direction")),
        "goal_status": normalize_text(action_payload.get("goal_status")),
        "wait_value": _normalize_wait_value(action_payload.get("seconds") or action_payload.get("duration")),
        "target_bbox": target.get("bbox"),
        "target_label": target.get("label"),
    }


def _canonical_to_android(action: CanonicalAction | None, step: DatasetStep) -> dict[str, Any]:
    if action is None:
        return {"action_type": None}
    action_type = action.action_type.upper()
    if action_type == "CLICK":
        x, y = _center_of_bbox(action.bbox)
        target = _lookup_android_target(step, x, y)
        return {
            "action_type": "click",
            "x": x,
            "y": y,
            "text": normalize_text(action.argument),
            "target_bbox": action.bbox if action.bbox is not None else target.get("bbox"),
            "target_label": target.get("label") or normalize_text(action.argument),
        }
    if action_type == "LONG_PRESS":
        x, y = _center_of_bbox(action.bbox)
        target = _lookup_android_target(step, x, y)
        return {
            "action_type": "long_press",
            "x": x,
            "y": y,
            "text": normalize_text(action.argument),
            "target_bbox": action.bbox if action.bbox is not None else target.get("bbox"),
            "target_label": target.get("label") or normalize_text(action.argument),
        }
    if action_type == "TYPE":
        x, y = _center_of_bbox(action.bbox)
        target = _lookup_android_target(step, x, y)
        return {
            "action_type": "input_text",
            "x": x,
            "y": y,
            "text": normalize_text(action.argument),
            "target_bbox": action.bbox if action.bbox is not None else target.get("bbox"),
            "target_label": target.get("label") or normalize_text(action.argument),
        }
    if action_type == "SCROLL":
        return {"action_type": "scroll", "direction": normalize_text(action.direction)}
    if action_type == "NAV":
        argument = normalize_text(action.argument)
        if argument == "home":
            return {"action_type": "navigate_home"}
        if argument == "back":
            return {"action_type": "navigate_back"}
        return {"action_type": "open_app", "app_name": normalize_text(action.app or action.argument)}
    if action_type == "WAIT":
        return {
            "action_type": "wait",
            "wait_value": _normalize_wait_value(action.argument),
        }
    if action_type == "TERMINATE":
        return {"action_type": "status", "goal_status": normalize_text(action.argument) or "successful"}
    return {"action_type": normalize_text(action.action_type)}


def _actions_match(predicted: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> bool:
    predicted_type = normalize_text(predicted.get("action_type"))
    ground_truth_type = normalize_text(ground_truth.get("action_type"))
    if predicted_type == ground_truth_type:
        return _same_type_actions_match(predicted, ground_truth)
    return _equivalent_actions_match(
        predicted,
        ground_truth,
        predicted_type=predicted_type,
        ground_truth_type=ground_truth_type,
    )


def _same_type_actions_match(predicted: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> bool:
    ground_truth_type = normalize_text(ground_truth.get("action_type"))
    if ground_truth_type in {"click", "long_press"}:
        point = _coerce_point(predicted.get("x"), predicted.get("y"))
        bbox = ground_truth.get("target_bbox")
        if point is None or bbox is None:
            return False
        return _point_in_bbox(point, bbox)
    if ground_truth_type == "input_text":
        point = _coerce_point(predicted.get("x"), predicted.get("y"))
        bbox = ground_truth.get("target_bbox")
        text_match = normalize_text(predicted.get("text")) == normalize_text(ground_truth.get("text"))
        if point is None or bbox is None:
            return text_match
        return _point_in_bbox(point, bbox) and text_match
    if ground_truth_type == "scroll":
        return normalize_text(predicted.get("direction")) == normalize_text(ground_truth.get("direction"))
    if ground_truth_type == "open_app":
        return normalize_text(predicted.get("app_name")) == normalize_text(ground_truth.get("app_name"))
    if ground_truth_type == "wait":
        ground_truth_wait = _normalize_wait_value(ground_truth.get("wait_value"))
        if ground_truth_wait is None:
            return True
        return _normalize_wait_value(predicted.get("wait_value")) == ground_truth_wait
    if ground_truth_type == "status":
        return normalize_text(predicted.get("goal_status")) == normalize_text(ground_truth.get("goal_status"))
    # Unknown action types: same type names match superficially (no further argument
    # to check), but we do not award credit for unrecognised types produced by future
    # dataset versions to avoid silently inflating accuracy.
    return False


def _equivalent_actions_match(
    predicted: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    predicted_type: str | None,
    ground_truth_type: str | None,
) -> bool:
    if predicted_type == "navigate_back" and ground_truth_type == "click":
        return _is_back_button_click(ground_truth)
    if predicted_type == "click" and ground_truth_type == "navigate_back":
        return _is_back_button_click(predicted)
    if predicted_type == "open_app" and ground_truth_type == "click":
        return _click_matches_app_name(ground_truth, predicted.get("app_name"))
    if predicted_type == "click" and ground_truth_type == "open_app":
        return _click_matches_app_name(predicted, ground_truth.get("app_name"))
    return False


def _extract_target(step: DatasetStep, action_payload: Mapping[str, Any]) -> dict[str, Any]:
    action_type = normalize_text(action_payload.get("action_type"))
    if action_type == "input_text":
        reference_bbox = _coerce_bbox(step.metadata.get("input_text_reference_bbox"))
        reference_label = normalize_text(step.metadata.get("input_text_reference_label"))
        if reference_bbox is not None or reference_label is not None:
            return {"bbox": reference_bbox, "label": reference_label}
    bbox_value = action_payload.get("target_bbox") or action_payload.get("bbox")
    bbox = _coerce_bbox(bbox_value)
    if bbox is not None:
        center_x, center_y = _center_of_bbox(bbox)
        target = _lookup_android_target(step, center_x, center_y)
        return {
            "bbox": bbox,
            "label": target.get("label"),
        }
    x = action_payload.get("x")
    y = action_payload.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        target = _lookup_android_target(step, int(x), int(y))
        if target.get("bbox") is not None:
            return target
        return {"bbox": _point_bbox(int(x), int(y)), "label": None}
    return {"bbox": None, "label": None}


def _lookup_android_target(step: DatasetStep, x: int | None, y: int | None) -> dict[str, Any]:
    if x is None or y is None:
        return {"bbox": None, "label": None}
    candidates: list[tuple[int, int, tuple[int, int, int, int], str | None]] = []
    for node in _android_accessibility_nodes(step):
        bbox = node.get("bbox")
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue
        coerced = tuple(int(value) for value in bbox)
        if coerced[0] <= x <= coerced[2] and coerced[1] <= y <= coerced[3]:
            label = normalize_text(
                node.get("text")
                or node.get("content_description")
                or node.get("view_id_resource_name")
                or node.get("class_name")
            )
            area = max(0, coerced[2] - coerced[0]) * max(0, coerced[3] - coerced[1])
            depth = int(node.get("depth") or 0)
            candidates.append((area, -depth, coerced, label))
    if not candidates:
        return {"bbox": None, "label": None}
    _, _, bbox, label = min(candidates)
    return {"bbox": bbox, "label": label}


def _android_accessibility_nodes(step: DatasetStep) -> tuple[Mapping[str, Any], ...]:
    nodes = step.metadata.get("accessibility_nodes")
    if isinstance(nodes, tuple):
        return tuple(node for node in nodes if isinstance(node, Mapping))
    if isinstance(nodes, list):
        return tuple(node for node in nodes if isinstance(node, Mapping))
    processed_step = step.metadata.get("processed_step")
    if isinstance(processed_step, Mapping):
        nested_nodes = processed_step.get("accessibility_nodes")
        if isinstance(nested_nodes, tuple):
            return tuple(node for node in nested_nodes if isinstance(node, Mapping))
        if isinstance(nested_nodes, list):
            return tuple(node for node in nested_nodes if isinstance(node, Mapping))
    return tuple()


def _result_partitions(result: StepEvaluationResult) -> tuple[str, ...]:
    repository_example = getattr(result.example, "repository_example", None)
    multi_membership_values = []
    singular_membership_values = []
    if repository_example is not None:
        record = getattr(repository_example, "record", None)
        dataset_step = getattr(repository_example, "dataset_step", None)
        if dataset_step is not None and isinstance(getattr(dataset_step, "metadata", None), Mapping):
            multi_membership_values.append(dataset_step.metadata.get("test_subsplits"))
            singular_membership_values.append(dataset_step.metadata.get("test_subsplit"))
        if record is not None:
            normalized_metadata = getattr(record, "normalized_metadata", {})
            if isinstance(normalized_metadata, Mapping):
                multi_membership_values.append(normalized_metadata.get("test_subsplits"))
                singular_membership_values.append(normalized_metadata.get("test_subsplit"))
    for value in multi_membership_values + singular_membership_values + [getattr(result.example, "partition", None)]:
        partitions = _coerce_partition_memberships(value)
        if partitions:
            return partitions
    return ("in_distribution",)


def _coerce_partition_memberships(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        materialized: list[str] = []
        for item in value:
            normalized = ANDROID_PARTITION_NAMES.get(str(item), str(item))
            if normalized not in materialized:
                materialized.append(normalized)
        return tuple(materialized)
    if value is None:
        return tuple()
    normalized = ANDROID_PARTITION_NAMES.get(str(value), str(value))
    return (normalized,) if normalized else tuple()


def _point_bbox(x: int, y: int, padding: int = 1) -> tuple[int, int, int, int]:
    return (int(x - padding), int(y - padding), int(x + padding), int(y + padding))


def _center_of_bbox(bbox: tuple[int, int, int, int] | None) -> tuple[int | None, int | None]:
    if bbox is None:
        return (None, None)
    left, top, right, bottom = bbox
    return (int(round((left + right) / 2.0)), int(round((top + bottom) / 2.0)))


def _coerce_point(x: Any, y: Any) -> tuple[int, int] | None:
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return (int(x), int(y))


def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    left, top, right, bottom = (int(item) for item in value)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _normalize_wait_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(str(value))
    return None if normalized == "wait" else normalized


def _is_back_button_click(action: Mapping[str, Any]) -> bool:
    label_tokens = set(tokenize(action.get("target_label") or action.get("text")))
    if not label_tokens:
        return False
    return "back" in label_tokens or {"navigate", "up"}.issubset(label_tokens)


def _click_matches_app_name(action: Mapping[str, Any], app_name: Any) -> bool:
    label = normalize_text(action.get("target_label") or action.get("text"))
    app = normalize_text(str(app_name)) if app_name is not None else None
    return bool(label and app and label == app)


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _fraction(values: Sequence[bool] | Sequence[int] | Sequence[float]) -> float:
    materialized = list(values)
    return float(sum(bool(value) for value in materialized)) / float(len(materialized)) if materialized else 0.0
