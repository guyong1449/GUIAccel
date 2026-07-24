"""Benchmark-specific canonicalization."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from guiaccel.types import BBox, CanonicalAction, CanonicalStepRecord, DatasetStep
from guiaccel.utils.text import lexical_alignment, normalize_text


def _point_bbox(x: int, y: int, padding: int = 1) -> BBox:
    return (int(x - padding), int(y - padding), int(x + padding), int(y + padding))


def _bbox_area(bbox: BBox) -> int:
    return max(0, int(bbox[2] - bbox[0])) * max(0, int(bbox[3] - bbox[1]))


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance_to_bbox(x: int, y: int, bbox: BBox) -> float:
    center_x, center_y = _bbox_center(bbox)
    return math.dist((x, y), (center_x, center_y))


def _interest_region_to_bbox(interest_region: Any) -> BBox | None:
    if not isinstance(interest_region, list) or len(interest_region) != 2:
        return None
    if not all(isinstance(item, list) and len(item) == 2 for item in interest_region):
        return None
    x1, y1 = interest_region[0]
    x2, y2 = interest_region[1]
    if (x1, y1, x2, y2) == (0, 0, 0, 0):
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _coerce_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    left, top, right, bottom = (int(item) for item in value)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _full_screen_bbox(step: DatasetStep) -> BBox | None:
    device_dim = step.metadata.get("device_dim")
    if isinstance(device_dim, list) and len(device_dim) == 2:
        width, height = int(device_dim[0]), int(device_dim[1])
        return (0, 0, width, height)
    if step.screenshot.width is not None and step.screenshot.height is not None:
        return (0, 0, int(step.screenshot.width), int(step.screenshot.height))
    return None


def _parse_learngui_action(step: DatasetStep) -> CanonicalAction:
    low_level_action = str(step.raw_action.get("low_level_action") or "")
    annotation_action = str(step.raw_action.get("annotation_action") or "")
    action_name = low_level_action
    action_argument: str | None = None
    if "[" in low_level_action and low_level_action.endswith("]"):
        action_name = low_level_action.split("[", 1)[0]
        action_argument = low_level_action.split("[", 1)[1][:-1]
    action_name = action_name.upper()

    touch_coord = step.raw_action.get("touch_coord") or []
    lift_coord = step.raw_action.get("lift_coord") or []
    bbox: BBox | None = None
    if len(touch_coord) == 2:
        bbox = _point_bbox(int(touch_coord[0]), int(touch_coord[1]))

    if action_name in {"CLICK", "TAP"} or annotation_action.upper() == "TAP":
        return CanonicalAction("CLICK", None, bbox, None, None)
    if action_name == "TYPE" or annotation_action.upper() == "TYPE":
        argument = normalize_text(step.raw_action.get("type_text") or action_argument)
        return CanonicalAction("TYPE", argument, bbox, None, None)
    if action_name in {"SWIPE", "SCROLL"} or annotation_action.upper() == "SWIPE":
        direction = None
        if action_argument:
            direction = normalize_text(action_argument)
        elif len(touch_coord) == 2 and len(lift_coord) == 2:
            delta_x = int(lift_coord[0]) - int(touch_coord[0])
            delta_y = int(lift_coord[1]) - int(touch_coord[1])
            if abs(delta_y) >= abs(delta_x):
                direction = "up" if delta_y < 0 else "down"
            else:
                direction = "left" if delta_x < 0 else "right"
        return CanonicalAction("SCROLL", None, _interest_region_to_bbox(step.metadata.get("interest_region")), None, direction)
    if action_name == "PRESS_ENTER":
        return CanonicalAction("NAV", "enter", None, None, None)
    if action_name in {"PRESS_BACK", "BACK"}:
        return CanonicalAction("NAV", "back", None, None, None)
    if action_name in {"PRESS_HOME", "HOME"}:
        return CanonicalAction("NAV", "home", None, None, None)
    if action_name.startswith("TASK_COMPLETE"):
        return CanonicalAction("TERMINATE", "success", None, None, None)
    argument = normalize_text(action_argument)
    return CanonicalAction(action_name, argument, bbox, None, None)


def _collect_learngui_elements(step: DatasetStep) -> list[dict[str, Any]]:
    screen_annotation = step.metadata.get("screen_annotation") or {}
    elements: list[dict[str, Any]] = []
    for group_name, value in screen_annotation.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict) or "bbox" not in item:
                continue
            bbox = tuple(int(number) for number in item["bbox"])
            if len(bbox) != 4:
                continue
            element = dict(item)
            element["bbox"] = bbox
            element["group_name"] = group_name
            elements.append(element)
    return elements


def _bind_learngui_target(step: DatasetStep, canonical_action: CanonicalAction) -> dict[str, Any] | None:
    elements = _collect_learngui_elements(step)
    touch_coord = step.raw_action.get("touch_coord") or []
    if canonical_action.action_type in {"CLICK", "TYPE"} and len(touch_coord) == 2:
        x, y = int(touch_coord[0]), int(touch_coord[1])
        containing = [
            element
            for element in elements
            if element["bbox"][0] <= x <= element["bbox"][2] and element["bbox"][1] <= y <= element["bbox"][3]
        ]
        candidates = containing or elements
        if candidates:
            best = min(candidates, key=lambda item: _distance_to_bbox(x, y, item["bbox"]))
            target_text = " ".join(best.get("xml_desc") or [])
            return {
                "type": "element",
                "source": "element_annotation",
                "label": normalize_text(target_text),
                "functionality": normalize_text(best.get("functionality")),
                "bbox": best["bbox"],
                "group_name": best.get("group_name"),
            }
        return {
            "type": "pseudo_element",
            "source": "touch_coordinate",
            "label": None,
            "bbox": _point_bbox(x, y),
        }

    if canonical_action.action_type == "SCROLL":
        region = canonical_action.bbox or _full_screen_bbox(step)
        return {
            "type": "pseudo_element",
            "source": "interest_region" if canonical_action.bbox is not None else "full_screen",
            "label": "scroll_region",
            "bbox": region,
        }
    if canonical_action.action_type == "NAV":
        return {
            "type": "pseudo_element",
            "source": "navigation",
            "label": canonical_action.argument or canonical_action.app,
            "bbox": None,
        }
    if canonical_action.action_type == "TERMINATE":
        return {
            "type": "pseudo_element",
            "source": "task_status",
            "label": canonical_action.argument,
            "bbox": None,
        }
    return None


def _normalized_learngui_metadata(step: DatasetStep, bound_target: dict[str, Any] | None) -> dict[str, Any]:
    screen_annotation = step.metadata.get("screen_annotation") or {}
    return {
        "benchmark": "LearnGUI",
        "representation": "screenshot-centric",
        "app": step.metadata.get("app"),
        "domain": step.metadata.get("domain"),
        "split": step.metadata.get("split"),
        "step_id": step.metadata.get("step_id"),
        "action_description": step.metadata.get("action_description"),
        "package_name": step.metadata.get("package_name"),
        "screen_size": step.metadata.get("device_dim"),
        "page_caption": screen_annotation.get("page_caption"),
        "clickable_element_count": len(_collect_learngui_elements(step)),
        "discovery_binding_used_element_artifacts": bound_target is not None,
    }


def _learngui_effect_delta(step: DatasetStep, canonical_action: CanonicalAction) -> dict[str, Any]:
    next_metadata = step.next_metadata or {}
    return {
        "terminal": canonical_action.action_type == "TERMINATE" or step.next_screenshot is None,
        "app_changed": bool(step.metadata.get("package_name")) and step.metadata.get("package_name") != next_metadata.get("package_name"),
        "next_action_description": next_metadata.get("action_description"),
        "page_caption_changed": (step.metadata.get("screen_annotation") or {}).get("page_caption")
        != (next_metadata.get("screen_annotation") or {}).get("page_caption"),
        "next_screenshot_available": step.next_screenshot is not None,
    }


def _canonicalize_learngui(step: DatasetStep) -> CanonicalStepRecord:
    canonical_action = _parse_learngui_action(step)
    bound_target = _bind_learngui_target(step, canonical_action)
    if canonical_action.bbox is None and bound_target and bound_target.get("bbox") is not None:
        canonical_action = CanonicalAction(
            canonical_action.action_type,
            canonical_action.argument,
            bound_target["bbox"],
            canonical_action.app,
            canonical_action.direction,
        )
    screen_text = None
    if bound_target is not None:
        screen_text = " ".join(
            item
            for item in [
                bound_target.get("label"),
                bound_target.get("functionality"),
                (step.metadata.get("screen_annotation") or {}).get("page_caption"),
            ]
            if item
        )
    lexical = lexical_alignment(
        goal=step.goal,
        screen_text=screen_text,
        action_argument=canonical_action.argument,
        action_text=step.metadata.get("action_description"),
        target_text=bound_target.get("label") if bound_target else None,
    )
    return CanonicalStepRecord(
        goal=step.goal,
        screenshot=step.screenshot,
        normalized_metadata=_normalized_learngui_metadata(step, bound_target),
        canonical_action=canonical_action,
        effect_delta=_learngui_effect_delta(step, canonical_action),
        bound_target=bound_target,
        lexical_alignment=lexical,
    )


def _point_target_from_android_action(action_payload: dict[str, Any]) -> dict[str, Any] | None:
    if "x" in action_payload and "y" in action_payload:
        x, y = int(action_payload["x"]), int(action_payload["y"])
        return {
            "type": "pseudo_element",
            "source": "action_coordinate",
            "label": None,
            "bbox": _point_bbox(x, y),
        }
    return None


def _canonicalize_android_action(step: DatasetStep) -> CanonicalAction:
    action_payload = step.raw_action
    action_type = str(action_payload.get("action_type", "")).lower()
    if action_type == "click":
        point_target = _point_target_from_android_action(action_payload)
        bbox = point_target["bbox"] if point_target is not None else None
        return CanonicalAction("CLICK", None, bbox, None, None)
    if action_type == "long_press":
        point_target = _point_target_from_android_action(action_payload)
        bbox = point_target["bbox"] if point_target is not None else None
        return CanonicalAction("LONG_PRESS", None, bbox, None, None)
    if action_type == "scroll":
        return CanonicalAction("SCROLL", None, None, None, normalize_text(action_payload.get("direction")))
    if action_type == "open_app":
        app_name = normalize_text(action_payload.get("app_name"))
        return CanonicalAction("NAV", app_name, None, app_name, None)
    if action_type in {"input_text", "type"}:
        return CanonicalAction("TYPE", normalize_text(action_payload.get("text")), None, None, None)
    if action_type == "navigate_home":
        return CanonicalAction("NAV", "home", None, None, None)
    if action_type == "navigate_back":
        return CanonicalAction("NAV", "back", None, None, None)
    if action_type == "wait":
        return CanonicalAction("WAIT", normalize_text(str(action_payload.get("seconds") or action_payload.get("duration") or "wait")), None, None, None)
    if action_type in {"status", "terminate"}:
        return CanonicalAction(
            "TERMINATE",
            normalize_text(action_payload.get("goal_status") or action_payload.get("status") or "successful"),
            None,
            None,
            None,
        )
    return CanonicalAction(action_type.upper(), None, None, None, None)


def _bind_android_target(step: DatasetStep, canonical_action: CanonicalAction) -> dict[str, Any] | None:
    action_payload = step.raw_action
    if canonical_action.action_type in {"CLICK", "LONG_PRESS"}:
        point_target = _point_target_from_android_action(action_payload)
        if point_target is None:
            return None
        x = action_payload.get("x")
        y = action_payload.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            bound_target = _android_accessibility_target(step, int(x), int(y))
            if bound_target is not None:
                return bound_target
        return point_target
    if canonical_action.action_type == "TYPE":
        target_bbox = _coerce_bbox(
            step.metadata.get("action_target_bbox")
            or step.metadata.get("input_text_reference_bbox")
            or action_payload.get("target_bbox")
            or action_payload.get("bbox")
        )
        target_label = normalize_text(
            step.metadata.get("action_target_label")
            or step.metadata.get("input_text_reference_label")
            or action_payload.get("target_label")
        )
        return {
            "type": "element" if target_bbox is not None else "pseudo_element",
            "source": "input_text_target" if target_bbox is not None else "input_text_action",
            "label": target_label or canonical_action.argument,
            "bbox": target_bbox,
        }
    if canonical_action.action_type == "SCROLL":
        return {
            "type": "pseudo_element",
            "source": "scroll_action",
            "label": canonical_action.direction,
            "bbox": None,
        }
    if canonical_action.action_type == "NAV":
        return {
            "type": "pseudo_element",
            "source": "navigation_action",
            "label": canonical_action.argument or canonical_action.app,
            "bbox": None,
        }
    if canonical_action.action_type == "WAIT":
        return {
            "type": "pseudo_element",
            "source": "wait_action",
            "label": canonical_action.argument,
            "bbox": None,
        }
    return None


def _normalized_android_metadata(step: DatasetStep) -> dict[str, Any]:
    return {
        "benchmark": "AndroidControl",
        "representation": "processed-step",
        "split": step.metadata.get("split"),
        "test_subsplit": step.metadata.get("test_subsplit"),
        "test_subsplits": tuple(step.metadata.get("test_subsplits") or ()),
        "episode_id": step.metadata.get("episode_id"),
        "step_index": step.metadata.get("step_index"),
        "high_level_step_index": step.metadata.get("high_level_step_index"),
        "step_instruction": step.metadata.get("step_instruction"),
        "screen_description": step.metadata.get("screen_description"),
        "screen_description_source": step.metadata.get("screen_description_source"),
        "screenshot_size": (
            step.metadata.get("screenshot_width"),
            step.metadata.get("screenshot_height"),
        ),
        "accessibility_text_fragments": tuple(step.metadata.get("accessibility_text_fragments") or ()),
        "accessibility_nodes": tuple(step.metadata.get("accessibility_nodes") or ()),
        "accessibility_node_count": step.metadata.get("accessibility_node_count"),
        "actionable_accessibility_node_count": step.metadata.get("actionable_accessibility_node_count"),
        "history_action_types": list(step.metadata.get("history_action_types") or []),
        "processed_step": dict(step.metadata.get("processed_step") or {}),
    }


def _android_effect_delta(step: DatasetStep, canonical_action: CanonicalAction) -> dict[str, Any]:
    next_metadata = step.next_metadata or {}
    return {
        "terminal": step.next_screenshot is None,
        "screen_description_changed": step.metadata.get("screen_description") != next_metadata.get("screen_description"),
        "step_instruction_changed": step.metadata.get("step_instruction") != next_metadata.get("step_instruction"),
        "action_type": canonical_action.action_type,
        "next_screenshot_available": step.next_screenshot is not None,
    }


def _canonicalize_android(step: DatasetStep) -> CanonicalStepRecord:
    canonical_action = _canonicalize_android_action(step)
    bound_target = _bind_android_target(step, canonical_action)
    if bound_target and bound_target.get("bbox") is not None and (
        canonical_action.bbox is None or canonical_action.action_type in {"CLICK", "LONG_PRESS"}
    ):
        canonical_action = CanonicalAction(
            canonical_action.action_type,
            canonical_action.argument,
            bound_target["bbox"],
            canonical_action.app,
            canonical_action.direction,
        )
    lexical = lexical_alignment(
        goal=step.goal,
        screen_text=step.metadata.get("screen_description"),
        action_argument=canonical_action.argument,
        action_text=step.metadata.get("step_instruction"),
        target_text=bound_target.get("label") if bound_target else None,
    )
    return CanonicalStepRecord(
        goal=step.goal,
        screenshot=step.screenshot,
        normalized_metadata=_normalized_android_metadata(step),
        canonical_action=canonical_action,
        effect_delta=_android_effect_delta(step, canonical_action),
        bound_target=bound_target,
        lexical_alignment=lexical,
    )


def _android_accessibility_target(step: DatasetStep, x: int, y: int) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, BBox, dict[str, Any]]] = []
    for node in _android_accessibility_nodes(step):
        if not isinstance(node, dict):
            continue
        bbox = node.get("bbox")
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue
        coerced_bbox = tuple(int(value) for value in bbox)
        if not (coerced_bbox[0] <= x <= coerced_bbox[2] and coerced_bbox[1] <= y <= coerced_bbox[3]):
            continue
        candidates.append((_bbox_area(coerced_bbox), int(node.get("depth") or 0), coerced_bbox, node))
    if not candidates:
        return None
    _, _, bbox, node = min(candidates, key=lambda item: (item[0], -item[1]))
    label = normalize_text(
        node.get("text")
        or node.get("content_description")
        or node.get("view_id_resource_name")
        or node.get("class_name")
    )
    return {
        "type": "element",
        "source": "accessibility_node",
        "label": label,
        "bbox": bbox,
        "editable": bool(node.get("editable")),
        "clickable": bool(node.get("clickable")),
        "class_name": normalize_text(node.get("class_name")),
    }


def _android_accessibility_nodes(step: DatasetStep) -> tuple[dict[str, Any], ...]:
    nodes = step.metadata.get("accessibility_nodes")
    if isinstance(nodes, tuple):
        return tuple(node for node in nodes if isinstance(node, dict))
    if isinstance(nodes, list):
        return tuple(node for node in nodes if isinstance(node, dict))
    processed_step = step.metadata.get("processed_step")
    if isinstance(processed_step, dict):
        nested_nodes = processed_step.get("accessibility_nodes")
        if isinstance(nested_nodes, tuple):
            return tuple(node for node in nested_nodes if isinstance(node, dict))
        if isinstance(nested_nodes, list):
            return tuple(node for node in nested_nodes if isinstance(node, dict))
    return tuple()


def canonicalize_step(step: DatasetStep) -> CanonicalStepRecord:
    """Canonicalize a dataset-specific step."""

    if step.dataset == "LearnGUI":
        return _canonicalize_learngui(step)
    if step.dataset == "AndroidControl":
        return _canonicalize_android(step)
    raise ValueError(f"Unsupported dataset for canonicalization: {step.dataset}")
