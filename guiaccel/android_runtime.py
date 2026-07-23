"""AndroidControl runtime-observable projection helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from guiaccel.types import CanonicalAction, CanonicalStepRecord
from guiaccel.utils.text import lexical_alignment, normalize_text, unique_preserve_order

ANDROID_RUNTIME_FORBIDDEN_SCREEN_FIELDS = frozenset(
    {
        "bound_target.bbox",
        "bound_target.functionality",
        "bound_target.label",
    }
)
ANDROID_RUNTIME_FORBIDDEN_HISTORY_FIELDS = frozenset(
    {
        "bound_target.bbox",
        "bound_target.label",
    }
)
ANDROID_LOCALIZED_ACTION_TYPES = frozenset({"CLICK", "LONG_PRESS", "TYPE"})
ANDROID_MAX_STORED_ACCESSIBILITY_NODES = 192
ANDROID_MAX_RUNTIME_CANDIDATES = 128

_CONTAINER_WIDGETS = frozenset(
    {
        "frame_layout",
        "linear_layout",
        "relative_layout",
        "view",
        "view_group",
    }
)
_INFORMATIVE_WIDGETS = frozenset(
    {
        "button",
        "checkbox",
        "checkedtextview",
        "edittext",
        "imagebutton",
        "imageview",
        "radiobutton",
        "recyclerview",
        "scrollview",
        "seekbar",
        "spinner",
        "switch",
        "tabwidget",
        "textview",
        "webview",
    }
)


def is_forbidden_android_screen_field(screen_field: str) -> bool:
    """Return whether one AndroidControl screen-copy field is GT-only."""

    return str(screen_field) in ANDROID_RUNTIME_FORBIDDEN_SCREEN_FIELDS


def is_forbidden_android_history_field(history_field: str) -> bool:
    """Return whether one AndroidControl history-copy field is GT-only."""

    return str(history_field) in ANDROID_RUNTIME_FORBIDDEN_HISTORY_FIELDS


def project_android_runtime_record(
    record: CanonicalStepRecord,
    *,
    mode: str | None = None,
    preserve_action: bool = False,
) -> CanonicalStepRecord:
    """Project one AndroidControl canonical record into runtime-observable fields only."""

    metadata = {key: _clone_value(value) for key, value in record.normalized_metadata.items()}
    processed_step = {
        key: _clone_value(value)
        for key, value in dict(metadata.get("processed_step") or {}).items()
    }
    metadata["processed_step"] = processed_step
    if mode is not None:
        metadata["instruction_mode"] = mode
        processed_step["instruction_mode"] = mode
    if mode == "high_level":
        metadata["step_instruction"] = None
        processed_step["step_instruction"] = None

    screen_description = metadata.get("screen_description")
    step_instruction = metadata.get("step_instruction")
    metadata["runtime_policy"] = "runtime_observable_only"
    return CanonicalStepRecord(
        goal=record.goal,
        screenshot=record.screenshot,
        normalized_metadata=metadata,
        canonical_action=record.canonical_action if preserve_action else CanonicalAction("UNOBSERVED", None, None, None, None),
        effect_delta={},
        bound_target=None,
        lexical_alignment=lexical_alignment(
            goal=record.goal,
            screen_text=screen_description if isinstance(screen_description, str) else None,
            action_argument=None,
            action_text=step_instruction if isinstance(step_instruction, str) else None,
            target_text=None,
        ),
    )


def is_android_localized_action_type(action_type: str | None) -> bool:
    """Return whether this Android action depends on localized target semantics."""

    return str(action_type or "").upper() in ANDROID_LOCALIZED_ACTION_TYPES


def android_runtime_observed_effect_delta(
    record: CanonicalStepRecord,
) -> dict[str, Any]:
    """Return attributable post-action evidence available to the Android runtime."""

    if is_android_localized_action_type(record.canonical_action.action_type):
        return {}
    return {
        str(key): _clone_value(value)
        for key, value in record.effect_delta.items()
    }


def android_runtime_effect_proxy_kind(record: CanonicalStepRecord) -> str:
    """Describe the strength of the Android offline effect signal."""

    if is_android_localized_action_type(record.canonical_action.action_type):
        return "none_localized_action"
    return "ground_truth_transition_proxy"


def android_runtime_roi_screenshot_change(
    record: CanonicalStepRecord,
) -> bool | None:
    """Return an ROI-local screenshot proxy only when it is not action-localized."""

    if is_android_localized_action_type(record.canonical_action.action_type):
        return None
    if "next_screenshot_available" not in record.effect_delta:
        return None
    return bool(record.effect_delta.get("next_screenshot_available"))


def filter_android_accessibility_nodes(
    nodes: Sequence[Mapping[str, Any]],
    *,
    screen_size: tuple[int | None, int | None] | None,
    max_nodes: int = ANDROID_MAX_STORED_ACCESSIBILITY_NODES,
) -> tuple[dict[str, Any], ...]:
    """Keep only salient runtime-observable accessibility nodes."""

    width, height = screen_size or (None, None)
    screen_area = None
    if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
        screen_area = float(width * height)

    scored: list[tuple[tuple[float, float, float, int], dict[str, Any]]] = []
    seen: set[tuple[Any, ...]] = set()
    for node in nodes:
        prepared = _prepare_accessibility_node(node, screen_area=screen_area)
        if prepared is None:
            continue
        dedupe_key = (
            prepared["bbox"],
            prepared.get("text"),
            prepared.get("content_description"),
            prepared.get("class_name"),
            prepared.get("view_id_resource_name"),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        scored.append((_node_priority(prepared, screen_area=screen_area), prepared))

    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(item for _, item in scored[: int(max_nodes)])


def build_android_runtime_candidates(
    metadata: Mapping[str, Any],
    *,
    max_candidates: int = ANDROID_MAX_RUNTIME_CANDIDATES,
) -> tuple[dict[str, Any], ...]:
    """Convert stored accessibility nodes into runtime screen candidates."""

    nodes = metadata.get("accessibility_nodes") or ()
    candidates: list[tuple[tuple[float, float, float, int], dict[str, Any]]] = []
    seen: set[tuple[Any, ...]] = set()
    width, height = _metadata_screen_size(metadata)
    screen_area = float(width * height) if width and height else None
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        bbox = _coerce_bbox(node.get("bbox"))
        if bbox is None:
            continue
        text = _candidate_text_from_node(node)
        role = _candidate_role_from_node(node)
        key = (bbox, text, role)
        if key in seen:
            continue
        seen.add(key)
        prepared_node = dict(node)
        prepared_node["bbox"] = bbox
        candidates.append(
            (
                _node_priority(prepared_node, screen_area=screen_area),
                {
                    "candidate_id": f"a11y-{index}",
                    "bbox": bbox,
                    "text": text,
                    "role": role,
                    "content_description": _clean_text(node.get("content_description")),
                    "class_name": _clean_text(node.get("class_name")),
                    "view_id_resource_name": _normalize_view_id(node.get("view_id_resource_name")),
                    "clickable": bool(node.get("clickable")),
                    "editable": bool(node.get("editable")),
                    "scrollable": bool(node.get("scrollable")),
                    "checkable": bool(node.get("checkable")),
                    "checked": bool(node.get("checked")),
                    "enabled": bool(node.get("enabled", True)),
                    "visible": bool(node.get("visible", True)),
                    "focusable": bool(node.get("focusable")),
                    "depth": int(node.get("depth") or 0),
                },
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(item for _, item in candidates[: int(max_candidates)])


def _prepare_accessibility_node(
    node: Mapping[str, Any],
    *,
    screen_area: float | None,
) -> dict[str, Any] | None:
    bbox = _coerce_bbox(node.get("bbox"))
    if bbox is None:
        return None
    if node.get("visible") is False:
        return None

    class_name = _clean_text(node.get("class_name"))
    widget = _widget_leaf(class_name)
    text = _clean_text(node.get("text"))
    content_description = _clean_text(node.get("content_description"))
    view_id = _normalize_view_id(node.get("view_id_resource_name"))
    actionable = _node_is_actionable(node)
    area_ratio = _bbox_area_ratio(bbox, screen_area)
    if area_ratio >= 0.95:
        return None
    if not actionable and not any((text, content_description, view_id)):
        return None
    if not actionable and area_ratio > 0.60:
        return None
    if not actionable and widget in _CONTAINER_WIDGETS:
        return None
    if not actionable and text and len(text) > 96:
        return None
    if not actionable and not any((text, content_description)) and widget not in _INFORMATIVE_WIDGETS:
        return None

    return {
        "bbox": bbox,
        "text": text,
        "content_description": content_description,
        "class_name": class_name,
        "view_id_resource_name": view_id,
        "clickable": bool(node.get("clickable")),
        "editable": bool(node.get("editable")),
        "scrollable": bool(node.get("scrollable")),
        "checkable": bool(node.get("checkable")),
        "checked": bool(node.get("checked")),
        "enabled": bool(node.get("enabled", True)),
        "visible": bool(node.get("visible", True)),
        "focusable": bool(node.get("focusable")),
        "depth": int(node.get("depth") or 0),
    }


def _node_priority(
    node: Mapping[str, Any],
    *,
    screen_area: float | None,
) -> tuple[float, float, float, int]:
    bbox = _coerce_bbox(node.get("bbox")) or (0, 0, 0, 0)
    actionable_score = float(_node_is_actionable(node))
    text_score = 1.0 if _candidate_text_from_node(node) else 0.0
    area_ratio = _bbox_area_ratio(bbox, screen_area)
    depth_bonus = int(node.get("depth") or 0)
    return (
        actionable_score,
        text_score,
        -area_ratio,
        depth_bonus,
    )


def _candidate_text_from_node(node: Mapping[str, Any]) -> str | None:
    parts = [
        _clean_text(node.get("text")),
        _clean_text(node.get("content_description")),
        _normalize_view_id(node.get("view_id_resource_name")),
    ]
    cleaned = unique_preserve_order([part for part in parts if part])
    if not cleaned:
        return None
    return " | ".join(cleaned[:2])[:160].rstrip()


def _candidate_role_from_node(node: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    class_name = _widget_leaf(_clean_text(node.get("class_name")))
    if class_name:
        parts.append(class_name)
    for key in ("editable", "clickable", "scrollable", "checkable", "checked"):
        if node.get(key):
            parts.append(key)
    normalized_parts = unique_preserve_order([part for part in parts if part])
    if not normalized_parts:
        return None
    return " ".join(normalized_parts)


def _metadata_screen_size(metadata: Mapping[str, Any]) -> tuple[int | None, int | None]:
    size = metadata.get("screenshot_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        width = int(size[0]) if isinstance(size[0], (int, float)) else None
        height = int(size[1]) if isinstance(size[1], (int, float)) else None
        return width, height
    return None, None


def _bbox_area_ratio(
    bbox: tuple[int, int, int, int],
    screen_area: float | None,
) -> float:
    if screen_area is None or screen_area <= 0.0:
        return 0.0
    left, top, right, bottom = bbox
    return float(max(0, right - left) * max(0, bottom - top)) / float(screen_area)


def _widget_leaf(class_name: str | None) -> str | None:
    normalized = normalize_text(class_name)
    if not normalized:
        return None
    leaf = normalized.rsplit(".", 1)[-1]
    return leaf.replace("$", "_")


def _node_is_actionable(node: Mapping[str, Any]) -> bool:
    return any(bool(node.get(key)) for key in ("clickable", "editable", "scrollable", "checkable"))


def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return None
    left, top, right, bottom = (int(item) for item in value)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _normalize_view_id(value: Any) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    leaf = cleaned.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    if not leaf or leaf.startswith("gbs."):
        return None
    return leaf[:64].rstrip()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned[:160].rstrip() if cleaned else None


def _clone_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


__all__ = [
    "ANDROID_MAX_RUNTIME_CANDIDATES",
    "ANDROID_MAX_STORED_ACCESSIBILITY_NODES",
    "build_android_runtime_candidates",
    "filter_android_accessibility_nodes",
    "is_forbidden_android_history_field",
    "is_forbidden_android_screen_field",
    "project_android_runtime_record",
]
