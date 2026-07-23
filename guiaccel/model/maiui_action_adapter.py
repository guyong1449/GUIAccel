"""MAI-UI baseline output adapter."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from guiaccel.types import BBox, CanonicalAction

SCALE_FACTOR = 999
_ACTION_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", flags=re.DOTALL | re.IGNORECASE)
_LEGACY_ACTION_CALL_RE = re.compile(
    r"(?P<action_type>CLICK|TYPE|SCROLL|NAV|TERMINATE|WAIT|LONG_PRESS)"
    r"(?:\[(?P<payload>.*)\])?$",
    flags=re.IGNORECASE,
)
_BROKEN_XY_RE = re.compile(
    r'("x"\s*:\s*-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(\s*[},])',
    flags=re.IGNORECASE,
)


def parse_model_action_output(
    output_text: str,
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None = None,
    assume_android_coordinate_scale: bool = False,
) -> CanonicalAction:
    """Parse MAI-UI or legacy benchmark output into a canonical action."""

    for payload, prefer_scaled_coordinates in _iter_payload_candidates(
        output_text,
        assume_android_coordinate_scale=assume_android_coordinate_scale,
    ):
        action = _payload_to_canonical_action(
            payload,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            roi_bbox=roi_bbox,
            prefer_scaled_coordinates=prefer_scaled_coordinates,
        )
        if action is not None and _is_action_semantically_complete(action):
            return action

    stripped = output_text.strip()
    legacy_match = _LEGACY_ACTION_CALL_RE.match(stripped)
    if legacy_match is not None:
        action = _action_from_legacy_call(
            legacy_match.group("action_type"),
            legacy_match.group("payload"),
        )
        if _is_action_semantically_complete(action):
            return action

    heuristic_action = _heuristic_action_from_text(
        output_text,
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        roi_bbox=roi_bbox,
    )
    if heuristic_action is not None and _is_action_semantically_complete(heuristic_action):
        return heuristic_action
    raise ValueError(f"Could not parse model output into CanonicalAction: {output_text!r}")


def _iter_payload_candidates(
    output_text: str,
    *,
    assume_android_coordinate_scale: bool,
) -> tuple[tuple[Mapping[str, Any], bool], ...]:
    candidates: list[tuple[Mapping[str, Any], bool]] = []
    seen: set[str] = set()

    for match in _TOOL_CALL_RE.finditer(output_text):
        payload = _extract_json_mapping(match.group("body"))
        if payload is None:
            continue
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        if serialized in seen:
            continue
        seen.add(serialized)
        candidates.append((payload, True))

    payload = _extract_json_mapping(output_text)
    if payload is not None:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        if serialized not in seen:
            candidates.append((payload, bool(assume_android_coordinate_scale)))

    return tuple(candidates)


def _extract_json_mapping(output_text: str) -> dict[str, Any] | None:
    stripped = output_text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    match = _ACTION_JSON_RE.search(stripped)
    if match is not None:
        candidates.append(match.group(0))
    repaired = _repair_truncated_json_object(stripped)
    if repaired is not None:
        candidates.append(repaired)
    for candidate in candidates:
        normalized = _repair_common_maiui_json_errors(candidate)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _repair_common_maiui_json_errors(candidate: str) -> str:
    normalized = candidate.strip().strip("`").strip()
    normalized = _BROKEN_XY_RE.sub(r'\1, "y": \2\3', normalized)
    return normalized


def _repair_truncated_json_object(output_text: str) -> str | None:
    start = output_text.find("{")
    if start < 0:
        return None
    candidate = output_text[start:].strip().strip("`").strip()
    if not candidate.startswith("{"):
        return None
    max_trim = min(64, max(0, len(candidate) - 1))
    for trim in range(max_trim + 1):
        trial = candidate[:-trim] if trim else candidate
        normalized = _normalize_partial_json(trial)
        if normalized is None:
            continue
        normalized = _repair_common_maiui_json_errors(normalized)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return normalized
    return None


def _normalize_partial_json(candidate: str) -> str | None:
    normalized = candidate.rstrip()
    while normalized and normalized[-1] in ",:{[":
        normalized = normalized[:-1].rstrip()
    if normalized.count('"') % 2 == 1:
        normalized = normalized.rsplit('"', 1)[0].rstrip()
        while normalized and normalized[-1] in ",:{[":
            normalized = normalized[:-1].rstrip()
    if not normalized.startswith("{"):
        return None
    missing_brackets = normalized.count("[") - normalized.count("]")
    missing_braces = normalized.count("{") - normalized.count("}")
    if missing_brackets < 0 or missing_braces < 0:
        return None
    return normalized + ("]" * missing_brackets) + ("}" * missing_braces)


def _payload_to_canonical_action(
    payload: Mapping[str, Any],
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
    prefer_scaled_coordinates: bool,
) -> CanonicalAction | None:
    action_payload, inferred_scaled_coordinates = _unwrap_action_payload(payload)
    action_name = _extract_action_name(action_payload)
    if action_name is None:
        return None

    normalized_action_name = _normalize_action_name(action_name)
    scale_coordinates = prefer_scaled_coordinates or inferred_scaled_coordinates
    app = _coerce_optional_text(
        _first_scalar_value(
            action_payload,
            "app",
            "app_name",
            "package_name",
        )
    )

    if normalized_action_name in {"CLICK", "LONG_PRESS"}:
        bbox = _extract_bbox(
            action_payload,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            roi_bbox=roi_bbox,
            scale_coordinates=scale_coordinates,
        )
        if bbox is None:
            point = _extract_point(
                action_payload,
                screenshot_width=screenshot_width,
                screenshot_height=screenshot_height,
                roi_bbox=roi_bbox,
                scale_coordinates=scale_coordinates,
            )
            if point is not None:
                bbox = _point_bbox_in_frame(
                    *point,
                    screenshot_width=screenshot_width,
                    screenshot_height=screenshot_height,
                    roi_bbox=roi_bbox,
                )
        return CanonicalAction(normalized_action_name, None, bbox, app, None)

    if normalized_action_name == "TYPE":
        argument = _extract_text_argument(action_payload)
        bbox = _extract_bbox(
            action_payload,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            roi_bbox=roi_bbox,
            scale_coordinates=scale_coordinates,
        )
        if bbox is None:
            point = _extract_point(
                action_payload,
                screenshot_width=screenshot_width,
                screenshot_height=screenshot_height,
                roi_bbox=roi_bbox,
                scale_coordinates=scale_coordinates,
            )
            if point is not None:
                bbox = _point_bbox_in_frame(
                    *point,
                    screenshot_width=screenshot_width,
                    screenshot_height=screenshot_height,
                    roi_bbox=roi_bbox,
                )
        return CanonicalAction("TYPE", argument, bbox, app, None)

    if normalized_action_name == "SCROLL":
        direction = _extract_scroll_direction(action_payload)
        bbox = _extract_bbox(
            action_payload,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            roi_bbox=roi_bbox,
            scale_coordinates=scale_coordinates,
        )
        return CanonicalAction("SCROLL", None, bbox, app, direction)

    if normalized_action_name == "NAV":
        token = _extract_nav_target(action_payload, action_name=action_name, default_app=app)
        app_name = token if token not in {"back", "home", "enter"} else None
        if action_name.lower() in {"open", "open_app", "launch"} and token is not None:
            app_name = token
        return CanonicalAction("NAV", token, None, app_name, None)

    if normalized_action_name == "WAIT":
        return CanonicalAction("WAIT", _extract_wait_argument(action_payload), None, app, None)

    if normalized_action_name == "TERMINATE":
        return CanonicalAction("TERMINATE", _extract_terminate_argument(action_payload), None, app, None)

    return None


def _unwrap_action_payload(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    if _normalize_text(payload.get("name")) == "mobile_use":
        arguments = _coerce_mapping(payload.get("arguments"))
        if arguments is not None:
            return arguments, True
    arguments = _coerce_mapping(payload.get("arguments"))
    if arguments is not None and _extract_action_name(arguments) is not None:
        return arguments, True
    if _extract_action_name(payload) is None:
        argument_payload = _coerce_mapping(payload.get("argument"))
        if argument_payload is not None and _extract_action_name(argument_payload) is not None:
            return argument_payload, True
    nested_argument = _coerce_mapping(payload.get("argument"))
    nested_has_coordinates = nested_argument is not None and (
        _coerce_bbox_like(nested_argument.get("bbox")) is not None
        or _extract_raw_point(nested_argument) is not None
    )
    action_name = _extract_action_name(payload)
    if action_name is not None and nested_has_coordinates:
        return payload, True
    return payload, False


def _extract_action_name(payload: Mapping[str, Any]) -> str | None:
    for key in ("action", "action_type", "operation", "name"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and normalized.lower() != "mobile_use":
            return normalized
    return None


def _normalize_action_name(action_name: str) -> str | None:
    normalized = _normalize_text(action_name)
    aliases = {
        "click": "CLICK",
        "tap": "CLICK",
        "press": "CLICK",
        "long_press": "LONG_PRESS",
        "longpress": "LONG_PRESS",
        "hold": "LONG_PRESS",
        "type": "TYPE",
        "input_text": "TYPE",
        "input": "TYPE",
        "text": "TYPE",
        "scroll": "SCROLL",
        "swipe": "SCROLL",
        "drag": "SCROLL",
        "open": "NAV",
        "open_app": "NAV",
        "launch": "NAV",
        "nav": "NAV",
        "system_button": "NAV",
        # mobile_use / MAI-UI specific action names
        "navigate_back": "NAV",
        "navigate_home": "NAV",
        "navigate_recent": "NAV",
        "go_back": "NAV",
        "go_home": "NAV",
        "back": "NAV",
        "home": "NAV",
        "wait": "WAIT",
        "terminate": "TERMINATE",
        "answer": "TERMINATE",
        "status": "TERMINATE",
        "finish_episode": "TERMINATE",
        "finish": "TERMINATE",
        "complete": "TERMINATE",
        "done": "TERMINATE",
        "stop": "TERMINATE",
        # AndroidWorld-specific action names
        "press_enter": "NAV",
        "keyboard_enter": "NAV",
        "double_tap": "CLICK",
    }
    return aliases.get(normalized)


def _extract_bbox(
    payload: Mapping[str, Any],
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
    scale_coordinates: bool,
) -> BBox | None:
    for value in (
        payload.get("bbox"),
        _nested_mapping_value(payload.get("argument"), "bbox"),
        payload.get("bounds"),
        _nested_mapping_value(payload.get("argument"), "bounds"),
    ):
        bbox = _coerce_bbox_like(value)
        if bbox is None:
            continue
        return _maybe_scale_bbox(
            bbox,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            roi_bbox=roi_bbox,
            scale_coordinates=scale_coordinates,
        )
    return None


def _extract_point(
    payload: Mapping[str, Any],
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
    scale_coordinates: bool,
) -> tuple[int, int] | None:
    point = _extract_raw_point(payload)
    if point is None:
        nested_argument = _coerce_mapping(payload.get("argument"))
        if nested_argument is not None:
            point = _extract_raw_point(nested_argument)
    if point is None:
        return None
    return _maybe_scale_point(
        point,
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        roi_bbox=roi_bbox,
        scale_coordinates=scale_coordinates,
    )


def _extract_raw_point(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    if _is_number(payload.get("x")) and _is_number(payload.get("y")):
        return (float(payload["x"]), float(payload["y"]))
    # "coordinate": [x, y]  — mobile_use / MAI-UI format
    for key in ("coordinate", "point", "position", "center"):
        seq = payload.get(key)
        if isinstance(seq, (list, tuple)) and len(seq) == 2 and all(_is_number(v) for v in seq):
            return (float(seq[0]), float(seq[1]))
    return None


def _extract_scroll_direction(payload: Mapping[str, Any]) -> str | None:
    direct = _coerce_optional_text(
        _first_scalar_value(
            payload,
            "direction",
        )
    )
    if direct is not None:
        return _normalize_text(direct)

    nested_argument = _coerce_mapping(payload.get("argument"))
    if nested_argument is not None:
        nested_direction = _coerce_optional_text(_first_scalar_value(nested_argument, "direction"))
        if nested_direction is not None:
            return _normalize_text(nested_direction)

    motion = _extract_motion_points(payload)
    if motion is None:
        return None
    start_x, start_y, end_x, end_y = motion
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    if abs(delta_y) >= abs(delta_x):
        return "down" if delta_y > 0 else "up"
    return "right" if delta_x > 0 else "left"


def _extract_motion_points(payload: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    if all(_is_number(payload.get(key)) for key in ("start_x", "start_y", "end_x", "end_y")):
        return (
            float(payload["start_x"]),
            float(payload["start_y"]),
            float(payload["end_x"]),
            float(payload["end_y"]),
        )
    if all(_is_number(payload.get(key)) for key in ("from_x", "from_y", "to_x", "to_y")):
        return (
            float(payload["from_x"]),
            float(payload["from_y"]),
            float(payload["to_x"]),
            float(payload["to_y"]),
        )
    start = _coerce_mapping(payload.get("start")) or _coerce_mapping(payload.get("from"))
    end = _coerce_mapping(payload.get("end")) or _coerce_mapping(payload.get("to"))
    if start is not None and end is not None:
        start_point = _extract_raw_point(start)
        end_point = _extract_raw_point(end)
        if start_point is not None and end_point is not None:
            return (start_point[0], start_point[1], end_point[0], end_point[1])
    return None


def _extract_text_argument(payload: Mapping[str, Any]) -> str | None:
    text_value = _first_scalar_value(payload, "text", "value", "content", "answer")
    if text_value is not None:
        return _coerce_optional_text(text_value)
    argument = payload.get("argument")
    if isinstance(argument, str):
        return _coerce_optional_text(argument)
    nested_argument = _coerce_mapping(argument)
    if nested_argument is not None:
        nested_text = _first_scalar_value(nested_argument, "text", "value", "content", "answer")
        if nested_text is not None:
            return _coerce_optional_text(nested_text)
    return None


def _extract_nav_target(
    payload: Mapping[str, Any],
    *,
    action_name: str,
    default_app: str | None,
) -> str | None:
    normalized_action_name = _normalize_text(action_name)
    # mobile_use action names carry their target in the name itself
    _mobile_use_nav_tokens = {
        "navigate_back": "back",
        "go_back": "back",
        "back": "back",
        "navigate_home": "home",
        "go_home": "home",
        "home": "home",
        "navigate_recent": "recent",
        "press_enter": "enter",
        "keyboard_enter": "enter",
    }
    if normalized_action_name in _mobile_use_nav_tokens:
        return _mobile_use_nav_tokens[normalized_action_name]
    if normalized_action_name == "system_button":
        token = _first_scalar_value(payload, "button", "argument", "target", "name")
        return _normalize_text(token) if token is not None else None
    token = _first_scalar_value(payload, "app", "app_name", "package_name", "target")
    if token is None:
        argument = payload.get("argument")
        if isinstance(argument, str):
            token = argument
    if token is None:
        token = default_app
    return _coerce_optional_text(token)


def _extract_wait_argument(payload: Mapping[str, Any]) -> str | None:
    for key in ("seconds", "duration", "value"):
        value = payload.get(key)
        if value is not None and not isinstance(value, Mapping):
            return _coerce_optional_text(value)
    argument = payload.get("argument")
    if isinstance(argument, str):
        return _coerce_optional_text(argument)
    nested_argument = _coerce_mapping(argument)
    if nested_argument is not None:
        for key in ("seconds", "duration", "value"):
            value = nested_argument.get(key)
            if value is not None and not isinstance(value, Mapping):
                return _coerce_optional_text(value)
    return None


def _extract_terminate_argument(payload: Mapping[str, Any]) -> str | None:
    value = _first_nested_scalar_value(
        payload,
        "goal_status",
        "task_status",
        "final_status",
        "status",
        "text",
        "content",
        "answer",
        "result",
        "outcome",
    )
    if value is not None:
        return _coerce_optional_text(value)
    argument = payload.get("argument")
    if isinstance(argument, str):
        return _coerce_optional_text(argument)
    return None


def _nested_mapping_value(value: Any, key: str) -> Any:
    mapping = _coerce_mapping(value)
    if mapping is None:
        return None
    return mapping.get(key)


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(_repair_common_maiui_json_errors(value))
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _coerce_bbox_like(value: Any) -> BBox | None:
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(_is_number(item) for item in value):
        left, top, right, bottom = (int(round(float(item))) for item in value)
        ordered_left, ordered_right = sorted((left, right))
        ordered_top, ordered_bottom = sorted((top, bottom))
        return (ordered_left, ordered_top, ordered_right, ordered_bottom)
    return None


def _maybe_scale_point(
    point: tuple[float, float],
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
    scale_coordinates: bool,
) -> tuple[int, int]:
    if not scale_coordinates or not _looks_scaled_coordinates(point):
        return (int(round(point[0])), int(round(point[1])))
    offset_x, offset_y, width, height = _coordinate_frame(
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        roi_bbox=roi_bbox,
    )
    if width is None or height is None:
        return (int(round(point[0])), int(round(point[1])))
    return (
        offset_x + _scale_coordinate(point[0], width),
        offset_y + _scale_coordinate(point[1], height),
    )


def _maybe_scale_bbox(
    bbox: BBox,
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
    scale_coordinates: bool,
) -> BBox:
    if not scale_coordinates or not _looks_scaled_coordinates(bbox):
        return bbox
    offset_x, offset_y, width, height = _coordinate_frame(
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        roi_bbox=roi_bbox,
    )
    if width is None or height is None:
        return bbox
    left, top, right, bottom = bbox
    scaled = (
        offset_x + _scale_coordinate(left, width),
        offset_y + _scale_coordinate(top, height),
        offset_x + _scale_coordinate(right, width),
        offset_y + _scale_coordinate(bottom, height),
    )
    return _coerce_bbox_like(scaled) or scaled


def _coordinate_frame(
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
) -> tuple[int, int, int | None, int | None]:
    if roi_bbox is not None:
        left, top, right, bottom = (int(value) for value in roi_bbox)
        return (left, top, max(1, right - left), max(1, bottom - top))
    width = max(1, int(screenshot_width)) if screenshot_width is not None else None
    height = max(1, int(screenshot_height)) if screenshot_height is not None else None
    return (0, 0, width, height)


def _scale_coordinate(value: float, size: int) -> int:
    if size <= 1:
        return 0
    upper_bound = float(size - 1)
    scaled = float(value) * upper_bound / float(SCALE_FACTOR)
    return int(round(max(0.0, min(upper_bound, scaled))))


def _point_bbox(x: int, y: int, padding: int = 1) -> BBox:
    return (x - padding, y - padding, x + padding, y + padding)


def _point_bbox_in_frame(
    x: int,
    y: int,
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
    padding: int = 1,
) -> BBox:
    offset_x, offset_y, width, height = _coordinate_frame(
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        roi_bbox=roi_bbox,
    )
    if width is None or height is None:
        return _point_bbox(x, y, padding=padding)
    max_x = offset_x + width - 1
    max_y = offset_y + height - 1
    return (
        max(offset_x, x - padding),
        max(offset_y, y - padding),
        min(max_x, x + padding),
        min(max_y, y + padding),
    )


def _looks_scaled_coordinates(values: tuple[float, ...] | BBox) -> bool:
    return all(0.0 <= float(value) <= float(SCALE_FACTOR) for value in values)


def _heuristic_action_from_text(
    output_text: str,
    *,
    screenshot_width: int | None,
    screenshot_height: int | None,
    roi_bbox: BBox | None,
) -> CanonicalAction | None:
    action_match = re.search(
        r'"(?:action|action_type)"\s*:\s*"(?P<action>[A-Za-z_]+)"',
        output_text,
        flags=re.IGNORECASE,
    )
    if action_match is None:
        return None
    action_name = action_match.group("action")
    normalized_action_name = _normalize_action_name(action_name)
    if normalized_action_name not in {"CLICK", "LONG_PRESS", "SCROLL"}:
        return None
    if normalized_action_name in {"CLICK", "LONG_PRESS"}:
        bbox_match = re.search(
            r'"bbox"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            output_text,
            flags=re.IGNORECASE,
        )
        if bbox_match is not None:
            bbox = tuple(int(round(float(value))) for value in bbox_match.groups())
            scaled_bbox = _maybe_scale_bbox(
                bbox,
                screenshot_width=screenshot_width,
                screenshot_height=screenshot_height,
                roi_bbox=roi_bbox,
                scale_coordinates=True,
            )
            return CanonicalAction(normalized_action_name, None, scaled_bbox, None, None)
        point_match = re.search(
            r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(?:"y"\s*:\s*)?(-?\d+(?:\.\d+)?)',
            output_text,
            flags=re.IGNORECASE,
        )
        if point_match is None:
            return None
        point = (float(point_match.group(1)), float(point_match.group(2)))
        scaled_point = _maybe_scale_point(
            point,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            roi_bbox=roi_bbox,
            scale_coordinates=True,
        )
        return CanonicalAction(
            normalized_action_name,
            None,
            _point_bbox_in_frame(
                *scaled_point,
                screenshot_width=screenshot_width,
                screenshot_height=screenshot_height,
                roi_bbox=roi_bbox,
            ),
            None,
            None,
        )
    direction_match = re.search(r'"direction"\s*:\s*"(?P<direction>[A-Za-z_]+)"', output_text, flags=re.IGNORECASE)
    if direction_match is None:
        return None
    return CanonicalAction("SCROLL", None, None, None, _normalize_text(direction_match.group("direction")))


def _action_from_legacy_call(action_type: str, payload: str | None) -> CanonicalAction:
    normalized_action_type = str(action_type).upper()
    normalized_payload = payload.strip() if payload is not None else ""
    bbox = None
    argument = None
    direction = None
    if normalized_payload:
        if "," in normalized_payload:
            parts = [part.strip() for part in normalized_payload.split(",")]
            if len(parts) == 4 and all(_looks_numeric(part) for part in parts):
                bbox = tuple(int(float(part)) for part in parts)
            elif len(parts) == 1:
                argument = parts[0]
        elif normalized_action_type == "SCROLL":
            direction = normalized_payload.lower()
        else:
            argument = normalized_payload
    if normalized_action_type == "SCROLL" and argument is not None and direction is None:
        direction = argument.lower()
        argument = None
    return CanonicalAction(
        action_type=normalized_action_type,
        argument=_coerce_optional_text(argument),
        bbox=_coerce_bbox_like(bbox),
        app=None,
        direction=_coerce_optional_text(direction),
    )


def _first_scalar_value(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is None or isinstance(value, Mapping):
            continue
        return value
    return None


def _first_nested_scalar_value(payload: Mapping[str, Any], *keys: str) -> Any:
    direct_value = _first_scalar_value(payload, *keys)
    if direct_value is not None:
        return direct_value
    for value in payload.values():
        if isinstance(value, Mapping):
            nested_value = _first_nested_scalar_value(value, *keys)
            if nested_value is not None:
                return nested_value
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                nested_value = _first_nested_scalar_value(item, *keys)
                if nested_value is not None:
                    return nested_value
    return None


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_action_semantically_complete(action: CanonicalAction) -> bool:
    action_type = str(action.action_type or "").upper()
    if action_type in {"CLICK", "LONG_PRESS"}:
        return action.bbox is not None
    if action_type == "TYPE":
        return action.argument is not None
    if action_type == "SCROLL":
        return action.direction is not None
    if action_type == "NAV":
        normalized = str(action.argument or "").strip().lower()
        if normalized in {"back", "home", "enter"}:
            return True
        return bool(action.app or action.argument)
    if action_type in {"WAIT", "TERMINATE"}:
        return True
    return False


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
