"""CanonicalAction → AndroidWorld JSONAction bridge."""
from __future__ import annotations

import time
from typing import Any

from guiaccel.types import BBox, CanonicalAction


def canonical_to_json_action(
    action: CanonicalAction,
    *,
    screen_width: int,
    screen_height: int,
) -> Any:
    """Map a CanonicalAction to an AndroidWorld JSONAction.

    android_world.env.json_action.JSONAction is imported lazily so this module
    can be loaded without android_world installed.
    """
    from android_world.env.json_action import JSONAction

    action_type = action.action_type

    if action_type == "CLICK":
        x, y = _bbox_center_to_pixel(action.bbox, screen_width, screen_height)
        return JSONAction(action_type="click", x=x, y=y)

    if action_type == "LONG_PRESS":
        x, y = _bbox_center_to_pixel(action.bbox, screen_width, screen_height)
        return JSONAction(action_type="long_press", x=x, y=y)

    if action_type == "TYPE":
        if action.bbox is not None:
            x, y = _bbox_center_to_pixel(action.bbox, screen_width, screen_height)
            return JSONAction(action_type="click", x=x, y=y)
        return JSONAction(action_type="input_text", text=action.argument or "")

    if action_type == "SCROLL":
        direction = _canonical_direction_to_aw(action.direction)
        if action.bbox is not None:
            x, y = _bbox_center_to_pixel(action.bbox, screen_width, screen_height)
            return JSONAction(action_type="scroll", direction=direction, x=x, y=y)
        return JSONAction(
            action_type="scroll",
            direction=direction,
            x=screen_width // 2,
            y=screen_height // 2,
        )

    if action_type == "NAV":
        token = (action.argument or "").lower()
        if token in ("back", ""):
            return JSONAction(action_type="navigate_back")
        if token == "home":
            return JSONAction(action_type="navigate_home")
        if token == "enter":
            return JSONAction(action_type="keyboard_enter")
        app_name = action.app or action.argument or ""
        return JSONAction(action_type="open_app", app_name=app_name)

    if action_type == "WAIT":
        return JSONAction(action_type="wait")

    if action_type == "TERMINATE":
        status = (action.argument or "").lower()
        goal_status = "complete" if status in ("success", "complete", "") else "infeasible"
        return JSONAction(action_type="status", goal_status=goal_status)

    raise ValueError(f"Unmapped CanonicalAction type: {action_type!r}")


def _execute_type_action(
    env: Any,
    action: CanonicalAction,
    screen_width: int,
    screen_height: int,
) -> None:
    """Execute a TYPE action: click to focus target element, then input_text.

    AndroidWorld's input_text action does not accept coordinates; it types into
    the currently focused element. When the canonical action carries a bbox, we
    first click to move focus, then type.
    """
    from android_world.env.json_action import JSONAction

    if action.bbox is not None:
        x, y = _bbox_center_to_pixel(action.bbox, screen_width, screen_height)
        env.execute_action(JSONAction(action_type="click", x=x, y=y))
        time.sleep(0.3)
    env.execute_action(JSONAction(action_type="input_text", text=action.argument or ""))


def _bbox_center_to_pixel(
    bbox: BBox | None,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Return the pixel-coordinate center of a CanonicalAction bbox.

    CanonicalAction.bbox already holds absolute pixel coordinates after
    parse_model_action_output has applied the 0-999 → pixel scaling.
    This function extracts the center and clamps it to screen bounds.
    """
    if bbox is None:
        return screen_width // 2, screen_height // 2
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cx = max(0, min(cx, screen_width - 1))
    cy = max(0, min(cy, screen_height - 1))
    return cx, cy


def _canonical_direction_to_aw(direction: str | None) -> str:
    mapping = {"up": "up", "down": "down", "left": "left", "right": "right"}
    return mapping.get((direction or "").lower(), "down")


__all__ = [
    "canonical_to_json_action",
    "_bbox_center_to_pixel",
    "_canonical_direction_to_aw",
    "_execute_type_action",
]
