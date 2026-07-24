"""Decode-side evaluation helpers for the coordinate regression head.

Compares two paths on each CLICK / LONG_PRESS step:

1. **AR decode** — full ``model.generate()`` (no hidden-state retention),
   parse mobile_use JSON → coordinates.
2. **Regression** — GT-Forcing Prefill → CoordHead → coordinates.

Both are scored against ground-truth 0–999 coordinates and optional GT bbox hit.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from guiaccel.model.hidden_state_extractor import (
    COORD_ACTION_TYPES,
    DEFAULT_EXTRACT_POINT,
    DEFAULT_THINKING_MODE,
    SCALE_FACTOR,
    _build_messages,
    _prepare_image,
    build_gt_prefix,  # shared with extract for train/eval prefix consistency
    extract_hidden_state,
)
from guiaccel.model.maiui_action_adapter import parse_model_action_output
from guiaccel.types import DatasetStep

# Re-export for callers that want the shared builder without importing extractor.
__all__ = (
    "autoregressive_decode_coords",
    "regression_predict_coords",
    "mae_999",
    "build_gt_prefix",
)


def _gt_coord_from_step(step: DatasetStep) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    """Return ((x999, y999), (x_abs, y_abs), sw, sh) or None."""
    raw = step.raw_action
    action_type = str(raw.get("action_type", "")).lower()
    if action_type not in COORD_ACTION_TYPES:
        return None

    gt_bbox = raw.get("target_bbox") or raw.get("bbox")
    if isinstance(gt_bbox, (list, tuple)) and len(gt_bbox) == 4:
        x_abs = (gt_bbox[0] + gt_bbox[2]) / 2.0
        y_abs = (gt_bbox[1] + gt_bbox[3]) / 2.0
    elif "x" in raw and "y" in raw:
        x_abs = float(raw["x"])
        y_abs = float(raw["y"])
    else:
        return None

    sw = int(step.metadata.get("screenshot_width", 1080))
    sh = int(step.metadata.get("screenshot_height", 1920))
    x999 = int(round(x_abs / max(sw, 1) * SCALE_FACTOR))
    y999 = int(round(y_abs / max(sh, 1) * SCALE_FACTOR))
    return (x999, y999), (int(round(x_abs)), int(round(y_abs))), sw, sh


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> bool | None:
    if bbox is None or len(bbox) != 4:
        return None
    x, y = point
    return bool(bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3])


def _bbox_center_to_999(
    bbox: tuple[int, int, int, int],
    *,
    sw: int,
    sh: int,
) -> tuple[int, int]:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return (
        int(round(cx / max(sw, 1) * SCALE_FACTOR)),
        int(round(cy / max(sh, 1) * SCALE_FACTOR)),
    )


def autoregressive_decode_coords(
    model: Any,
    processor: Any,
    step: DatasetStep,
    *,
    max_new_tokens: int = 512,
    max_pixels: int = 1_000_000,
) -> dict[str, Any]:
    """Full AR generate → parse click/long_press coordinates.

    Does **not** request ``output_hidden_states`` (avoids OOM).
    """
    gt = _gt_coord_from_step(step)
    if gt is None:
        return {"ok": False, "reason": "non_coord_or_missing_gt"}

    gt_999, gt_abs, sw, sh = gt
    image = _prepare_image(step.screenshot, max_pixels=max_pixels)
    messages = _build_messages(step, image)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"truncation": True},
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_len = int(inputs["input_ids"].shape[-1])

    t0 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,
        )
    ar_ms = (time.perf_counter() - t0) * 1000.0

    new_ids = generated[0, input_len:]
    gen_tokens = int(new_ids.shape[0])
    output_text = processor.batch_decode(
        [new_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    ar_999: tuple[int, int] | None = None
    ar_abs: tuple[int, int] | None = None
    parse_ok = False
    parse_error: str | None = None
    try:
        action = parse_model_action_output(
            output_text,
            screenshot_width=sw,
            screenshot_height=sh,
            assume_android_coordinate_scale=True,
        )
        if action.bbox is not None:
            ar_abs = (
                int(round((action.bbox[0] + action.bbox[2]) / 2.0)),
                int(round((action.bbox[1] + action.bbox[3]) / 2.0)),
            )
            ar_999 = _bbox_center_to_999(action.bbox, sw=sw, sh=sh)
            parse_ok = True
        else:
            parse_error = "parsed_action_missing_bbox"
    except Exception as exc:  # noqa: BLE001 — record and continue
        parse_error = f"{type(exc).__name__}: {exc}"

    target_bbox = step.metadata.get("action_target_bbox")
    if isinstance(target_bbox, (list, tuple)) and len(target_bbox) == 4:
        target_bbox_t = tuple(int(v) for v in target_bbox)
    else:
        target_bbox_t = None

    return {
        "ok": True,
        "episode_id": step.episode_id,
        "step_index": step.step_index,
        "action_type": str(step.raw_action.get("action_type", "")).lower(),
        "gt_999": list(gt_999),
        "gt_abs": list(gt_abs),
        "screenshot_width": sw,
        "screenshot_height": sh,
        "ar_999": list(ar_999) if ar_999 is not None else None,
        "ar_abs": list(ar_abs) if ar_abs is not None else None,
        "ar_parse_ok": parse_ok,
        "ar_parse_error": parse_error,
        "ar_gen_tokens": gen_tokens,
        "ar_ms": round(ar_ms, 1),
        "ar_hit_bbox": _point_in_bbox(ar_abs, target_bbox_t) if ar_abs is not None else None,
        "raw_output": output_text[:2000],
        "target_bbox": list(target_bbox_t) if target_bbox_t is not None else None,
    }


def regression_predict_coords(
    model: Any,
    processor: Any,
    coord_head: Any,
    step: DatasetStep,
    *,
    layer: int = -1,
    max_pixels: int = 1_000_000,
    device: Any = None,
    thinking_mode: str = DEFAULT_THINKING_MODE,
    extract_point: str = DEFAULT_EXTRACT_POINT,
) -> dict[str, Any]:
    """GT-Forcing Prefill + CoordHead → 0–999 coordinates.

    Uses the same :func:`build_gt_prefix` / ``extract_hidden_state`` path as
    training extract so train/eval prefixes stay aligned.

    Note: with ``thinking_mode=template``, extract latency is *not* fair deploy
    latency — fair eval must still measure AR thinking decode separately (T4).
    """
    t0 = time.perf_counter()
    sample = extract_hidden_state(
        model, processor, step,
        layer=layer,
        max_pixels=max_pixels,
        thinking_mode=thinking_mode,
        extract_point=extract_point,
    )
    extract_ms = (time.perf_counter() - t0) * 1000.0
    if sample is None:
        return {"ok": False, "reason": "extract_failed"}

    if device is None:
        device = next(coord_head.parameters()).device
    h = sample.hidden_state.to(device).unsqueeze(0)

    t1 = time.perf_counter()
    with torch.no_grad():
        pred_999_t = coord_head.predict_999(h)[0].cpu()
    head_ms = (time.perf_counter() - t1) * 1000.0
    pred_999 = (int(pred_999_t[0]), int(pred_999_t[1]))

    sw, sh = sample.screenshot_width, sample.screenshot_height
    pred_abs = (
        int(round(pred_999[0] / SCALE_FACTOR * sw)),
        int(round(pred_999[1] / SCALE_FACTOR * sh)),
    )
    target_bbox = step.metadata.get("action_target_bbox")
    if isinstance(target_bbox, (list, tuple)) and len(target_bbox) == 4:
        target_bbox_t = tuple(int(v) for v in target_bbox)
    else:
        target_bbox_t = None

    return {
        "ok": True,
        "reg_999": list(pred_999),
        "reg_abs": list(pred_abs),
        "reg_extract_ms": round(extract_ms, 1),
        "reg_head_ms": round(head_ms, 3),
        "reg_ms": round(extract_ms + head_ms, 1),
        "reg_prefix_tokens": sample.generated_tokens,
        "reg_thinking_mode": sample.thinking_mode,
        "reg_extract_point": sample.extract_point,
        "reg_hit_bbox": _point_in_bbox(pred_abs, target_bbox_t),
        "gt_999": list(sample.gt_coord_999),
    }


def mae_999(a: list[int] | tuple[int, int], b: list[int] | tuple[int, int]) -> float:
    return (abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))) / 2.0
