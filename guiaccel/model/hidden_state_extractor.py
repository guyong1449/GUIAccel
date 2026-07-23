"""Extract hidden states from Qwen3-VL at the action-type token position.

Usage flow:
    1. Load Qwen3-VL + processor via ``load_model_for_extraction``.
    2. For each CLICK / LONG_PRESS step, call ``extract_hidden_state``
       to get the last-layer hidden vector at the trigger token.
    3. Pair it with the ground-truth normalised coordinate and save.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

# Qwen3-VL mobile_use system prompt (must match service_backend)
from guiaccel.model.qwen_backend import (
    ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT,
    QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
    _load_image,
    _pad_image_to_safe_aspect_ratio,
)
from guiaccel.types import DatasetStep, ScreenshotAsset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COORD_ACTION_TYPES = frozenset({"click", "long_press"})
SCALE_FACTOR = 999

# Tokens that signal "I'm about to emit coordinates next" in Qwen3-VL output.
# We search the generated token-id sequence for these keywords.
_ACTION_KEYWORD_RE = re.compile(
    r'"action"\s*:\s*"(?P<action>click|long_press)"',
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSample:
    """One (hidden_state, ground_truth) pair for regression head training."""

    episode_id: str
    step_index: int
    action_type: str
    hidden_state: torch.Tensor          # shape [hidden_dim]
    gt_coord_norm: torch.Tensor         # shape [2], in [0, 1]
    gt_coord_999: tuple[int, int]       # 0-999 scale
    screenshot_width: int
    screenshot_height: int
    generated_tokens: int
    extraction_layer: int               # which transformer layer (-1 = last)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_extraction(
    model_path: str | Path,
    *,
    device: str | int = 0,
    dtype: str = "bfloat16",
) -> tuple[Any, Any, Any]:
    """Load Qwen3-VL model + processor for hidden-state extraction.

    Returns ``(model, processor, torch_module)`` where ``torch_module``
    is the ``torch`` package reference.
    """
    import torch as _torch
    from PIL import Image as _Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype_map = {
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
    }
    resolved_dtype = dtype_map.get(dtype, _torch.bfloat16)
    resolved_path = str(Path(model_path).resolve())

    processor = AutoProcessor.from_pretrained(resolved_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved_path,
        device_map=device,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor, _torch


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _build_messages(
    step: DatasetStep,
    image: Any,
) -> list[Mapping[str, Any]]:
    """Build chat messages for a single AndroidControl step (same format as qwen_backend)."""
    goal = step.goal
    step_instruction = step.metadata.get("step_instruction") or ""
    history_types = step.metadata.get("history_action_types", ())

    parts = [f"Task goal: {goal}"]
    if step_instruction:
        parts.append(f"Instruction for the current step: {step_instruction}")
    if history_types:
        history_lines = [f"{at.lower()}: (done)" for at in history_types]
        parts.append("Actions already performed:\n" + "\n".join(history_lines))
    prompt_text = "\n".join(parts)

    user_text = f"{prompt_text}\nOutput exactly one next action (mobile_use) for the current screenshot."
    return [
        {"role": "system", "content": [{"type": "text", "text": ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _prepare_image(
    screenshot: ScreenshotAsset,
    max_pixels: int = 1_000_000,
) -> Any:
    """Load and preprocess a screenshot exactly as the qwen backend does."""
    from PIL import Image as _Image

    image = _load_image(screenshot, _Image)
    image = _pad_image_to_safe_aspect_ratio(
        image,
        image_module=_Image,
        max_aspect_ratio=QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
    )
    width, height = image.size
    if width * height > max_pixels:
        scale = (float(max_pixels) / float(width * height)) ** 0.5
        image = image.resize((
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ))
    return image


def _find_action_token_position(
    generated_ids: torch.Tensor,
    tokenizer: Any,
    action_type: str = "click",
) -> int | None:
    """Find the token position where the model committed to the action type.

    Strategy: decode the generated tokens, search for the *last* occurrence
    of ``"action": "click"`` (or ``long_press``), then map that character
    offset back to a token index.
    """
    if generated_ids.dim() == 2:
        generated_ids = generated_ids[0]

    token_count = int(generated_ids.shape[0])
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Find the last occurrence of the action keyword
    target = f'"action": "{action_type}"'
    # Also handle without space
    alt_target = f'"action":"{action_type}"'
    
    last_pos = max(decoded_text.rfind(target), decoded_text.rfind(alt_target))
    if last_pos < 0:
        # Try more flexible regex
        match = None
        for m in _ACTION_KEYWORD_RE.finditer(decoded_text):
            if m.group("action").lower() == action_type.lower():
                match = m
        if match is None:
            return None
        last_pos = match.end() - 1

    # Map character offset to token index: decode token-by-token and track char positions
    char_count = 0
    for token_idx in range(token_count):
        single_token_text = tokenizer.decode(
            generated_ids[token_idx:token_idx + 1],
            skip_special_tokens=True,
        )
        char_count += len(single_token_text)
        if char_count > last_pos:
            return token_idx

    # Fallback: return the token right after the action type string ends
    # by searching for the action type name token ids
    action_token_ids = tokenizer.encode(action_type, add_special_tokens=False)
    if not action_token_ids:
        return None

    # Search backwards for the last occurrence of the action type token sequence
    gen_list = generated_ids.tolist()
    for start in range(len(gen_list) - len(action_token_ids), -1, -1):
        if gen_list[start:start + len(action_token_ids)] == action_token_ids:
            return start + len(action_token_ids) - 1

    return None


def extract_hidden_state(
    model: Any,
    processor: Any,
    step: DatasetStep,
    *,
    layer: int = -1,
    max_new_tokens: int = 512,
    max_pixels: int = 1_000_000,
) -> ExtractedSample | None:
    """Run Qwen3-VL on one step and extract the hidden state at the action trigger token.

    Returns ``None`` if the step is not a coordinate action or if the action
    token cannot be located in the output.
    """
    # Determine ground-truth action type and coordinates
    raw_action = step.raw_action
    action_type = str(raw_action.get("action_type", "")).lower()
    if action_type not in COORD_ACTION_TYPES:
        return None

    # Extract ground-truth coordinate (absolute pixels)
    gt_bbox = raw_action.get("target_bbox") or raw_action.get("bbox")
    gt_x_abs: float | None = None
    gt_y_abs: float | None = None
    if isinstance(gt_bbox, (list, tuple)) and len(gt_bbox) == 4:
        gt_x_abs = (gt_bbox[0] + gt_bbox[2]) / 2.0
        gt_y_abs = (gt_bbox[1] + gt_bbox[3]) / 2.0
    elif "x" in raw_action and "y" in raw_action:
        gt_x_abs = float(raw_action["x"])
        gt_y_abs = float(raw_action["y"])
    else:
        return None

    sw = int(step.metadata.get("screenshot_width", 1080))
    sh = int(step.metadata.get("screenshot_height", 1920))

    # Normalise to [0, 1]
    gt_x_norm = gt_x_abs / max(sw, 1)
    gt_y_norm = gt_y_abs / max(sh, 1)
    gt_x_999 = int(round(gt_x_norm * SCALE_FACTOR))
    gt_y_999 = int(round(gt_y_norm * SCALE_FACTOR))

    # Prepare image and messages
    image = _prepare_image(step.screenshot, max_pixels=max_pixels)
    messages = _build_messages(step, image)

    # Tokenise
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
    input_ids = inputs["input_ids"]
    input_length = int(input_ids.shape[-1])

    # Generate with hidden states
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0, input_length:]
    generated_count = int(generated_ids.shape[0])

    if generated_count == 0:
        return None

    # Find action token position in generated sequence
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    token_pos = _find_action_token_position(
        generated_ids, tokenizer, action_type=action_type,
    )
    if token_pos is None:
        return None

    # Extract hidden state from the specified layer at the action token position.
    # outputs.hidden_states is a tuple of (num_generated_tokens) entries.
    # Each entry is a tuple of (num_layers+1) tensors of shape (batch, 1, hidden_dim).
    # Entry at index `token_pos` corresponds to the decode step for that token.
    if token_pos >= len(outputs.hidden_states):
        token_pos = len(outputs.hidden_states) - 1

    step_hidden_states = outputs.hidden_states[token_pos]  # tuple of (num_layers+1) tensors
    h = step_hidden_states[layer]  # shape (batch, 1, hidden_dim)
    h = h[0, 0, :].detach().cpu().float()  # shape (hidden_dim,)

    return ExtractedSample(
        episode_id=step.episode_id,
        step_index=step.step_index,
        action_type=action_type,
        hidden_state=h,
        gt_coord_norm=torch.tensor([gt_x_norm, gt_y_norm], dtype=torch.float32),
        gt_coord_999=(gt_x_999, gt_y_999),
        screenshot_width=sw,
        screenshot_height=sh,
        generated_tokens=generated_count,
        extraction_layer=layer,
    )


# ---------------------------------------------------------------------------
# Batch persistence
# ---------------------------------------------------------------------------

def save_extracted_samples(
    samples: Sequence[ExtractedSample],
    path: Path,
) -> None:
    """Save a list of ExtractedSamples to a .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hidden_states": torch.stack([s.hidden_state for s in samples]),
        "gt_coords_norm": torch.stack([s.gt_coord_norm for s in samples]),
        "meta": [
            {
                "episode_id": s.episode_id,
                "step_index": s.step_index,
                "action_type": s.action_type,
                "gt_coord_999": s.gt_coord_999,
                "screenshot_width": s.screenshot_width,
                "screenshot_height": s.screenshot_height,
                "generated_tokens": s.generated_tokens,
                "extraction_layer": s.extraction_layer,
            }
            for s in samples
        ],
    }
    torch.save(data, path)
    print(f"Saved {len(samples)} samples to {path} "
          f"({data['hidden_states'].shape})")


def load_extracted_samples(path: Path) -> dict[str, Any]:
    """Load the .pt file saved by ``save_extracted_samples``.

    Returns a dict with keys ``hidden_states``, ``gt_coords_norm``, ``meta``.
    """
    return torch.load(path, map_location="cpu", weights_only=False)
