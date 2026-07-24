"""Extract hidden states from Qwen3-VL using GT-Forcing Prefill.

GT-Forcing Prefill constructs:

    input_ids = tokenize(system + image + user_prompt) ‖ tokenize(gt_prefix)

where ``gt_prefix`` is a forced assistant prefix ending at a chosen extract
point (default: action-type keyword), then runs a **single**
``model.forward()`` to obtain the last-layer hidden state at that position.

**Thinking-aware (E1-A) note:** Template ``<thinking>...</thinking>`` text is
injected into the GT prefix so train-time \(h_t\) better matches the AR trigger
distribution. This is for **training feature alignment only**. Fair deploy /
decode-eval latency must still measure real thinking decode cost separately
(do not claim template-GT extract latency as deployable \(\rho\)).

Due to causal attention masking, position *t*'s hidden state only depends on
tokens 0..t — regardless of whether they were autoregressively generated or
fed in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

# 与 qwen_backend 共用同一套 system prompt / 图像预处理，保证提取分布与推理一致
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

# 只有这两种动作需要坐标；其它动作（type/scroll/...）不做回归头提取
COORD_ACTION_TYPES = frozenset({"click", "long_press"})

# Qwen3-VL mobile_use 原生坐标尺度：像素归一化后再 ×999
SCALE_FACTOR = 999

# Thinking / extract-point API (E1-A default = template thinking @ action)
THINKING_MODES = frozenset({"none", "template", "ar_cache"})
# Single-point positions; ``multi`` = E2 one-forward capture of all three.
EXTRACT_POINTS = frozenset({"thinking_end", "action", "coord_bracket", "multi"})
SINGLE_EXTRACT_POINTS = ("thinking_end", "action", "coord_bracket")
DEFAULT_THINKING_MODE = "template"
DEFAULT_EXTRACT_POINT = "action"

# Tool-call skeleton up through the action-type keyword (no coordinates).
TOOL_CALL_UP_TO_ACTION: dict[str, str] = {
    "click": '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click"',
    "long_press": '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "long_press"',
}

# Extend past action to the coordinate array open-bracket (E2 ablation point).
COORD_BRACKET_SUFFIX = ', "coordinate": ['

# Backward-compatible alias used by older call sites / docs.
GT_PREFIX_TEMPLATES: dict[str, str] = dict(TOOL_CALL_UP_TO_ACTION)


# ---------------------------------------------------------------------------
# Prefix builder (shared by extract / train / eval)
# ---------------------------------------------------------------------------

def build_template_thinking(step: DatasetStep | Mapping[str, Any], action_type: str) -> str:
    """Deterministic E1-A thinking body from step_instruction + action_type.

    AndroidControl has no GT CoT; this synthesizes a short format-aligned
    thinking block so GT-Forcing \(h_t\) is conditioned on a thinking prefix.
    """
    if isinstance(step, Mapping):
        meta = step.get("metadata") or step
        step_instruction = str(meta.get("step_instruction") or "").strip()
    else:
        step_instruction = str(step.metadata.get("step_instruction") or "").strip()

    action_label = action_type.replace("_", " ")
    if step_instruction:
        body = (
            f"The current step instruction is: {step_instruction}\n"
            f"I will perform a {action_label} on the target UI element "
            f"described by that instruction."
        )
    else:
        body = (
            f"I will perform a {action_label} on the appropriate UI element "
            f"for this step."
        )
    return f"<thinking>\n{body}\n</thinking>\n"


def build_gt_prefix(
    step: DatasetStep | Mapping[str, Any] | None = None,
    *,
    action_type: str,
    thinking_mode: str = DEFAULT_THINKING_MODE,
    extract_point: str = DEFAULT_EXTRACT_POINT,
    thinking_text: str | None = None,
) -> str:
    """Build the forced assistant prefix for GT-Forcing Prefill.

    Parameters
    ----------
    thinking_mode :
        ``none`` — tool_call only (legacy baseline).
        ``template`` — E1-A synthetic ``<thinking>...</thinking>`` (default).
        ``ar_cache`` — use ``thinking_text`` from a prior AR generate (E1-B/E3).
    extract_point :
        ``thinking_end`` — end at ``</thinking>`` (requires thinking).
        ``action`` — end at action-type keyword (default; matches AR trigger).
        ``coord_bracket`` — end at ``"coordinate": [`` (fewer tokens saved).
    """
    mode = (thinking_mode or DEFAULT_THINKING_MODE).strip().lower()
    point = (extract_point or DEFAULT_EXTRACT_POINT).strip().lower()
    atype = action_type.strip().lower()

    if mode not in THINKING_MODES:
        raise ValueError(f"Unknown thinking_mode={thinking_mode!r}; expected one of {sorted(THINKING_MODES)}")
    if point == "multi":
        raise ValueError(
            "extract_point='multi' is only valid for extract_hidden_state "
            "(one forward → three positions); build_gt_prefix needs a single point"
        )
    if point not in SINGLE_EXTRACT_POINTS:
        raise ValueError(
            f"Unknown extract_point={extract_point!r}; expected one of {sorted(SINGLE_EXTRACT_POINTS)}"
        )
    if atype not in COORD_ACTION_TYPES:
        raise ValueError(f"Unsupported action_type={action_type!r} for GT prefix")

    thinking_block = ""
    if mode == "none":
        if point == "thinking_end":
            raise ValueError("extract_point='thinking_end' requires thinking_mode != 'none'")
    elif mode == "template":
        if step is None:
            raise ValueError("thinking_mode='template' requires step (for step_instruction)")
        thinking_block = build_template_thinking(step, atype)
    elif mode == "ar_cache":
        if not thinking_text or not str(thinking_text).strip():
            raise ValueError(
                "thinking_mode='ar_cache' requires non-empty thinking_text "
                "(E1-B/E3 cache; not built in T1)"
            )
        text = str(thinking_text).strip()
        if "<thinking>" not in text:
            text = f"<thinking>\n{text}\n</thinking>"
        if not text.endswith("\n"):
            text += "\n"
        thinking_block = text

    if point == "thinking_end":
        # Keep through closing tag + the conventional trailing newline after
        # </thinking> when present (matches tokenization of the longer prefixes).
        end = thinking_block.rfind("</thinking>")
        if end < 0:
            raise ValueError("thinking block missing </thinking> for extract_point=thinking_end")
        end_pos = end + len("</thinking>")
        if end_pos < len(thinking_block) and thinking_block[end_pos] == "\n":
            end_pos += 1
        return thinking_block[:end_pos]

    tool = TOOL_CALL_UP_TO_ACTION[atype]
    if point == "action":
        return f"{thinking_block}{tool}"
    # coord_bracket
    return f"{thinking_block}{tool}{COORD_BRACKET_SUFFIX}"


def worker_meta_list(data: Mapping[str, Any]) -> list[Any]:
    """Normalize worker/merged sample meta list (``meta`` or ``metadata``)."""
    meta = data.get("metadata")
    if meta is None:
        meta = data.get("meta")
    if meta is None:
        return []
    return list(meta)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSample:
    """一条训练样本：(hidden_state, GT 坐标) """

    episode_id: str
    step_index: int
    action_type: str
    hidden_state: torch.Tensor          # [hidden_dim]，默认 4096 (primary / action)
    gt_coord_norm: torch.Tensor         # [2]，按截图像素归一化到约 [0,1]
    gt_coord_999: tuple[int, int]       # 同坐标的 0–999 整数表示（评测/日志用）
    screenshot_width: int
    screenshot_height: int
    generated_tokens: int               # 实际是 gt_prefix 的 token 数（兼容旧字段名）
    extraction_layer: int               # 提取层索引；-1 = 最后一层
    thinking_mode: str = DEFAULT_THINKING_MODE
    extract_point: str = DEFAULT_EXTRACT_POINT
    prefix_token_len: int = 0           # alias of generated_tokens; explicit for meta
    # E2 multi-point fields (set when extract_point == "multi")
    h_thinking_end: torch.Tensor | None = None
    h_action: torch.Tensor | None = None
    h_coord_bracket: torch.Tensor | None = None
    prefix_token_len_thinking_end: int = 0
    prefix_token_len_action: int = 0
    prefix_token_len_coord_bracket: int = 0


def _encode_prefix_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _multi_point_prefix_token_lengths(
    tokenizer: Any,
    step: DatasetStep | Mapping[str, Any],
    *,
    action_type: str,
    thinking_mode: str,
    thinking_text: str | None = None,
) -> tuple[str, dict[str, int]]:
    """Build longest (coord_bracket) prefix + per-point token lengths for one forward.

    Returns ``(full_prefix_text, {point: prefix_token_len})``.
    Requires thinking_mode != 'none' so thinking_end is defined.

    Token lengths are derived via ``return_offsets_mapping`` on the *full*
    coord_bracket string (Qwen BPE is not prefix-stable across separate
    ``encode()`` calls of nested substrings).
    """
    mode = (thinking_mode or DEFAULT_THINKING_MODE).strip().lower()
    if mode == "none":
        raise ValueError("extract_point='multi' requires thinking_mode != 'none'")

    prefixes = {
        point: build_gt_prefix(
            step,
            action_type=action_type,
            thinking_mode=mode,
            extract_point=point,
            thinking_text=thinking_text,
        )
        for point in SINGLE_EXTRACT_POINTS
    }
    full = prefixes["coord_bracket"]
    # String nesting: thinking_end ⊂ action ⊂ coord_bracket
    if not prefixes["action"].startswith(prefixes["thinking_end"]):
        raise ValueError(
            "multi-point prefix nesting broken: action must start with thinking_end text"
        )
    if not full.startswith(prefixes["action"]):
        raise ValueError(
            "multi-point prefix nesting broken: coord_bracket must start with action text"
        )

    encoded = tokenizer(
        full,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    # HF tokenizer may return BatchEncoding with lists or nested lists.
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
        offsets = offsets[0]
    if len(input_ids) != len(offsets) or len(input_ids) < 1:
        raise ValueError(
            f"offset_mapping length mismatch: ids={len(input_ids)} offs={len(offsets)}"
        )

    def _token_len_at_char_end(char_end: int) -> int:
        """Number of tokens covering characters [0, char_end)."""
        if char_end <= 0:
            raise ValueError(f"char_end must be > 0, got {char_end}")
        last_idx = -1
        for i, (start, _end) in enumerate(offsets):
            if start < char_end:
                last_idx = i
            else:
                break
        if last_idx < 0:
            raise ValueError(f"no token covers char_end={char_end}")
        return last_idx + 1

    lengths = {
        "thinking_end": _token_len_at_char_end(len(prefixes["thinking_end"])),
        "action": _token_len_at_char_end(len(prefixes["action"])),
        "coord_bracket": len(input_ids),
    }
    if lengths["coord_bracket"] != _token_len_at_char_end(len(full)):
        # Full string should map to all tokens.
        raise ValueError(
            f"coord_bracket token length mismatch: "
            f"{lengths['coord_bracket']} vs {_token_len_at_char_end(len(full))}"
        )
    if not (
        1 <= lengths["thinking_end"] <= lengths["action"] <= lengths["coord_bracket"]
    ):
        raise ValueError(f"multi-point token lengths not nested: {lengths}")
    return full, lengths


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_extraction(
    model_path: str | Path,
    *,
    device: str | int = 0,
    dtype: str = "bfloat16",
) -> tuple[Any, Any, Any]:
    """加载 Qwen3-VL + processor，供 hidden-state 提取使用。

    Returns
    -------
    (model, processor, torch_module)
        torch_module 是 ``torch`` 包引用，方便调用方不必再 import。
    """
    import torch as _torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # 字符串 dtype → torch.dtype
    dtype_map = {
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
    }
    resolved_dtype = dtype_map.get(dtype, _torch.bfloat16)
    resolved_path = str(Path(model_path).resolve())

    # Processor：chat template + 图像预处理 + tokenizer
    processor = AutoProcessor.from_pretrained(resolved_path)
    if hasattr(processor, "tokenizer"):
        # left padding 是 generate batch 惯例；单样本 forward 无影响，保持一致
        processor.tokenizer.padding_side = "left"

    # 整模放到单卡；强制 FA2（ViT + LLM 均走 flash_attention_2）
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved_path,
        device_map=device,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()  # 关闭 dropout；提取阶段不做训练
    return model, processor, _torch


# ---------------------------------------------------------------------------
# Core extraction — GT-Forcing Prefill
# ---------------------------------------------------------------------------

def _build_messages(
    step: DatasetStep,
    image: Any,
) -> list[Mapping[str, Any]]:
    """构造与 qwen_backend 一致的 chat messages (system + user[image,text]) """
    goal = step.goal
    step_instruction = step.metadata.get("step_instruction") or ""
    history_types = step.metadata.get("history_action_types", ())

    # 用户文本：goal / 当前指令 / 历史动作类型
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
    """加载并预处理截图，流程与 qwen_backend 完全对齐。

    步骤：读图 → 过长宽比 pad → 超过 max_pixels 则等比缩小。
    """
    from PIL import Image as _Image

    image = _load_image(screenshot, _Image)
    # 极端长宽比会触发 Qwen processor 校验失败，先 pad 到安全比
    image = _pad_image_to_safe_aspect_ratio(
        image,
        image_module=_Image,
        max_aspect_ratio=QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
    )
    width, height = image.size
    # 像素上限：默认 1M，与 SkillReuse / GUIAccel 推理侧一致
    if width * height > max_pixels:
        scale = (float(max_pixels) / float(width * height)) ** 0.5
        image = image.resize((
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ))
    return image


def extract_hidden_state(
    model: Any,
    processor: Any,
    step: DatasetStep,
    *,
    layer: int = -1,
    max_new_tokens: int = 512,
    max_pixels: int = 1_000_000,
    thinking_mode: str = DEFAULT_THINKING_MODE,
    extract_point: str = DEFAULT_EXTRACT_POINT,
    thinking_text: str | None = None,
) -> ExtractedSample | None:
    """用 GT-Forcing Prefill 提取指定 extract_point 的 hidden state。

    核心流程
    --------
    1. 过滤非坐标动作 / 缺失坐标样本
    2. 读 GT 像素坐标并归一化
    3. 构造 prompt tokens + thinking-aware GT 前缀 tokens
    4. **单次** ``model.forward(output_hidden_states=True)``
    5. 取指定层、对应位置向量作为回归头输入

    When ``extract_point='multi'`` (E2), the forward uses the longest
    ``coord_bracket`` prefix and indexes ``h_thinking_end`` / ``h_action`` /
    ``h_coord_bracket`` in that same pass (still 1 forward). Primary
    ``hidden_state`` mirrors ``h_action`` for backward compatibility.

    Parameters
    ----------
    layer : int
        Transformer 层索引。``-1`` = 最后一层（过 final LayerNorm，接 LM head
        前的表示）；``-2`` = 倒数第二层，用于消融。
    max_new_tokens : int
        旧版 autoregressive 提取器遗留参数，当前未使用，仅保持 API 兼容。
    thinking_mode / extract_point :
        See :func:`build_gt_prefix`. Default E1-A: template thinking @ action.
        ``extract_point='multi'`` enables E2 multi-point capture.
    """
    del max_new_tokens  # API compat only
    # ── 1. 校验动作类型：只处理 click / long_press ───────────────────────
    raw_action = step.raw_action
    action_type = str(raw_action.get("action_type", "")).lower()
    if action_type not in COORD_ACTION_TYPES:
        return None

    # ── 2. 解析 GT 坐标（绝对像素）───────────────────────────────────────
    # 优先 bbox 中心；否则用 raw_action 的 x/y。
    # 注意：AndroidControl 极少数标注会越界（x>W 或 y>H），此处不做 clip，
    # 保留原始分布，由训练阶段决定 drop/clip。
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

    # 归一化分母用**原始截图**宽高（metadata），不是 resize 后的图像尺寸；
    # 与 AndroidControl 评测的绝对像素坐标系一致。
    sw = int(step.metadata.get("screenshot_width", 1080))
    sh = int(step.metadata.get("screenshot_height", 1920))

    gt_x_norm = gt_x_abs / max(sw, 1)
    gt_y_norm = gt_y_abs / max(sh, 1)
    gt_x_999 = int(round(gt_x_norm * SCALE_FACTOR))
    gt_y_999 = int(round(gt_y_norm * SCALE_FACTOR))

    # ── 3. 图像预处理 + chat messages ────────────────────────────────────
    image = _prepare_image(step.screenshot, max_pixels=max_pixels)
    messages = _build_messages(step, image)

    # ── 4. Tokenize prompt（system + image + user + generation 起始符）───
    # add_generation_prompt=True 会追加 assistant 起始 token，
    # 后续 GT 前缀等价于“强制写入”的已生成内容。
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
    prompt_len = int(inputs["input_ids"].shape[-1])

    mode_norm = (thinking_mode or DEFAULT_THINKING_MODE).strip().lower()
    point_norm = (extract_point or DEFAULT_EXTRACT_POINT).strip().lower()
    if point_norm not in EXTRACT_POINTS:
        raise ValueError(
            f"Unknown extract_point={extract_point!r}; expected one of {sorted(EXTRACT_POINTS)}"
        )

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    multi = point_norm == "multi"
    point_lens: dict[str, int] | None = None
    if multi:
        gt_prefix, point_lens = _multi_point_prefix_token_lengths(
            tokenizer,
            step,
            action_type=action_type,
            thinking_mode=mode_norm,
            thinking_text=thinking_text,
        )
        gt_prefix_len = point_lens["coord_bracket"]
    else:
        gt_prefix = build_gt_prefix(
            step,
            action_type=action_type,
            thinking_mode=thinking_mode,
            extract_point=point_norm,
            thinking_text=thinking_text,
        )
        gt_prefix_len = len(_encode_prefix_ids(tokenizer, gt_prefix))

    # ── 5. Tokenize GT 前缀（thinking + tool skeleton；不含坐标值）────────
    gt_prefix_id_list = _encode_prefix_ids(tokenizer, gt_prefix)
    if len(gt_prefix_id_list) != gt_prefix_len:
        raise ValueError("gt_prefix token length inconsistency")
    gt_prefix_tensor = torch.tensor(
        [gt_prefix_id_list],
        dtype=inputs["input_ids"].dtype,
        device=device,
    )

    # ── 6. 拼接：prompt ‖ gt_prefix，并同步扩展各类 mask / 多模态字段 ──
    # 最终序列末位 = extract_point 对应最后一个 token → 提取位置。
    input_ids = torch.cat([inputs["input_ids"], gt_prefix_tensor], dim=-1)

    # attention_mask：GT 前缀全部有效（1）
    if "attention_mask" in inputs:
        gt_attn = torch.ones(
            (1, gt_prefix_len),
            dtype=inputs["attention_mask"].dtype,
            device=device,
        )
        attention_mask = torch.cat([inputs["attention_mask"], gt_attn], dim=-1)
    else:
        attention_mask = torch.ones_like(input_ids)

    # mm_token_type_ids：Qwen3-VL 的 M-RoPE 需要区分 image/video/text。
    # processor 已返回该字段；GT 前缀是纯文本 → 填 0。
    # 若遗漏会在 compute_3d_position_ids 处报错。
    if "mm_token_type_ids" in inputs:
        gt_mm_type = torch.zeros(
            (1, gt_prefix_len),
            dtype=inputs["mm_token_type_ids"].dtype,
            device=device,
        )
        mm_token_type_ids = torch.cat(
            [inputs["mm_token_type_ids"], gt_mm_type], dim=-1,
        )
    else:
        mm_token_type_ids = None

    # 组装 forward 参数：文本序列变长，视觉特征张量长度不变
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,  # 需要各层 hidden；显存仍远低于 generate
    }
    if mm_token_type_ids is not None:
        forward_kwargs["mm_token_type_ids"] = mm_token_type_ids
    for key in ("pixel_values", "image_grid_thw",
                "pixel_values_videos", "video_grid_thw"):
        if key in inputs:
            forward_kwargs[key] = inputs[key]

    # ── 7. 单次 forward（GT-Forcing Prefill）─────────────────────────────
    # 因果 mask 保证末位 hidden 只依赖此前所有 token，与逐步 decode 等价，
    # 但无 KV cache 累积、无逐步 hidden 存储。
    with torch.no_grad():
        outputs = model(**forward_kwargs)

    # ── 8. 取指定层向量 ──────────────────────────────────────────────────
    # outputs.hidden_states: 长度 = num_layers+1 的 tuple
    #   [0] = embedding 输出
    #   [N] = 最终 LayerNorm 后（接 LM head）
    # 形状每个都是 (batch, seq_len, hidden_dim)
    layer_h = outputs.hidden_states[layer][0]  # (seq_len, hidden_dim)

    if multi:
        assert point_lens is not None
        # Absolute indices = prompt_len + prefix_token_len - 1
        idx_te = prompt_len + point_lens["thinking_end"] - 1
        idx_act = prompt_len + point_lens["action"] - 1
        idx_cb = prompt_len + point_lens["coord_bracket"] - 1
        h_te = layer_h[idx_te, :].detach().cpu().float()
        h_act = layer_h[idx_act, :].detach().cpu().float()
        h_cb = layer_h[idx_cb, :].detach().cpu().float()
        primary_len = point_lens["action"]
        return ExtractedSample(
            episode_id=step.episode_id,
            step_index=step.step_index,
            action_type=action_type,
            hidden_state=h_act,  # default primary = action
            gt_coord_norm=torch.tensor([gt_x_norm, gt_y_norm], dtype=torch.float32),
            gt_coord_999=(gt_x_999, gt_y_999),
            screenshot_width=sw,
            screenshot_height=sh,
            generated_tokens=primary_len,
            extraction_layer=layer,
            thinking_mode=mode_norm,
            extract_point="multi",
            prefix_token_len=primary_len,
            h_thinking_end=h_te,
            h_action=h_act,
            h_coord_bracket=h_cb,
            prefix_token_len_thinking_end=point_lens["thinking_end"],
            prefix_token_len_action=point_lens["action"],
            prefix_token_len_coord_bracket=point_lens["coord_bracket"],
        )

    h = layer_h[-1, :].detach().cpu().float()
    return ExtractedSample(
        episode_id=step.episode_id,
        step_index=step.step_index,
        action_type=action_type,
        hidden_state=h,
        gt_coord_norm=torch.tensor([gt_x_norm, gt_y_norm], dtype=torch.float32),
        gt_coord_999=(gt_x_999, gt_y_999),
        screenshot_width=sw,
        screenshot_height=sh,
        generated_tokens=gt_prefix_len,
        extraction_layer=layer,
        thinking_mode=mode_norm,
        extract_point=point_norm,
        prefix_token_len=gt_prefix_len,
    )


# ---------------------------------------------------------------------------
# Batch persistence
# ---------------------------------------------------------------------------

def save_extracted_samples(
    samples: Sequence[ExtractedSample],
    path: Path,
) -> None:
    """将样本列表落盘为 .pt。

    文件结构
    --------
    - ``hidden_states`` : Tensor [N, hidden_dim] (primary; = h_action when multi)
    - ``gt_coords_norm`` : Tensor [N, 2]
    - ``meta`` / ``metadata`` : list[dict]，逐样本元信息（同一 list 的两个键）
    - When multi-point: also ``h_thinking_end`` / ``h_action`` / ``h_coord_bracket``
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_multi = bool(samples) and samples[0].extract_point == "multi"
    if is_multi and any(s.extract_point != "multi" for s in samples):
        raise ValueError("Cannot mix multi-point and single-point samples in one save")
    if is_multi and any(
        s.h_thinking_end is None or s.h_action is None or s.h_coord_bracket is None
        for s in samples
    ):
        raise ValueError("multi-point samples missing h_* tensors")

    meta = []
    for s in samples:
        row: dict[str, Any] = {
            "episode_id": s.episode_id,
            "step_index": s.step_index,
            "action_type": s.action_type,
            "gt_coord_999": s.gt_coord_999,
            "screenshot_width": s.screenshot_width,
            "screenshot_height": s.screenshot_height,
            "generated_tokens": s.generated_tokens,
            "prefix_token_len": s.prefix_token_len or s.generated_tokens,
            "extraction_layer": s.extraction_layer,
            "thinking_mode": s.thinking_mode,
            "extract_point": s.extract_point,
        }
        if s.extract_point == "multi":
            row.update({
                "multi_point": True,
                "prefix_token_len_thinking_end": s.prefix_token_len_thinking_end,
                "prefix_token_len_action": s.prefix_token_len_action,
                "prefix_token_len_coord_bracket": s.prefix_token_len_coord_bracket,
            })
        meta.append(row)

    data: dict[str, Any] = {
        "hidden_states": torch.stack([s.hidden_state for s in samples]),
        "gt_coords_norm": torch.stack([s.gt_coord_norm for s in samples]),
        # Dual keys: workers historically wrote ``meta``; merge expects ``metadata``.
        "meta": meta,
        "metadata": meta,
    }
    if is_multi:
        data["h_thinking_end"] = torch.stack([s.h_thinking_end for s in samples])  # type: ignore[misc]
        data["h_action"] = torch.stack([s.h_action for s in samples])  # type: ignore[misc]
        data["h_coord_bracket"] = torch.stack([s.h_coord_bracket for s in samples])  # type: ignore[misc]
        data["multi_point"] = True
    torch.save(data, path)
    extra = ""
    if is_multi:
        extra = (
            f" multi=[h_te{tuple(data['h_thinking_end'].shape)}"
            f",h_act{tuple(data['h_action'].shape)}"
            f",h_cb{tuple(data['h_coord_bracket'].shape)}]"
        )
    print(
        f"Saved {len(samples)} samples to {path} "
        f"({data['hidden_states'].shape}){extra}"
    )


def select_hidden_states_for_point(
    data: Mapping[str, Any],
    extract_point: str = DEFAULT_EXTRACT_POINT,
) -> torch.Tensor:
    """Pick the hidden-state tensor for a given extract point from a .pt dict.

    Prefer explicit ``h_*`` multi-point fields; fall back to ``hidden_states``
    for single-point extracts (must match ``extract_point`` or default action).
    """
    point = (extract_point or DEFAULT_EXTRACT_POINT).strip().lower()
    if point == "multi":
        raise ValueError("select_hidden_states_for_point requires a single point, not 'multi'")
    if point not in SINGLE_EXTRACT_POINTS:
        raise ValueError(f"Unknown extract_point={extract_point!r}")

    key = f"h_{point}"
    if key in data and data[key] is not None:
        return data[key]
    if data.get("multi_point") and key not in data:
        raise KeyError(f"multi-point extract missing {key}")
    # Single-point legacy artifact: hidden_states is the only vector.
    meta = worker_meta_list(data)
    if meta:
        recorded = str(meta[0].get("extract_point") or DEFAULT_EXTRACT_POINT).lower()
        if recorded not in (point, "multi"):
            raise ValueError(
                f"Artifact extract_point={recorded!r} does not match "
                f"requested {point!r} and no {key} tensor is present"
            )
    return data["hidden_states"]


def load_extracted_samples(path: Path) -> dict[str, Any]:
    """加载 ``save_extracted_samples`` 写出的 .pt。

    Returns
    -------
    dict
        键为 ``hidden_states`` / ``gt_coords_norm`` / ``meta`` (and often ``metadata``).
        Multi-point artifacts also expose ``h_thinking_end`` / ``h_action`` /
        ``h_coord_bracket``.
        ``weights_only=False``：文件含非纯 tensor 的 meta list。
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    # Normalize dual keys so callers can use either name.
    meta = worker_meta_list(data)
    data["meta"] = meta
    data["metadata"] = meta
    if any(k in data for k in ("h_thinking_end", "h_action", "h_coord_bracket")):
        data["multi_point"] = True
    return data
