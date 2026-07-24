"""MSD Qwen2-VL backend for offline AndroidControl evaluation.

The implementation intentionally imports MSD and its pinned runtime lazily so
the rest of GUIAccel remains importable outside the dedicated MSD environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from PIL import Image

from guiaccel.model.maiui_action_adapter import parse_model_action_output
from guiaccel.routing.common import TokenUsage, estimate_visual_tokens
from guiaccel.routing.fallback import FullModelRequest, FullModelResponse


ANDROIDCONTROL_MSD_SYSTEM_PROMPT = """You are a GUI agent. Given an Android screenshot, a goal, and action history, output exactly one next action.

Return this format:
<thinking>brief reason</thinking>
<tool_call>
{"name":"mobile_use","arguments":<action-json>}
</tool_call>

Allowed action-json values:
{"action":"click","coordinate":[x,y]}
{"action":"long_press","coordinate":[x,y]}
{"action":"type","text":"text"}
{"action":"swipe","direction":"up or down or left or right","coordinate":[x,y]}
{"action":"open","text":"app_name"}
{"action":"system_button","button":"back or home or enter"}
{"action":"wait"}
{"action":"terminate","status":"success or fail"}

Coordinates use the 0..999 screenshot scale. Output one action only."""


@dataclass(frozen=True)
class MSDQwen2VLConfig:
    base_model_path: str
    draft_model_path: str
    use_msd: bool = True
    max_new_tokens: int = 256
    max_pixels: int = 1_000_000
    min_pixels: int = 3_136
    total_token: int = 59
    depth: int = 5
    top_k: int = 10
    threshold: float = 1.0
    torch_dtype: str = "float16"

    def validate(self) -> None:
        if not self.base_model_path:
            raise ValueError("base_model_path is required")
        if not self.draft_model_path:
            raise ValueError("draft_model_path is required")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.total_token <= 0:
            raise ValueError("total_token must be positive; disable MSD with use_msd=false")


class MSDQwen2VLBackend:
    """BenchmarkFaithfulModel-compatible wrapper around MSD's ``EaModel``."""

    def __init__(self, config: MSDQwen2VLConfig, *, eager_load: bool = False) -> None:
        config.validate()
        self.config = config
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._get_input_embeds: Any | None = None
        if eager_load:
            self._ensure_loaded()

    def generate(self, request: FullModelRequest) -> FullModelResponse:
        if request.benchmark != "AndroidControl":
            raise ValueError(f"MSDQwen2VLBackend only supports AndroidControl, got {request.benchmark!r}")
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._get_input_embeds is not None

        device = next(self._model.base_model.parameters()).device
        if device.type == "cuda":
            self._torch.cuda.synchronize(device)
            self._torch.cuda.reset_peak_memory_stats(device)
        request_started_at = time.perf_counter()

        image = _load_request_image(request, max_pixels=self.config.max_pixels)
        messages = _build_messages(request, image=image)
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        input_embeds = self._get_input_embeds(
            inputs.input_ids,
            inputs.pixel_values,
            inputs.image_grid_thw,
            self._model.base_model,
        )

        prior_accept_length = int(self._model.acclen)
        prior_rounds = int(self._model.accnum)
        if device.type == "cuda":
            self._torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        generate_fn = self._model.msdgenerate if self.config.use_msd else self._model.naivegenerate
        output_ids = generate_fn(
            inputs.input_ids,
            inputs_embeds=input_embeds,
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            max_new_tokens=min(int(request.max_new_tokens), self.config.max_new_tokens),
        )
        if device.type == "cuda":
            self._torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            action = parse_model_action_output(
                output_text,
                screenshot_width=request.screenshot.width,
                screenshot_height=request.screenshot.height,
                roi_bbox=request.roi_bbox,
                assume_android_coordinate_scale=True,
            )
        except ValueError:
            action = None

        accept_length = int(self._model.acclen) - prior_accept_length
        verification_rounds = int(self._model.accnum) - prior_rounds
        end_to_end_latency_ms = (time.perf_counter() - request_started_at) * 1000.0
        peak_gpu_memory_bytes = (
            int(self._torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        timing = {
            "model_latency_ms": latency_ms,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "peak_gpu_memory_bytes": float(peak_gpu_memory_bytes),
            "accepted_draft_tokens": float(accept_length),
            "verification_rounds": float(verification_rounds),
            "accepted_tokens_per_round": (
                float(accept_length) / verification_rounds if verification_rounds else 0.0
            ),
        }
        return FullModelResponse(
            action=action,
            token_usage=TokenUsage(
                prompt_tokens=int(inputs.input_ids.shape[1]),
                visual_tokens=estimate_visual_tokens(
                    request.screenshot.width,
                    request.screenshot.height,
                ),
                generated_tokens=int(generated_ids.shape[1]),
                full_model_calls=1,
            ),
            latency_ms=end_to_end_latency_ms,
            raw_output=output_text,
            model_name="MSD-Qwen2VL-7B-Instruct",
            timing=timing,
        )

    def generate_batch(
        self,
        requests: Sequence[FullModelRequest],
    ) -> tuple[FullModelResponse, ...]:
        # MSD's public Qwen2-VL tree decoder owns mutable KV buffers and is only
        # safe for one request at a time.
        return tuple(self.generate(request) for request in requests)

    def estimate_request_cost(self, request: FullModelRequest) -> Mapping[str, int]:
        return {
            "text_tokens": max(1, int(request.estimated_token_usage.prompt_tokens)),
            "image_count": 1,
            "visual_tokens": estimate_visual_tokens(
                request.screenshot.width,
                request.screenshot.height,
            ),
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from guiaccel.model.msd.core.ea_model import EaModel
            from guiaccel.model.msd.core.utils import get_input_embeds_qwen2vl
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MSD runtime is incomplete. Activate the dedicated msd-androidcontrol "
                "environment and install MSD/EAGLE plus qwen-vl-utils."
            ) from exc

        dtype = getattr(torch, self.config.torch_dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {self.config.torch_dtype}")
        model, _ = EaModel.from_pretrained(
            base_model_path=self.config.base_model_path,
            ea_model_path=self.config.draft_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=self.config.total_token,
            depth=self.config.depth,
            top_k=self.config.top_k,
            threshold=self.config.threshold,
        )
        model.eval()
        model.base_model.tie_weights()
        self._model = model
        self._processor = AutoProcessor.from_pretrained(
            self.config.base_model_path,
            max_pixels=self.config.max_pixels,
            min_pixels=self.config.min_pixels,
        )
        self._torch = torch
        self._get_input_embeds = get_input_embeds_qwen2vl


def _build_messages(request: FullModelRequest, *, image: Image.Image) -> list[dict[str, Any]]:
    user_text = (
        f"{request.prompt_text}\n"
        "Choose the single next action for the current screenshot."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": ANDROIDCONTROL_MSD_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _load_request_image(request: FullModelRequest, *, max_pixels: int) -> Image.Image:
    image = Image.open(BytesIO(request.screenshot.read_bytes())).convert("RGB")
    if request.roi_bbox is not None:
        image = image.crop(request.roi_bbox)
    if image.width * image.height > max_pixels:
        scale = (max_pixels / float(image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


__all__ = [
    "ANDROIDCONTROL_MSD_SYSTEM_PROMPT",
    "MSDQwen2VLBackend",
    "MSDQwen2VLConfig",
]
