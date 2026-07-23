"""Benchmark-faithful full-baseline calls shared by repository build and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from guiaccel.routing.common import StepContext, TokenUsage, estimate_visual_tokens
from guiaccel.types import BBox, CanonicalAction, ScreenshotAsset


@dataclass(frozen=True)
class ModelRuntimeSpec:
    """Runtime model specification supporting both local and vLLM service backends."""

    benchmark: str
    backend: str = "vllm"
    base_model_path: Path | str | None = None
    lora_adapter_path: Path | str | None = None
    model_name: str = "Qwen3-VL-8B-Instruct"
    served_model_name: str | None = None

    @property
    def request_model_name(self) -> str:
        if self.served_model_name:
            return str(self.served_model_name)
        if self.model_name:
            return str(self.model_name)
        if self.base_model_path is not None:
            return str(self.base_model_path)
        return "unknown-model"


# Backward-compat alias — old code that references QwenLoRAModelSpec still works.
QwenLoRAModelSpec = ModelRuntimeSpec


@dataclass(frozen=True)
class FallbackModelConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 32
    repetition_penalty: float = 1.0
    image_max_pixels: int = 1_000_000


@dataclass(frozen=True)
class FullModelRequest:
    observation_id: str
    reason: str
    benchmark: str
    screenshot: ScreenshotAsset
    prompt_text: str
    history_length: int
    support_context: Mapping[str, Any]
    model_spec: QwenLoRAModelSpec | None
    temperature: float
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    image_max_pixels: int
    estimated_token_usage: TokenUsage
    roi_bbox: BBox | None = None
    reduced_valid_labels: tuple[str, ...] = ()
    reduced_label_action_map: tuple[tuple[str, CanonicalAction], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FullModelResponse:
    action: CanonicalAction | None
    token_usage: TokenUsage = TokenUsage(full_model_calls=1)
    latency_ms: float = 0.0
    raw_output: str | None = None
    model_name: str | None = None
    timing: dict[str, float] | None = None


@runtime_checkable
class BenchmarkFaithfulModel(Protocol):
    def generate(self, request: FullModelRequest) -> FullModelResponse:
        raise NotImplementedError


@runtime_checkable
class BatchBenchmarkFaithfulModel(BenchmarkFaithfulModel, Protocol):
    def generate_batch(self, requests: Sequence[FullModelRequest]) -> Sequence[FullModelResponse]:
        raise NotImplementedError


@runtime_checkable
class CostAwareBenchmarkFaithfulModel(BenchmarkFaithfulModel, Protocol):
    def estimate_request_cost(self, request: FullModelRequest) -> Mapping[str, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class FallbackResult:
    request: FullModelRequest
    response: FullModelResponse

    @property
    def action(self) -> CanonicalAction | None:
        return self.response.action

    @property
    def token_usage(self) -> TokenUsage:
        return self.response.token_usage

    @property
    def latency_ms(self) -> float:
        return self.response.latency_ms


def generate_full_model_responses(
    model: BenchmarkFaithfulModel,
    requests: Sequence[FullModelRequest],
) -> tuple[FullModelResponse, ...]:
    """Run one or more prepared requests, using batched generation when available."""

    if not requests:
        return tuple()
    if isinstance(model, BatchBenchmarkFaithfulModel):
        raw_responses = tuple(model.generate_batch(tuple(requests)))
    else:
        raw_responses = tuple(model.generate(request) for request in requests)
    if len(raw_responses) != len(requests):
        raise ValueError(
            f"Expected {len(requests)} full-model responses, received {len(raw_responses)}."
        )
    return tuple(
        finalize_full_model_response(request, response)
        for request, response in zip(requests, raw_responses)
    )


def build_fallback_request(
    context: StepContext,
    *,
    reason: str,
    config: FallbackModelConfig | None = None,
    model_spec: QwenLoRAModelSpec | None = None,
) -> FullModelRequest:
    resolved_config = config or FallbackModelConfig()
    prompt_text = build_full_prompt(context)
    estimated_usage = estimate_fallback_usage(context, config=resolved_config)
    return FullModelRequest(
        observation_id=context.observation_id,
        reason=reason,
        benchmark=str(context.record.normalized_metadata.get("benchmark") or "unknown"),
        screenshot=context.record.screenshot,
        prompt_text=prompt_text,
        history_length=len(context.history),
        support_context=dict(context.support_context),
        model_spec=model_spec,
        temperature=float(resolved_config.temperature),
        top_p=float(resolved_config.top_p),
        max_new_tokens=int(resolved_config.max_new_tokens),
        repetition_penalty=float(resolved_config.repetition_penalty),
        image_max_pixels=int(resolved_config.image_max_pixels),
        estimated_token_usage=estimated_usage,
        roi_bbox=None,
    )


def run_fallback(
    context: StepContext,
    model: BenchmarkFaithfulModel,
    *,
    reason: str,
    config: FallbackModelConfig | None = None,
    model_spec: QwenLoRAModelSpec | None = None,
    prepared_request: FullModelRequest | None = None,
) -> FallbackResult:
    request = prepared_request or build_fallback_request(
        context,
        reason=reason,
        config=config,
        model_spec=model_spec,
    )
    response = generate_full_model_responses(model, (request,))[0]
    return FallbackResult(request=request, response=response)


def run_fallback_batch(
    contexts: Sequence[StepContext],
    model: BenchmarkFaithfulModel,
    *,
    reason: str,
    config: FallbackModelConfig | None = None,
    model_spec: QwenLoRAModelSpec | None = None,
    prepared_requests: Sequence[FullModelRequest] | None = None,
) -> tuple[FallbackResult, ...]:
    context_items = tuple(contexts)
    requests = (
        tuple(prepared_requests)
        if prepared_requests is not None
        else tuple(
            build_fallback_request(
                context,
                reason=reason,
                config=config,
                model_spec=model_spec,
            )
            for context in context_items
        )
    )
    if len(requests) != len(context_items):
        raise ValueError("prepared_requests must align one-to-one with contexts.")
    responses = generate_full_model_responses(model, requests)
    return tuple(
        FallbackResult(request=request, response=response)
        for request, response in zip(requests, responses)
    )


def finalize_full_model_response(
    request: FullModelRequest,
    response: FullModelResponse,
) -> FullModelResponse:
    if response.token_usage.total_tokens == 0:
        return FullModelResponse(
            action=response.action,
            token_usage=request.estimated_token_usage,
            latency_ms=response.latency_ms,
            raw_output=response.raw_output,
            model_name=response.model_name,
            timing=response.timing,
        )
    return response


def fallback_request_cache_key(request: FullModelRequest) -> str:
    if request.screenshot.path is not None:
        screenshot_id = f"path:{request.screenshot.path.resolve()}"
    elif request.screenshot.png_bytes is not None:
        digest = hashlib.blake2b(request.screenshot.png_bytes, digest_size=16).hexdigest()
        screenshot_id = f"bytes:{digest}:{len(request.screenshot.png_bytes)}"
    else:
        screenshot_id = f"blank:{request.screenshot.width}x{request.screenshot.height}"
    payload = (
        request.benchmark,
        request.observation_id,
        request.prompt_text,
        screenshot_id,
        request.history_length,
        request.temperature,
        request.top_p,
        request.max_new_tokens,
        request.image_max_pixels,
    )
    return json.dumps(payload)


def build_full_prompt(context: StepContext) -> str:
    benchmark = str(context.record.normalized_metadata.get("benchmark") or "")
    if benchmark == "AndroidControl":
        return _build_androidcontrol_prompt(context)

    history_lines = [
        f"{record.canonical_action.action_type}:{record.canonical_action.argument or ''}"
        for record in context.history
    ]
    support_summary = _serialize_mapping(context.support_context)
    metadata_summary = _serialize_mapping(context.record.normalized_metadata)
    return "\n".join(
        [
            f"goal: {context.record.goal}",
            f"metadata: {metadata_summary or 'none'}",
            f"history: {' | '.join(history_lines) if history_lines else 'none'}",
            f"support: {support_summary or 'none'}",
        ]
    )


def _build_androidcontrol_prompt(context: StepContext) -> str:
    """Build AndroidControl prompt matching the native Qwen3-VL mobile_use evaluation format.

    Matches the reference evaluation setup (step05_AC_Qwen3VL):
      "Task goal: X\nInstruction for the current step: Y\nActions already performed:\n..."
    Excludes screen_description, screen_size, and support fields that add noise
    when the model processes the screenshot visually with 0-999 coordinates.
    """
    metadata = dict(context.record.normalized_metadata)
    processed_step = dict(metadata.get("processed_step") or {})

    parts: list[str] = [f"Task goal: {context.record.goal}"]

    step_instruction = (
        metadata.get("step_instruction")
        or processed_step.get("step_instruction")
    )
    if step_instruction:
        parts.append(f"Instruction for the current step: {step_instruction}")

    if context.history:
        history_lines = [
            f"{record.canonical_action.action_type.lower()}: {record.canonical_action.argument or record.canonical_action.direction or record.canonical_action.app or '(done)'}"
            for record in context.history
        ]
        parts.append("Actions already performed:\n" + "\n".join(history_lines))

    return "\n".join(parts)


def estimate_fallback_usage(
    context: StepContext,
    *,
    config: FallbackModelConfig | None = None,
) -> TokenUsage:
    resolved_config = config or FallbackModelConfig()
    prompt_tokens = len(build_full_prompt(context).split())
    width = int(context.record.screenshot.width or 1)
    height = int(context.record.screenshot.height or 1)
    pixels = min(int(width * height), int(resolved_config.image_max_pixels))
    if width * height > pixels:
        scale = (float(pixels) / float(width * height)) ** 0.5
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        visual_tokens=estimate_visual_tokens(width, height),
        generated_tokens=max(1, int(resolved_config.max_new_tokens // 4)),
        full_model_calls=1,
    )


def _serialize_mapping(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return ""
    return "; ".join(f"{key}={value}" for key, value in sorted(mapping.items()))


__all__ = [
    "BatchBenchmarkFaithfulModel",
    "BenchmarkFaithfulModel",
    "CostAwareBenchmarkFaithfulModel",
    "FallbackModelConfig",
    "FallbackResult",
    "FullModelRequest",
    "FullModelResponse",
    "QwenLoRAModelSpec",
    "build_fallback_request",
    "build_full_prompt",
    "estimate_fallback_usage",
    "fallback_request_cache_key",
    "finalize_full_model_response",
    "generate_full_model_responses",
    "run_fallback",
    "run_fallback_batch",
]
