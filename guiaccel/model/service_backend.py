"""Generic runtime backends, with vLLM OpenAI-compatible serving as the primary path.

Adapted from SkillReuse for Qwen3-VL-8B-Instruct:
  - Uses native Qwen3-VL mobile_use format with 0-999 coordinate scale for AndroidControl.
  - parse_model_action_output handles coordinate back-projection (0-999 → absolute pixels).
  - Default visible_cuda_devices covers all 8 H20 GPUs.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import hashlib
import io
import json
from pathlib import Path
import ssl
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from guiaccel.model.qwen_backend import (
    ACTION_CALL_RE,
    ACTION_JSON_RE,
    DependencyUnavailableError,
    QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
    QwenBackendConfig as LocalQwenBackendConfig,
    _action_from_call,
    _extract_json_object,
    _format_valid_action_label_block,
    _is_action_semantically_complete,
    _is_reduced_request_reason,
    _load_image,
    _normalize_crop_bbox_for_processor,
    _normalize_label_output,
    _pad_image_to_safe_aspect_ratio,
    _payload_to_action,
    _resolved_request_reduced_label_action_map,
    _resolved_request_valid_labels,
    _select_valid_label_from_output,
    build_qwen_backend,
)
from guiaccel.routing.common import TokenUsage, estimate_visual_tokens
from guiaccel.routing.execution import (
    BatchLocalDisambiguationModel,
    DisambiguationRequest,
    DisambiguationResponse,
)
from guiaccel.routing.fallback import (
    BenchmarkFaithfulModel,
    CostAwareBenchmarkFaithfulModel,
    FullModelRequest,
    FullModelResponse,
    ModelRuntimeSpec,
)
from guiaccel.model.maiui_action_adapter import parse_model_action_output
from guiaccel.types import BBox, CanonicalAction, ScreenshotAsset

# Native Qwen3-VL / MAI-UI system prompt for AndroidControl evaluation.
# Qwen3-VL-8B-Instruct outputs <thinking>...<tool_call>mobile_use(...)</tool_call>
# with 0-999 scale coordinates; parse_model_action_output converts to absolute pixels.
# 官方 MAI-UI / Qwen3-VL navigation system prompt（逐字，与参考仓库 step05_AC_Qwen3VL 一致）
# 包含完整 action space 描述，模型需要它来知道有哪些动作可用
ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]}
{"action": "open", "text": "app_name"}
{"action": "system_button", "button": "button_name"}
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}

## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
- Coordinates follow the 0..999 screenshot scale (x: 0=left, 999=right; y: 0=top, 999=bottom)."""


@dataclass(frozen=True)
class ModelServiceConfig:
    """Provider/runtime configuration shared by discovery, routing, and evaluation."""

    kind: str | None = None
    api_base: str = "http://127.0.0.1:8000/v1"
    api_bases: tuple[str, ...] = ()
    api_key: str = "EMPTY"
    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    request_retries: int = 2
    retry_backoff_seconds: float = 1.0
    client_max_parallel_requests: int = 8
    image_input_mode: str = "base64_data_url"
    allow_local_file_urls: bool = False
    verify_ssl: bool = True
    default_image_max_pixels: int = 1_000_000
    full_model_system_prompt: str = (
        "You are a benchmark-faithful GUI action model. "
        "Return exactly one JSON object with keys: action_type, argument, bbox, app, direction. "
        "Use null for fields not needed by the action. "
        "bbox=[left,top,right,bottom] must use ABSOLUTE pixel coordinates from the screenshot "
        "(x: 0=left edge, y: 0=top edge of screen). Do NOT use normalized or 0-999 scale. "
        "action_type values: "
        "CLICK (bbox=target element); "
        "LONG_PRESS (bbox=target element); "
        "TYPE (argument=text string); "
        "SCROLL (direction=up/down/left/right); "
        "NAV (argument=back/home/enter or app name); "
        "TERMINATE; "
        "WAIT."
    )
    reduced_model_system_prompt: str = (
        "You are a benchmark-faithful reduced GUI action model. Return exactly one legal action label "
        "from the prompt. Do not return JSON."
    )
    local_disambiguation_system_prompt: str = (
        "You are selecting one benchmark-native GUI action label. Return exactly one label from the valid label list."
    )
    # All 8 H20 GPUs are available on the target server.
    visible_cuda_devices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    device_map: int | str | None = 0
    torch_dtype: str = "bfloat16"
    attn_implementation: str | Mapping[str, str] | None = "flash_attention_2"
    trust_remote_code: bool = False
    processor_from_adapter_first: bool = True
    allow_cpu: bool = False
    keep_ratio: float | None = None
    extra_request_headers: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedImageURL:
    url: str
    width: int
    height: int


SERVICE_BACKEND_KINDS = frozenset({"vllm", "vllm_openai", "openai"})
LOCAL_TRANSFORMERS_BACKEND_KINDS = frozenset({"transformers", "qwen_transformers", "local_transformers", "visionzip"})
DIVPRUNE_BACKEND_KINDS = frozenset({"divprune"})


class VLLMOpenAIBackend:
    """OpenAI-compatible vLLM client backend."""

    def __init__(
        self,
        model_spec: ModelRuntimeSpec,
        *,
        config: ModelServiceConfig | None = None,
        eager_load: bool = False,
    ) -> None:
        self.model_spec = model_spec
        self.config = config or ModelServiceConfig()
        self._service_ready = False
        if eager_load:
            self._ensure_service_ready()

    def generate(self, request: FullModelRequest) -> FullModelResponse:
        return self.generate_batch((request,))[0]

    def generate_batch(self, requests: Sequence[FullModelRequest]) -> tuple[FullModelResponse, ...]:
        if not requests:
            return tuple()
        self._ensure_service_ready()
        return self._run_parallel(requests, self._generate_one)

    def estimate_request_cost(self, request: FullModelRequest) -> Mapping[str, int]:
        _, width, height = _resolve_image_geometry(
            request.screenshot,
            crop_bbox=request.roi_bbox,
            max_pixels=request.image_max_pixels or self.config.default_image_max_pixels,
        )
        return {
            "text_tokens": max(1, int(request.estimated_token_usage.prompt_tokens)),
            "image_count": 1,
            "visual_tokens": estimate_visual_tokens(width, height),
        }

    def disambiguate(self, request: DisambiguationRequest) -> DisambiguationResponse:
        return self.disambiguate_batch((request,))[0]

    def disambiguate_batch(
        self,
        requests: Sequence[DisambiguationRequest],
    ) -> tuple[DisambiguationResponse, ...]:
        if not requests:
            return tuple()
        self._ensure_service_ready()
        return self._run_parallel(requests, self._disambiguate_one)

    def _ensure_service_ready(self) -> None:
        if self._service_ready:
            return
        payload = self._request_json("GET", "/models")
        model_items = payload.get("data")
        if not isinstance(model_items, list) or not model_items:
            raise RuntimeError(f"vLLM /models returned an unexpected payload: {payload!r}")
        available_models = {
            str(item.get("id"))
            for item in model_items
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        request_model_name = self.model_spec.request_model_name
        if available_models and request_model_name not in available_models:
            raise RuntimeError(
                "Configured vLLM request model "
                f"{request_model_name!r} was not returned by /models. Available: {sorted(available_models)!r}."
            )
        self._service_ready = True

    def _run_parallel(self, items: Sequence[Any], fn: Any) -> tuple[Any, ...]:
        max_workers = max(1, min(len(items), int(self.config.client_max_parallel_requests)))
        if max_workers == 1:
            return tuple(fn(item) for item in items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(fn, items))

    def _generate_one(self, request: FullModelRequest) -> FullModelResponse:
        prepared_image = self._prepare_image_url(
            request.screenshot,
            crop_bbox=request.roi_bbox,
            max_pixels=request.image_max_pixels or self.config.default_image_max_pixels,
        )
        reduced_mode = bool(_resolved_request_valid_labels(request)) or _is_reduced_request_reason(request.reason)
        payload = {
            "model": self.model_spec.request_model_name,
            "messages": self._build_full_messages(request, prepared_image=prepared_image, reduced_mode=reduced_mode),
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "max_tokens": int(request.max_new_tokens),
            "repetition_penalty": float(request.repetition_penalty),
            "seed": 42,  # 确定性推理，与参考仓库一致
        }
        started_at = time.perf_counter()
        response_payload = self._request_json("POST", "/chat/completions", payload)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        output_text = _chat_completion_text(response_payload)
        action = None
        if reduced_mode:
            action = self._parse_reduced_request_output(output_text, request=request)
        if action is None:
            try:
                action = self._parse_action_output(output_text, request=request)
            except ValueError:
                action = None
        return FullModelResponse(
            action=action,
            token_usage=_resolve_full_token_usage(
                request,
                response_payload=response_payload,
                prepared_image=prepared_image,
            ),
            latency_ms=latency_ms,
            raw_output=output_text,
            model_name=str(response_payload.get("model") or self.model_spec.request_model_name),
        )

    def _disambiguate_one(self, request: DisambiguationRequest) -> DisambiguationResponse:
        prepared_image = self._prepare_image_url(
            request.visual.screenshot,
            crop_bbox=request.visual.crop_bbox,
            max_pixels=request.visual.pixel_budget or self.config.default_image_max_pixels,
        )
        payload = {
            "model": self.model_spec.request_model_name,
            "messages": self._build_disambiguation_messages(request, prepared_image=prepared_image),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(request.max_new_tokens),
            "repetition_penalty": 1.0,
        }
        started_at = time.perf_counter()
        response_payload = self._request_json("POST", "/chat/completions", payload)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        output_text = _chat_completion_text(response_payload)
        selected_label = self._parse_disambiguation_output(
            output_text,
            valid_labels=request.valid_labels,
        )
        return DisambiguationResponse(
            selected_label=selected_label,
            valid=selected_label in request.valid_labels if selected_label is not None else False,
            raw_output=output_text,
            token_usage=_resolve_disambiguation_token_usage(
                request,
                response_payload=response_payload,
                prepared_image=prepared_image,
            ),
            latency_ms=latency_ms,
        )

    def _build_full_messages(
        self,
        request: FullModelRequest,
        *,
        prepared_image: _PreparedImageURL,
        reduced_mode: bool,
    ) -> list[Mapping[str, Any]]:
        valid_labels = _resolved_request_valid_labels(request)
        is_android = str(request.benchmark) == "AndroidControl"
        if reduced_mode:
            system_prompt = self.config.reduced_model_system_prompt
        elif is_android:
            # Use native Qwen3-VL mobile_use format for AndroidControl.
            # Model outputs <thinking>...<tool_call>mobile_use(...)</tool_call>
            # with 0-999 coordinates, which parse_model_action_output converts.
            system_prompt = ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT
        else:
            system_prompt = self.config.full_model_system_prompt
        labels_block = ""
        if valid_labels:
            labels_block = f"\n{_format_valid_action_label_block(valid_labels)}"
        if reduced_mode:
            response_instruction = "Return exactly one label from the valid action label list."
            user_text = (
                f"Benchmark: {request.benchmark}\n"
                f"Reason: {request.reason}\n"
                f"{request.prompt_text}"
                f"{labels_block}\n\n"
                f"{response_instruction}"
            )
        elif is_android:
            # Native Qwen3-VL mobile_use format: clean prompt matching model's pre-training.
            # No Benchmark/Reason prefix — matches reference evaluation (step05_AC_Qwen3VL).
            user_text = (
                f"{request.prompt_text}"
                "\nOutput exactly one next action (mobile_use) for the current screenshot."
            )
        else:
            response_instruction = "Return exactly one JSON object."
            user_text = (
                f"Benchmark: {request.benchmark}\n"
                f"Reason: {request.reason}\n"
                f"{request.prompt_text}"
                f"{labels_block}\n\n"
                f"{response_instruction}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": prepared_image.url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _build_disambiguation_messages(
        self,
        request: DisambiguationRequest,
        *,
        prepared_image: _PreparedImageURL,
    ) -> list[Mapping[str, Any]]:
        user_text = (
            f"{request.prompt.text}\n"
            f"{_format_valid_action_label_block(request.valid_labels)}\n"
            "Return exactly one label from the valid action label list."
        )
        return [
            {"role": "system", "content": self.config.local_disambiguation_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": prepared_image.url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _prepare_image_url(
        self,
        screenshot: ScreenshotAsset,
        *,
        crop_bbox: BBox | None,
        max_pixels: int,
    ) -> _PreparedImageURL:
        input_mode = str(self.config.image_input_mode).strip().lower()
        if input_mode not in {"auto", "base64_data_url", "file_url"}:
            raise ValueError(
                f"Unsupported image_input_mode={self.config.image_input_mode!r}; expected auto, base64_data_url, or file_url."
            )
        if input_mode == "file_url" and not bool(self.config.allow_local_file_urls):
            raise ValueError(
                "image_input_mode='file_url' requires allow_local_file_urls=True. "
                "Use 'base64_data_url' or 'auto' otherwise."
            )
        if input_mode in {"auto", "file_url"} and bool(self.config.allow_local_file_urls):
            passthrough_issue = _file_url_passthrough_issue(
                screenshot,
                requested_crop_bbox=crop_bbox,
                max_pixels=max_pixels,
            )
            if passthrough_issue is None and screenshot.path is not None:
                return _PreparedImageURL(
                    url=_path_to_file_url(screenshot.path),
                    width=max(1, int(screenshot.width or 1)),
                    height=max(1, int(screenshot.height or 1)),
                )
            if input_mode == "file_url":
                raise ValueError(
                    "image_input_mode='file_url' cannot preserve shared preprocessing semantics for this request: "
                    f"{passthrough_issue}. Use 'base64_data_url' or 'auto' instead."
                )
        image_module = _import_image_module()
        image = _load_image(screenshot, image_module)
        if crop_bbox is not None:
            image = image.crop(
                _normalize_crop_bbox_for_processor(
                    crop_bbox,
                    image_size=image.size,
                )
            )
        image = _pad_image_to_safe_aspect_ratio(image, image_module=image_module)
        width, height = image.size
        max_pixels_value = max(1, int(max_pixels))
        if width * height > max_pixels_value:
            scale = (float(max_pixels_value) / float(width * height)) ** 0.5
            image = image.resize(
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        final_width, final_height = image.size
        return _PreparedImageURL(
            url=f"data:image/png;base64,{encoded}",
            width=int(final_width),
            height=int(final_height),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        request_url = _join_api_url(self.config.api_base, path)
        headers = {
            "Accept": "application/json",
            **{str(key): str(value) for key, value in dict(self.config.extra_request_headers).items()},
        }
        api_key = str(self.config.api_key or "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(dict(payload)).encode("utf-8")
        attempts = max(0, int(self.config.request_retries)) + 1
        ssl_context = None if self.config.verify_ssl else ssl._create_unverified_context()
        timeout_seconds = max(float(self.config.timeout_seconds), float(self.config.connect_timeout_seconds))
        parsed_url = urlparse(request_url)
        bypass_proxy = str(parsed_url.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}
        opener = build_opener(ProxyHandler({})) if bypass_proxy else None
        last_error: Exception | None = None
        for attempt_index in range(attempts):
            try:
                request = Request(request_url, data=body, headers=headers, method=method.upper())
                if opener is not None:
                    with opener.open(request, timeout=timeout_seconds) as response:
                        raw_body = response.read().decode("utf-8")
                else:
                    with urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
                        raw_body = response.read().decode("utf-8")
                decoded = json.loads(raw_body)
                if not isinstance(decoded, Mapping):
                    raise RuntimeError(f"Expected JSON object from {request_url}, received: {decoded!r}")
                return decoded
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code >= 500 or exc.code == 429
                last_error = RuntimeError(
                    f"vLLM request to {request_url} failed with HTTP {exc.code}: {error_body}"
                )
                if not retryable or attempt_index + 1 >= attempts:
                    raise last_error
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt_index + 1 >= attempts:
                    break
            time.sleep(float(self.config.retry_backoff_seconds) * float(attempt_index + 1))
        raise RuntimeError(f"vLLM request to {request_url} failed after {attempts} attempts.") from last_error

    def _parse_action_output(
        self,
        output_text: str,
        *,
        request: FullModelRequest | None = None,
    ) -> CanonicalAction:
        """Parse Qwen3-VL output into a CanonicalAction.

        For AndroidControl: uses native mobile_use format with 0-999 coordinate scale.
        parse_model_action_output converts 0-999 back to absolute pixel coordinates.

        For other benchmarks: falls back to JSON object parse.
        """
        is_android = request is not None and str(request.benchmark) == "AndroidControl"
        if is_android:
            # Qwen3-VL outputs <thinking>...<tool_call>mobile_use(...)</tool_call>
            # with 0-999 scale coordinates. Convert to absolute pixels via adapter.
            w = int(request.screenshot.width or 1080)
            h = int(request.screenshot.height or 1920)
            try:
                action = parse_model_action_output(
                    output_text,
                    screenshot_width=w,
                    screenshot_height=h,
                    roi_bbox=request.roi_bbox,
                    assume_android_coordinate_scale=True,
                )
                if action is not None:
                    return action
            except Exception:
                pass  # fall through to JSON parse as last resort

        # Non-AndroidControl or fallback: try JSON parse
        payload = _extract_json_object(output_text)
        if payload is not None:
            action = _payload_to_action(payload)
            if _is_action_semantically_complete(action):
                return action

        stripped = output_text.strip()
        match = ACTION_CALL_RE.match(stripped)
        if match:
            action = _action_from_call(match.group("action_type"), match.group("payload"))
            if _is_action_semantically_complete(action):
                return action

        raise ValueError(f"Could not parse model output into CanonicalAction: {output_text!r}")

    def _parse_reduced_request_output(
        self,
        output_text: str,
        *,
        request: FullModelRequest,
    ) -> CanonicalAction | None:
        valid_labels = _resolved_request_valid_labels(request)
        if not valid_labels:
            return None
        label = self._parse_disambiguation_output(
            output_text,
            valid_labels=valid_labels,
        )
        if label is None:
            return None
        return _resolved_request_reduced_label_action_map(request).get(label)

    def _parse_disambiguation_output(
        self,
        output_text: str,
        *,
        valid_labels: Sequence[str],
    ) -> str | None:
        return _select_valid_label_from_output(output_text, valid_labels=valid_labels)


class ShardedVLLMOpenAIBackend:
    """Client-side sharding across multiple OpenAI-compatible replicas."""

    def __init__(
        self,
        model_spec: ModelRuntimeSpec,
        *,
        config: ModelServiceConfig | None = None,
        eager_load: bool = False,
    ) -> None:
        self.model_spec = model_spec
        self.config = config or ModelServiceConfig()
        self.api_bases = _resolve_service_api_bases(self.config)
        if len(self.api_bases) < 2:
            raise ValueError("ShardedVLLMOpenAIBackend requires at least two configured api_bases.")
        self._backends = tuple(
            VLLMOpenAIBackend(
                model_spec,
                config=replace(self.config, api_base=api_base, api_bases=()),
                eager_load=False,
            )
            for api_base in self.api_bases
        )
        self._service_ready = False
        if eager_load:
            self._ensure_service_ready()

    def generate(self, request: FullModelRequest) -> FullModelResponse:
        return self._backend_for_observation_id(request.observation_id).generate(request)

    def generate_batch(self, requests: Sequence[FullModelRequest]) -> tuple[FullModelResponse, ...]:
        return self._run_sharded_batches(requests, batch_method_name="generate_batch")

    def estimate_request_cost(self, request: FullModelRequest) -> Mapping[str, int]:
        return self._backends[0].estimate_request_cost(request)

    def disambiguate(self, request: DisambiguationRequest) -> DisambiguationResponse:
        return self._backend_for_observation_id(request.observation_id).disambiguate(request)

    def disambiguate_batch(
        self,
        requests: Sequence[DisambiguationRequest],
    ) -> tuple[DisambiguationResponse, ...]:
        return self._run_sharded_batches(requests, batch_method_name="disambiguate_batch")

    def _ensure_service_ready(self) -> None:
        if self._service_ready:
            return
        for backend in self._backends:
            backend._ensure_service_ready()
        self._service_ready = True

    def _backend_for_observation_id(self, observation_id: str) -> VLLMOpenAIBackend:
        return self._backends[self._backend_index_for_observation_id(observation_id)]

    def _backend_index_for_observation_id(self, observation_id: str) -> int:
        return _stable_shard_index(str(observation_id), len(self._backends))

    def _run_sharded_batches(self, requests: Sequence[Any], *, batch_method_name: str) -> tuple[Any, ...]:
        if not requests:
            return tuple()
        self._ensure_service_ready()
        indexed_requests_by_shard: dict[int, list[tuple[int, Any]]] = {}
        for index, request in enumerate(requests):
            shard_index = self._backend_index_for_observation_id(str(request.observation_id))
            indexed_requests_by_shard.setdefault(shard_index, []).append((index, request))

        results: list[Any | None] = [None] * len(requests)
        shard_groups = tuple(indexed_requests_by_shard.items())
        max_workers = max(1, min(len(shard_groups), int(self.config.client_max_parallel_requests)))
        per_shard_parallelism = max(1, int(self.config.client_max_parallel_requests) // max(1, len(shard_groups)))

        def run_group(item: tuple[int, list[tuple[int, Any]]]) -> tuple[tuple[int, Any], ...]:
            shard_index, indexed_group_requests = item
            backend = self._backends[shard_index]
            effective_backend = backend
            if int(backend.config.client_max_parallel_requests) != per_shard_parallelism:
                effective_backend = VLLMOpenAIBackend(
                    self.model_spec,
                    config=replace(
                        backend.config,
                        client_max_parallel_requests=per_shard_parallelism,
                    ),
                    eager_load=False,
                )
                effective_backend._service_ready = backend._service_ready
            batch_requests = tuple(request for _, request in indexed_group_requests)
            batch_method = getattr(effective_backend, batch_method_name)
            shard_responses = tuple(batch_method(batch_requests))
            if len(shard_responses) != len(batch_requests):
                raise ValueError(
                    f"Shard backend {effective_backend.config.api_base!r} returned {len(shard_responses)} responses "
                    f"for {len(batch_requests)} requests."
                )
            return tuple(zip((index for index, _ in indexed_group_requests), shard_responses))

        if max_workers == 1:
            completed_groups = tuple(run_group(group) for group in shard_groups)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                completed_groups = tuple(executor.map(run_group, shard_groups))

        for completed_group in completed_groups:
            for index, response in completed_group:
                results[index] = response
        if any(response is None for response in results):
            raise RuntimeError("Shard dispatch did not populate every response slot.")
        return tuple(results)


def build_backend(
    model_spec: ModelRuntimeSpec,
    *,
    config: ModelServiceConfig | None = None,
    eager_load: bool = False,
) -> BenchmarkFaithfulModel:
    """Build the configured runtime backend.

    Routes to ShardedVLLMOpenAIBackend / VLLMOpenAIBackend for service kinds,
    or QwenLoRABackend for local transformers kinds.
    """
    resolved_config = config or ModelServiceConfig()
    backend_kind = resolve_backend_kind(model_spec, config=resolved_config)
    if backend_kind in SERVICE_BACKEND_KINDS:
        if len(_resolve_service_api_bases(resolved_config)) > 1:
            return ShardedVLLMOpenAIBackend(model_spec, config=resolved_config, eager_load=eager_load)
        return VLLMOpenAIBackend(model_spec, config=resolved_config, eager_load=eager_load)
    if backend_kind == "visionzip":
        from guiaccel.model.visionzip_backend import VisionZipConfig, VisionZipQwenBackend

        extra = dict(resolved_config.extra) if resolved_config.extra else {}
        vz_config = VisionZipConfig(
            dominant_ratio=float(extra.get("dominant_ratio", 0.65)),
            contextual_ratio=float(extra.get("contextual_ratio", 0.05)),
        )
        qwen_config = LocalQwenBackendConfig(
            visible_cuda_devices=tuple(int(d) for d in resolved_config.visible_cuda_devices),
            device_map=resolved_config.device_map,
            torch_dtype=str(resolved_config.torch_dtype),
            attn_implementation=resolved_config.attn_implementation,
            trust_remote_code=bool(resolved_config.trust_remote_code),
            processor_from_adapter_first=bool(resolved_config.processor_from_adapter_first),
            default_image_max_pixels=int(resolved_config.default_image_max_pixels),
            allow_cpu=bool(resolved_config.allow_cpu),
            full_model_system_prompt=str(resolved_config.full_model_system_prompt),
            reduced_model_system_prompt=str(resolved_config.reduced_model_system_prompt),
            local_disambiguation_system_prompt=str(resolved_config.local_disambiguation_system_prompt),
        )
        return VisionZipQwenBackend(model_spec, config=qwen_config, visionzip_config=vz_config, eager_load=eager_load)
    if backend_kind in LOCAL_TRANSFORMERS_BACKEND_KINDS:
        qwen_config = LocalQwenBackendConfig(
            visible_cuda_devices=tuple(int(device) for device in resolved_config.visible_cuda_devices),
            device_map=resolved_config.device_map,
            torch_dtype=str(resolved_config.torch_dtype),
            attn_implementation=resolved_config.attn_implementation,
            trust_remote_code=bool(resolved_config.trust_remote_code),
            processor_from_adapter_first=bool(resolved_config.processor_from_adapter_first),
            default_image_max_pixels=int(resolved_config.default_image_max_pixels),
            allow_cpu=bool(resolved_config.allow_cpu),
            full_model_system_prompt=str(resolved_config.full_model_system_prompt),
            reduced_model_system_prompt=str(resolved_config.reduced_model_system_prompt),
            local_disambiguation_system_prompt=str(resolved_config.local_disambiguation_system_prompt),
        )
        return build_qwen_backend(model_spec, config=qwen_config, eager_load=eager_load)
    if backend_kind in DIVPRUNE_BACKEND_KINDS:
        from guiaccel.model.divprune_backend import DivPruneQwenBackend
        qwen_config = LocalQwenBackendConfig(
            visible_cuda_devices=tuple(int(device) for device in resolved_config.visible_cuda_devices),
            device_map=resolved_config.device_map,
            torch_dtype=str(resolved_config.torch_dtype),
            attn_implementation=resolved_config.attn_implementation,
            trust_remote_code=bool(resolved_config.trust_remote_code),
            processor_from_adapter_first=bool(resolved_config.processor_from_adapter_first),
            default_image_max_pixels=int(resolved_config.default_image_max_pixels),
            allow_cpu=bool(resolved_config.allow_cpu),
            full_model_system_prompt=str(resolved_config.full_model_system_prompt),
            reduced_model_system_prompt=str(resolved_config.reduced_model_system_prompt),
            local_disambiguation_system_prompt=str(resolved_config.local_disambiguation_system_prompt),
        )
        return DivPruneQwenBackend(
            model_spec,
            config=qwen_config,
            keep_ratio=float(resolved_config.keep_ratio or 0.098),
            eager_load=eager_load,
        )
    raise ValueError(f"Unsupported runtime backend kind: {backend_kind}")


def resolve_backend_kind(
    model_spec: ModelRuntimeSpec,
    *,
    config: ModelServiceConfig | None = None,
) -> str:
    resolved_config = config or ModelServiceConfig()
    return str(resolved_config.kind or model_spec.backend or "vllm").strip().lower()


def require_local_worker_backend(
    model_spec: ModelRuntimeSpec,
    *,
    config: ModelServiceConfig | None,
    context: str,
) -> None:
    backend_kind = resolve_backend_kind(model_spec, config=config)
    if backend_kind in LOCAL_TRANSFORMERS_BACKEND_KINDS or backend_kind in DIVPRUNE_BACKEND_KINDS:
        return
    raise ValueError(
        f"{context} requires a local transformers runtime backend when worker_gpus are configured; "
        f"resolved backend kind {backend_kind!r} uses shared service semantics instead. "
        "Clear worker_gpus for vLLM/OpenAI-compatible serving."
    )


def _resolve_full_token_usage(
    request: FullModelRequest,
    *,
    response_payload: Mapping[str, Any],
    prepared_image: _PreparedImageURL,
) -> TokenUsage:
    usage = response_payload.get("usage")
    if not isinstance(usage, Mapping):
        return request.estimated_token_usage
    prompt_total = int(usage.get("prompt_tokens") or 0)
    generated_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    visual_tokens = estimate_visual_tokens(prepared_image.width, prepared_image.height)
    text_tokens = max(0, prompt_total - visual_tokens) if prompt_total > 0 else int(request.estimated_token_usage.prompt_tokens)
    return TokenUsage(
        prompt_tokens=text_tokens,
        visual_tokens=visual_tokens,
        generated_tokens=generated_tokens or int(request.estimated_token_usage.generated_tokens),
        full_model_calls=1,
    )


def _resolve_disambiguation_token_usage(
    request: DisambiguationRequest,
    *,
    response_payload: Mapping[str, Any],
    prepared_image: _PreparedImageURL,
) -> TokenUsage:
    usage = response_payload.get("usage")
    if not isinstance(usage, Mapping):
        return TokenUsage(
            prompt_tokens=int(request.prompt.token_count),
            visual_tokens=int(request.visual.estimated_visual_tokens),
            generated_tokens=max(1, int(request.max_new_tokens // 2)),
            local_model_calls=1,
        )
    prompt_total = int(usage.get("prompt_tokens") or 0)
    generated_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    visual_tokens = estimate_visual_tokens(prepared_image.width, prepared_image.height)
    text_tokens = max(0, prompt_total - visual_tokens) if prompt_total > 0 else int(request.prompt.token_count)
    return TokenUsage(
        prompt_tokens=text_tokens,
        visual_tokens=visual_tokens,
        generated_tokens=generated_tokens or max(1, int(request.max_new_tokens // 2)),
        local_model_calls=1,
    )


def _resolve_image_geometry(
    screenshot: ScreenshotAsset,
    *,
    crop_bbox: BBox | None,
    max_pixels: int,
) -> tuple[BBox | None, int, int]:
    width = int(screenshot.width or 1)
    height = int(screenshot.height or 1)
    if crop_bbox is not None:
        left, top, right, bottom = crop_bbox
        width = max(1, int(right - left))
        height = max(1, int(bottom - top))
    max_pixels_value = max(1, int(max_pixels))
    if width * height > max_pixels_value:
        scale = (float(max_pixels_value) / float(width * height)) ** 0.5
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return crop_bbox, width, height


def _chat_completion_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"vLLM chat completion payload had no choices: {payload!r}")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise RuntimeError(f"vLLM chat completion choice was not an object: {first_choice!r}")
    message = first_choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        return _coerce_message_content(content)
    text = first_choice.get("text")
    if isinstance(text, str):
        return text.strip()
    raise RuntimeError(f"Unable to extract text from vLLM chat completion payload: {payload!r}")


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def _join_api_url(api_base: str, path: str) -> str:
    base = str(api_base).rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def _resolve_service_api_bases(config: ModelServiceConfig) -> tuple[str, ...]:
    raw_api_bases = config.api_bases
    if isinstance(raw_api_bases, str):
        candidate_api_bases = (raw_api_bases,)
    else:
        candidate_api_bases = tuple(raw_api_bases)
    configured_api_bases = tuple(
        str(api_base).strip()
        for api_base in candidate_api_bases
        if str(api_base).strip()
    )
    if configured_api_bases:
        return configured_api_bases
    api_base = str(config.api_base).strip()
    if not api_base:
        raise ValueError("Service backend requires at least one non-empty api_base.")
    return (api_base,)


def _stable_shard_index(key: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(shard_count)


def _path_to_file_url(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return f"file://{quote(str(resolved))}"


def _file_url_passthrough_issue(
    screenshot: ScreenshotAsset,
    *,
    requested_crop_bbox: BBox | None,
    max_pixels: int,
) -> str | None:
    if screenshot.path is None:
        return "the screenshot has no local file path"
    if requested_crop_bbox is not None:
        return "ROI cropping is requested"
    width = int(screenshot.width or 0)
    height = int(screenshot.height or 0)
    if width <= 0 or height <= 0:
        return "image dimensions are unavailable"
    if max(width, height) / float(min(width, height)) >= float(QWEN_PROCESSOR_SAFE_ASPECT_RATIO):
        return "processor-safe aspect-ratio padding would modify the image"
    if width * height > max(1, int(max_pixels)):
        return "pixel-budget resizing would modify the image"
    return None


def _import_image_module() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise DependencyUnavailableError(
            "Pillow is required for local ROI crop/base64 preparation in the vLLM backend."
        ) from exc
    return Image


__all__ = [
    "DependencyUnavailableError",
    "DIVPRUNE_BACKEND_KINDS",
    "LOCAL_TRANSFORMERS_BACKEND_KINDS",
    "ModelServiceConfig",
    "SERVICE_BACKEND_KINDS",
    "ShardedVLLMOpenAIBackend",
    "VLLMOpenAIBackend",
    "build_backend",
    "require_local_worker_backend",
    "resolve_backend_kind",
]
