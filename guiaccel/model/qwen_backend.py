"""Concrete Qwen3-VL + LoRA runtime backend."""

from __future__ import annotations

import json
import math
import os
import re
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from guiaccel.routing.execution import (
    BatchLocalDisambiguationModel,
    DisambiguationRequest,
    DisambiguationResponse,
    PromptPlan,
)
from guiaccel.routing.common import TokenUsage, estimate_visual_tokens
from guiaccel.routing.fallback import (
    BenchmarkFaithfulModel,
    FullModelRequest,
    FullModelResponse,
    QwenLoRAModelSpec,
)
from guiaccel.types import BBox, CanonicalAction, ScreenshotAsset
from guiaccel.model.maiui_action_adapter import parse_model_action_output

ALLOWED_CUDA_DEVICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
ACTION_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
ACTION_CALL_RE = re.compile(
    r"(?P<action_type>CLICK|TYPE|SCROLL|NAV|TERMINATE|WAIT|LONG_PRESS)"
    r"(?:\[(?P<payload>.*)\])?$",
    flags=re.IGNORECASE,
)
QWEN_PROCESSOR_ABSOLUTE_ASPECT_RATIO_LIMIT = 200.0
QWEN_PROCESSOR_SAFE_ASPECT_RATIO = 180.0

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


class DependencyUnavailableError(RuntimeError):
    """Raised when the Qwen runtime dependencies are not installed."""


def _resolve_qwen3_vl_model(model: Any) -> Any:
    """Return Qwen3VLModel (feat_host) regardless of PeftModel wrapping."""
    outer = model
    if hasattr(outer, "base_model"):
        inner = outer.base_model
        if hasattr(inner, "get_image_features"):
            return inner
        if hasattr(inner, "model"):
            inner = inner.model
            if hasattr(inner, "get_image_features"):
                return inner
        outer = inner
    if hasattr(outer, "model") and hasattr(outer.model, "get_image_features"):
        return outer.model
    if hasattr(outer, "get_image_features"):
        return outer
    raise AttributeError("Cannot resolve Qwen3VLModel.get_image_features host")


@dataclass(frozen=True)
class QwenBackendConfig:
    """Runtime configuration for the concrete Qwen backend."""

    visible_cuda_devices: tuple[int, ...] = ALLOWED_CUDA_DEVICES
    device_map: int | str | None = 0
    torch_dtype: str = "bfloat16"
    attn_implementation: str | Mapping[str, str] | None = None
    trust_remote_code: bool = False
    processor_from_adapter_first: bool = True
    default_image_max_pixels: int = 1_000_000
    allow_cpu: bool = False
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


@dataclass(frozen=True)
class AdapterCompatibilityReport:
    """Static compatibility report for a LoRA adapter bundle."""

    adapter_path: str
    base_model_path: str
    compatible: bool
    adapter_exists: bool
    base_model_exists: bool
    adapter_peft_version: str | None
    base_model_transformers_version: str | None
    base_model_architecture: str | None
    target_modules: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RuntimeDependencies:
    torch: Any
    image_module: Any
    processor_cls: Any
    model_cls: Any
    peft_model_cls: Any


def resolve_visible_cuda_devices(
    requested_devices: Sequence[int] | None = None,
    *,
    env_value: str | None = None,
) -> tuple[int, ...]:
    """Resolve and validate the CUDA device set under the repository restriction."""

    raw_value = env_value if env_value is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is not None and raw_value.strip():
        devices = tuple(_parse_device_list(raw_value))
    else:
        devices = tuple(int(device) for device in (requested_devices or ALLOWED_CUDA_DEVICES))
    if not devices:
        return tuple()
    invalid = [device for device in devices if device not in ALLOWED_CUDA_DEVICES]
    if invalid:
        raise ValueError(
            f"CUDA devices must be drawn only from {ALLOWED_CUDA_DEVICES}; got invalid devices {invalid}."
        )
    return tuple(dict.fromkeys(devices))


def inspect_lora_adapter_compatibility(
    adapter_path: str | Path,
    *,
    base_model_path: str | Path | None = None,
) -> AdapterCompatibilityReport:
    """Inspect the adapter metadata without importing the heavy runtime stack."""

    resolved_adapter_path = Path(adapter_path).resolve()
    adapter_exists = resolved_adapter_path.exists()
    adapter_config = _load_optional_json(resolved_adapter_path / "adapter_config.json")
    base_model_reference = base_model_path or adapter_config.get("base_model_name_or_path")
    resolved_base_model_path = Path(base_model_reference).resolve() if base_model_reference else None
    base_model_exists = resolved_base_model_path is not None and resolved_base_model_path.exists()
    base_model_config = _load_optional_json(resolved_base_model_path / "config.json") if base_model_exists else {}

    missing_files: list[str] = []
    for required_path in (
        resolved_adapter_path / "adapter_config.json",
        resolved_adapter_path / "adapter_model.safetensors",
    ):
        if not required_path.exists():
            missing_files.append(str(required_path))
    if resolved_base_model_path is None:
        missing_files.append("missing_base_model_path")
    elif base_model_exists and not (resolved_base_model_path / "config.json").exists():
        missing_files.append(str(resolved_base_model_path / "config.json"))

    warnings: list[str] = []
    model_type = str(base_model_config.get("model_type") or "")
    architecture = None
    architectures = base_model_config.get("architectures")
    if isinstance(architectures, list) and architectures:
        architecture = str(architectures[0])
    if model_type and model_type != "qwen3_vl":
        warnings.append(f"Unexpected base model_type={model_type}.")
    if architecture and architecture != "Qwen3VLForConditionalGeneration":
        warnings.append(f"Unexpected base architecture={architecture}.")
    transformers_version = base_model_config.get("transformers_version")
    if transformers_version and "4.57" not in str(transformers_version):
        warnings.append(
            f"Base model metadata expects transformers 4.57-era support, got {transformers_version!s}."
        )
    peft_version = adapter_config.get("peft_version")
    if peft_version and tuple(int(part) for part in str(peft_version).split(".")[:2]) < (0, 18):
        warnings.append(f"Adapter was exported with PEFT {peft_version}; verify loader compatibility.")

    compatible = adapter_exists and base_model_exists and not missing_files and not warnings
    if adapter_exists and base_model_exists and not missing_files and not compatible:
        compatible = True

    target_modules = adapter_config.get("target_modules")
    return AdapterCompatibilityReport(
        adapter_path=str(resolved_adapter_path),
        base_model_path=str(resolved_base_model_path) if resolved_base_model_path is not None else "",
        compatible=compatible,
        adapter_exists=adapter_exists,
        base_model_exists=base_model_exists,
        adapter_peft_version=str(peft_version) if peft_version is not None else None,
        base_model_transformers_version=(
            str(transformers_version) if transformers_version is not None else None
        ),
        base_model_architecture=architecture,
        target_modules=tuple(str(item) for item in target_modules) if isinstance(target_modules, list) else (),
        missing_files=tuple(missing_files),
        warnings=tuple(warnings),
    )


class QwenLoRABackend(BenchmarkFaithfulModel, BatchLocalDisambiguationModel):
    """Concrete Qwen3-VL-8B-Instruct backend with optional LoRA loading."""

    def __init__(
        self,
        model_spec: QwenLoRAModelSpec,
        *,
        config: QwenBackendConfig | None = None,
        eager_load: bool = False,
    ) -> None:
        self.model_spec = model_spec
        self.config = config or QwenBackendConfig()
        self.visible_cuda_devices = resolve_visible_cuda_devices(self.config.visible_cuda_devices)
        self._dependencies: _RuntimeDependencies | None = None
        self._processor: Any = None
        self._model: Any = None
        self._model_device: Any = None
        self._load_error: Exception | None = None
        self._model_load_time_ms: float = 0.0
        self._last_gen_timing: dict[str, float] = {}
        if eager_load:
            self._ensure_model_loaded()

    def generate(self, request: FullModelRequest) -> FullModelResponse:
        return self.generate_batch((request,))[0]

    def generate_batch(self, requests: Sequence[FullModelRequest]) -> tuple[FullModelResponse, ...]:
        if not requests:
            return tuple()
        dependencies = self._ensure_model_loaded()
        self._validate_generation_batch(requests)
        self._last_gen_timing = {}

        t0 = time.perf_counter()
        prepared: list[tuple[Any, list[Mapping[str, Any]]]] = []
        image_prep_accum = 0.0
        message_build_accum = 0.0
        for request in requests:
            t_img = time.perf_counter()
            image = self._prepare_image(
                request.screenshot,
                crop_bbox=request.roi_bbox,
                max_pixels=request.image_max_pixels or self.config.default_image_max_pixels,
                dependencies=dependencies,
            )
            image_prep_accum += (time.perf_counter() - t_img) * 1000.0
            t_msg = time.perf_counter()
            messages = self._build_full_model_messages(request, image=image)
            message_build_accum += (time.perf_counter() - t_msg) * 1000.0
            prepared.append((image, messages))

        t1 = time.perf_counter()
        output_texts, input_token_counts, generated_token_counts = self._run_generation_batch(
            [messages for _, messages in prepared],
            max_new_tokens=int(requests[0].max_new_tokens),
            temperature=float(requests[0].temperature),
            top_p=float(requests[0].top_p),
            repetition_penalty=float(requests[0].repetition_penalty),
        )
        total_generation_ms = (time.perf_counter() - t1) * 1000.0
        total_latency_ms = total_generation_ms
        per_request_latency_ms = total_latency_ms / float(len(requests))

        responses: list[FullModelResponse] = []
        for request, (image, _), output_text, input_token_count, generated_token_count in zip(
            requests,
            prepared,
            output_texts,
            input_token_counts,
            generated_token_counts,
        ):
            t_parse = time.perf_counter()
            action = self._parse_reduced_request_output(output_text, request=request)
            if action is None:
                try:
                    action = self._parse_action_output(output_text, benchmark=request.benchmark, request=request)
                except ValueError:
                    action = None
            output_parse_ms = (time.perf_counter() - t_parse) * 1000.0
            visual_tokens = estimate_visual_tokens(image.width, image.height)
            gen_timing = self._last_gen_timing
            timing: dict[str, float] = {
                "image_prep_ms": image_prep_accum,
                "message_build_ms": message_build_accum,
                "tokenization_ms": gen_timing.get("tokenization_ms", 0.0),
                "gpu_transfer_ms": gen_timing.get("gpu_transfer_ms", 0.0),
                "model_generate_ms": gen_timing.get("model_generate_ms", 0.0),
                "output_decode_ms": gen_timing.get("output_decode_ms", 0.0),
                "output_parse_ms": output_parse_ms,
                "prefill_ms": gen_timing.get("prefill_ms", 0.0),
                "decode_ms": gen_timing.get("decode_ms", 0.0),
                "llm_forward_ms": gen_timing.get("llm_forward_ms", 0.0),
                "decode_steps": gen_timing.get("decode_steps", 0.0),
                "total_generation_ms": per_request_latency_ms,
                "total_e2e_step_ms": (
                    image_prep_accum
                    + message_build_accum
                    + gen_timing.get("tokenization_ms", 0.0)
                    + gen_timing.get("gpu_transfer_ms", 0.0)
                    + gen_timing.get("model_generate_ms", 0.0)
                    + gen_timing.get("output_decode_ms", 0.0)
                    + output_parse_ms
                ),
            }
            responses.append(
                FullModelResponse(
                    action=action,
                    token_usage=TokenUsage(
                        prompt_tokens=int(input_token_count),
                        visual_tokens=visual_tokens,
                        generated_tokens=int(generated_token_count),
                        full_model_calls=1,
                    ),
                    latency_ms=per_request_latency_ms,
                    raw_output=output_text,
                    model_name=self.model_spec.model_name,
                    timing=timing,
                )
            )
        return tuple(responses)

    def estimate_request_cost(self, request: FullModelRequest) -> Mapping[str, int]:
        dependencies = self._ensure_model_loaded()
        image = self._prepare_image(
            request.screenshot,
            crop_bbox=request.roi_bbox,
            max_pixels=request.image_max_pixels or self.config.default_image_max_pixels,
            dependencies=dependencies,
        )
        messages = self._build_full_model_messages(request, image=image)
        batch = self._processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"truncation": True, "padding": True},
        )
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            text_tokens = int(batch["input_ids"].shape[-1])
        else:
            text_tokens = int(attention_mask.sum().item())
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is None:
            return {
                "text_tokens": text_tokens,
                "image_count": 0,
                "visual_tokens": 0,
            }
        merge_size = int(getattr(self._processor.image_processor, "merge_size", 2))
        visual_tokens = int((image_grid_thw.prod(-1) // (merge_size**2)).sum().item())
        return {
            "text_tokens": text_tokens,
            "image_count": int(image_grid_thw.shape[0]),
            "visual_tokens": visual_tokens,
        }

    def disambiguate(self, request: DisambiguationRequest) -> DisambiguationResponse:
        return self.disambiguate_batch((request,))[0]

    def disambiguate_batch(
        self,
        requests: Sequence[DisambiguationRequest],
    ) -> tuple[DisambiguationResponse, ...]:
        if not requests:
            return tuple()
        dependencies = self._ensure_model_loaded()
        self._validate_disambiguation_batch(requests)
        prepared: list[tuple[Any, list[Mapping[str, Any]]]] = []
        for request in requests:
            image = self._prepare_image(
                request.visual.screenshot,
                crop_bbox=request.visual.crop_bbox,
                max_pixels=request.visual.pixel_budget or self.config.default_image_max_pixels,
                dependencies=dependencies,
            )
            prepared.append((image, self._build_disambiguation_messages(request, image=image)))
        started_at = time.perf_counter()
        output_texts, input_token_counts, generated_token_counts = self._run_generation_batch(
            [messages for _, messages in prepared],
            max_new_tokens=int(requests[0].max_new_tokens),
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0
        per_request_latency_ms = total_latency_ms / float(len(requests))
        responses: list[DisambiguationResponse] = []
        for request, (image, _), output_text, input_token_count, generated_token_count in zip(
            requests,
            prepared,
            output_texts,
            input_token_counts,
            generated_token_counts,
        ):
            selected_label = self._parse_disambiguation_output(output_text, valid_labels=request.valid_labels)
            responses.append(
                DisambiguationResponse(
                    selected_label=selected_label,
                    valid=selected_label in request.valid_labels if selected_label is not None else False,
                    raw_output=output_text,
                    token_usage=TokenUsage(
                        prompt_tokens=input_token_count,
                        visual_tokens=estimate_visual_tokens(image.width, image.height),
                        generated_tokens=generated_token_count,
                        local_model_calls=1,
                    ),
                    latency_ms=per_request_latency_ms,
                )
            )
        return tuple(responses)

    def _ensure_model_loaded(self) -> _RuntimeDependencies:
        if self._load_error is not None:
            raise RuntimeError("Qwen backend initialization failed earlier.") from self._load_error
        if self._model is not None and self._processor is not None and self._dependencies is not None:
            return self._dependencies

        try:
            dependencies = _import_runtime_dependencies()
            load_start = time.perf_counter()
            if self.visible_cuda_devices:
                os.environ.setdefault(
                    "CUDA_VISIBLE_DEVICES",
                    ",".join(str(device) for device in self.visible_cuda_devices),
                )

            torch = dependencies.torch
            if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
                if not self.config.allow_cpu:
                    raise RuntimeError(
                        "CUDA is required for the Qwen backend. "
                        "Use only GPUs 0,1,2,3 or reconfigure QwenBackendConfig.allow_cpu explicitly."
                    )
                device_map: str | None = None
            else:
                device_map = _resolve_single_device_map(self.config.device_map)

            processor_source = self._processor_source_path()
            self._processor = dependencies.processor_cls.from_pretrained(
                str(processor_source),
                trust_remote_code=self.config.trust_remote_code,
            )
            _configure_processor_for_generation(self._processor)
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": self.config.trust_remote_code,
                "low_cpu_mem_usage": True,
                "torch_dtype": _resolve_torch_dtype(torch, self.config.torch_dtype),
            }
            if self.config.attn_implementation is not None:
                model_kwargs["attn_implementation"] = self.config.attn_implementation
            self._model = dependencies.model_cls.from_pretrained(
                str(self._base_model_path()),
                **model_kwargs,
            )
            adapter_path = self._adapter_path()
            if adapter_path is not None:
                self._model = dependencies.peft_model_cls.from_pretrained(
                    self._model,
                    str(adapter_path),
                    is_trainable=False,
                )
            self._model.eval()
            self._dependencies = dependencies
            self._model_device = getattr(self._model, "device", None)
            if self._model_device is None:
                parameter = next(self._model.parameters())
                self._model_device = parameter.device
            self._model_load_time_ms = (time.perf_counter() - load_start) * 1000.0
            return dependencies
        except Exception as exc:  # pragma: no cover - exercised only in runtime environments.
            self._load_error = exc
            raise

    def _install_forward_timing_patch(self, feat_host: Any) -> None:
        if getattr(feat_host, "_baseline_timing_patch_installed", False):
            return

        original_forward = feat_host.forward

        def patched_forward(self_model: Any, *args: Any, **kwargs: Any) -> Any:
            past_key_values = kwargs.get("past_key_values")
            cache_is_empty = past_key_values is None or (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
            ) or (
                hasattr(past_key_values, "seen_tokens")
                and past_key_values.seen_tokens == 0
            )
            if cache_is_empty:
                self_model._baseline_llm_forward_ms = 0.0
                self_model._baseline_prefill_ms = 0.0
                self_model._baseline_decode_ms = 0.0
                self_model._baseline_decode_steps = 0

            original_lm_forward = self_model.language_model.forward

            def timed_lm_forward(*lm_args: Any, **lm_kwargs: Any) -> Any:
                t0 = time.perf_counter()
                result = original_lm_forward(*lm_args, **lm_kwargs)
                elapsed = (time.perf_counter() - t0) * 1000.0
                self_model._baseline_llm_forward_ms = (
                    getattr(self_model, "_baseline_llm_forward_ms", 0.0) + elapsed
                )
                if cache_is_empty:
                    self_model._baseline_prefill_ms = elapsed
                else:
                    self_model._baseline_decode_ms = (
                        getattr(self_model, "_baseline_decode_ms", 0.0) + elapsed
                    )
                    self_model._baseline_decode_steps = (
                        getattr(self_model, "_baseline_decode_steps", 0) + 1
                    )
                return result

            self_model.language_model.forward = timed_lm_forward
            try:
                return original_forward(*args, **kwargs)
            finally:
                self_model.language_model.forward = original_lm_forward

        feat_host._baseline_original_forward = original_forward
        feat_host.forward = types.MethodType(patched_forward, feat_host)
        feat_host._baseline_timing_patch_installed = True

    def _uninstall_forward_timing_patch(self, feat_host: Any) -> None:
        if not getattr(feat_host, "_baseline_timing_patch_installed", False):
            return
        feat_host.forward = feat_host._baseline_original_forward
        delattr(feat_host, "_baseline_original_forward")
        delattr(feat_host, "_baseline_timing_patch_installed")

    def _collect_llm_timing(self, feat_host: Any) -> dict[str, float]:
        timing = {
            "prefill_ms": getattr(feat_host, "_baseline_prefill_ms", 0.0),
            "decode_ms": getattr(feat_host, "_baseline_decode_ms", 0.0),
            "decode_steps": float(getattr(feat_host, "_baseline_decode_steps", 0)),
            "llm_forward_ms": getattr(feat_host, "_baseline_llm_forward_ms", 0.0),
        }
        for attr in (
            "_baseline_prefill_ms",
            "_baseline_decode_ms",
            "_baseline_decode_steps",
            "_baseline_llm_forward_ms",
        ):
            if hasattr(feat_host, attr):
                delattr(feat_host, attr)
        return timing

    def _build_full_model_messages(
        self,
        request: FullModelRequest,
        *,
        image: Any,
    ) -> list[Mapping[str, Any]]:
        reduced_mode = str(request.reason).startswith(("reduced_", "witness_"))
        is_android = str(request.benchmark) == "AndroidControl"
        if reduced_mode:
            system_prompt = self.config.reduced_model_system_prompt
        elif is_android:
            system_prompt = ANDROIDCONTROL_MAIUI_SYSTEM_PROMPT
        else:
            system_prompt = self.config.full_model_system_prompt
        if reduced_mode:
            user_text = (
                f"Benchmark: {request.benchmark}\n"
                f"Reason: {request.reason}\n"
                f"{request.prompt_text}\n\n"
                "Return exactly one label."
            )
        elif is_android:
            user_text = (
                f"{request.prompt_text}"
                "\nOutput exactly one next action (mobile_use) for the current screenshot."
            )
        else:
            user_text = (
                f"Benchmark: {request.benchmark}\n"
                f"Reason: {request.reason}\n"
                f"{request.prompt_text}\n\n"
                "Return exactly one JSON object."
            )
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _build_disambiguation_messages(
        self,
        request: DisambiguationRequest,
        *,
        image: Any,
    ) -> list[Mapping[str, Any]]:
        labels = ", ".join(request.valid_labels)
        user_text = (
            f"{request.prompt.text}\n"
            f"Valid action labels: {labels}\n"
            "Return exactly one action label."
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.local_disambiguation_system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _run_generation(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[str, int, int]:
        inputs = self._processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"truncation": True},
        )
        if hasattr(inputs, "to") and self._model_device is not None:
            inputs = inputs.to(self._model_device)
        input_ids = inputs["input_ids"]
        input_token_count = int(input_ids.shape[-1])
        generate_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
            do_sample=bool(float(temperature) > 0.0),
        )
        if float(temperature) > 0.0:
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)
        generated_ids = self._model.generate(**inputs, **generate_kwargs)
        trimmed = [
            output_row[len(input_row) :]
            for input_row, output_row in zip(input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_token_count = int(trimmed[0].shape[-1]) if trimmed else 0
        return output_text.strip(), input_token_count, generated_token_count

    def _run_generation_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        t_tok = time.perf_counter()
        inputs = self._processor.apply_chat_template(
            [list(messages) for messages in messages_batch],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"truncation": True, "padding": True},
        )
        tokenization_ms = (time.perf_counter() - t_tok) * 1000.0

        t_xfer = time.perf_counter()
        if hasattr(inputs, "to") and self._model_device is not None:
            inputs = inputs.to(self._model_device)
        gpu_transfer_ms = (time.perf_counter() - t_xfer) * 1000.0

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_token_counts = tuple(int(input_ids.shape[-1]) for _ in range(int(input_ids.shape[0])))
        else:
            input_token_counts = tuple(int(value) for value in attention_mask.sum(dim=-1).tolist())
        generate_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
            do_sample=bool(float(temperature) > 0.0),
        )
        if float(temperature) > 0.0:
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)

        t_gen = time.perf_counter()
        feat_host = _resolve_qwen3_vl_model(self._model)
        llm_timing: dict[str, float] = {}
        self._install_forward_timing_patch(feat_host)
        try:
            generated_ids = self._model.generate(**inputs, **generate_kwargs)
        finally:
            llm_timing = self._collect_llm_timing(feat_host)
            self._uninstall_forward_timing_patch(feat_host)
        model_generate_ms = (time.perf_counter() - t_gen) * 1000.0

        prompt_width = int(input_ids.shape[-1])
        trimmed = [output_row[prompt_width:] for output_row in generated_ids]

        t_dec = time.perf_counter()
        output_texts = tuple(
            text.strip()
            for text in self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        output_decode_ms = (time.perf_counter() - t_dec) * 1000.0

        generated_token_counts = tuple(int(row.shape[-1]) for row in trimmed)

        self._last_gen_timing = {
            "tokenization_ms": tokenization_ms,
            "gpu_transfer_ms": gpu_transfer_ms,
            "model_generate_ms": model_generate_ms,
            "output_decode_ms": output_decode_ms,
            **llm_timing,
        }

        return output_texts, input_token_counts, generated_token_counts

    def _validate_generation_batch(self, requests: Sequence[FullModelRequest]) -> None:
        first = requests[0]
        first_signature = (
            float(first.temperature),
            float(first.top_p),
            int(first.max_new_tokens),
            float(first.repetition_penalty),
        )
        for request in requests[1:]:
            signature = (
                float(request.temperature),
                float(request.top_p),
                int(request.max_new_tokens),
                float(request.repetition_penalty),
            )
            if signature != first_signature:
                raise ValueError(
                    "Batched Qwen generation requires identical decoding parameters across the batch."
                )

    def _prepare_image(
        self,
        screenshot: ScreenshotAsset,
        *,
        crop_bbox: BBox | None,
        max_pixels: int,
        dependencies: _RuntimeDependencies,
    ) -> Any:
        image = _load_image(screenshot, dependencies.image_module)
        if crop_bbox is not None:
            image = image.crop(
                _normalize_crop_bbox_for_processor(
                    crop_bbox,
                    image_size=image.size,
                    max_aspect_ratio=QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
                )
            )
        image = _pad_image_to_safe_aspect_ratio(
            image,
            image_module=dependencies.image_module,
            max_aspect_ratio=QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
        )
        width, height = image.size
        max_pixels_value = max(1, int(max_pixels))
        if width * height > max_pixels_value:
            scale = (float(max_pixels_value) / float(width * height)) ** 0.5
            resized = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(resized)
        return image

    def _processor_source_path(self) -> Path:
        adapter_path = self._adapter_path()
        if self.config.processor_from_adapter_first and adapter_path is not None:
            return adapter_path
        return self._base_model_path()

    def _base_model_path(self) -> Path:
        if self.model_spec.base_model_path is None:
            raise ValueError("QwenLoRABackend requires model_spec.base_model_path.")
        return Path(self.model_spec.base_model_path).resolve()

    def _adapter_path(self) -> Path | None:
        if self.model_spec.lora_adapter_path is None:
            return None
        return Path(self.model_spec.lora_adapter_path).resolve()

    def _parse_action_output(
        self,
        output_text: str,
        *,
        benchmark: str,
        request: FullModelRequest | None = None,
    ) -> CanonicalAction:
        is_android = str(benchmark) == "AndroidControl"
        if is_android and request is not None:
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
                pass

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
        if not request.reduced_valid_labels:
            return None
        label = self._parse_disambiguation_output(
            output_text,
            valid_labels=request.reduced_valid_labels,
        )
        if label is None:
            return None
        label_action_map = dict(request.reduced_label_action_map)
        return label_action_map.get(label)

    def _parse_disambiguation_output(
        self,
        output_text: str,
        *,
        valid_labels: Sequence[str],
    ) -> str | None:
        stripped = _normalize_label_output(output_text)
        payload = _extract_json_object(stripped)
        if payload is not None:
            for key in ("label", "selected_label", "action_label", "choice"):
                value = payload.get(key)
                if isinstance(value, str):
                    normalized = _normalize_label_output(value)
                    if normalized in valid_labels:
                        return normalized
        if stripped in valid_labels:
            return stripped
        compact = _normalize_label_output(stripped.splitlines()[0]) if stripped else ""
        return compact if compact in valid_labels else None

    def _validate_disambiguation_batch(self, requests: Sequence[DisambiguationRequest]) -> None:
        first = requests[0]
        first_signature = int(first.max_new_tokens)
        for request in requests[1:]:
            if int(request.max_new_tokens) != first_signature:
                raise ValueError(
                    "Batched Qwen disambiguation requires identical decoding parameters across the batch."
                )


def build_qwen_backend(
    model_spec: QwenLoRAModelSpec,
    *,
    config: QwenBackendConfig | None = None,
    eager_load: bool = False,
) -> QwenLoRABackend:
    """Build the concrete Qwen backend from a model spec."""

    return QwenLoRABackend(model_spec, config=config, eager_load=eager_load)


def _import_runtime_dependencies() -> _RuntimeDependencies:
    try:
        import torch
        from PIL import Image
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise DependencyUnavailableError(
            "Qwen runtime dependencies are not installed. "
            "Install torch, transformers with Qwen3-VL support, peft, accelerate, pillow, and safetensors."
        ) from exc
    return _RuntimeDependencies(
        torch=torch,
        image_module=Image,
        processor_cls=AutoProcessor,
        model_cls=Qwen3VLForConditionalGeneration,
        peft_model_cls=PeftModel,
    )


def _resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = str(dtype_name).strip().lower()
    mapping = {
        "auto": "auto",
        "bfloat16": getattr(torch, "bfloat16", None),
        "float16": getattr(torch, "float16", None),
        "float32": getattr(torch, "float32", None),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype setting: {dtype_name}")
    return mapping[normalized]


def _configure_processor_for_generation(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"


def _normalize_label_output(value: str | None) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().strip("`").strip()
    if normalized.startswith(("- ", "* ")):
        normalized = normalized[2:].strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {'"', "'"}
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def _resolve_single_device_map(device_map: int | str | None) -> int | str:
    if device_map is None:
        return 0
    if isinstance(device_map, int):
        return device_map
    normalized = str(device_map).strip().lower()
    if normalized == "auto":
        raise ValueError(
            "device_map='auto' is forbidden in this repository because it can shard one model across multiple GPUs. "
            "Use a single device ordinal such as 0 or a single device string such as 'cuda:0'."
        )
    return str(device_map)


def _parse_device_list(value: str) -> list[int]:
    devices: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        devices.append(int(stripped))
    return devices


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _load_image(screenshot: ScreenshotAsset, image_module: Any) -> Any:
    import io

    raw_bytes = screenshot.read_bytes()
    if raw_bytes:
        image = image_module.open(io.BytesIO(raw_bytes))
        return image.convert("RGB")
    return image_module.new("RGB", (max(1, int(screenshot.width or 1)), max(1, int(screenshot.height or 1))), color="white")


def _normalize_crop_bbox_for_processor(
    crop_bbox: BBox,
    *,
    image_size: tuple[int, int],
    max_aspect_ratio: float = QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
) -> tuple[int, int, int, int]:
    image_width = max(1, int(image_size[0] or 1))
    image_height = max(1, int(image_size[1] or 1))
    left, top, right, bottom = _coerce_crop_bbox(crop_bbox, image_size=(image_width, image_height))
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    if max(crop_width, crop_height) / float(min(crop_width, crop_height)) < float(max_aspect_ratio):
        return (left, top, right, bottom)

    if crop_width < crop_height:
        target_width = min(
            image_width,
            max(crop_width + 1, int(math.ceil(float(crop_height) / float(max_aspect_ratio)))),
        )
        left, right = _expand_span_to_length(left, right, total_length=image_width, target_length=target_width)
    else:
        target_height = min(
            image_height,
            max(crop_height + 1, int(math.ceil(float(crop_width) / float(max_aspect_ratio)))),
        )
        top, bottom = _expand_span_to_length(top, bottom, total_length=image_height, target_length=target_height)
    return (left, top, right, bottom)


def _coerce_crop_bbox(
    crop_bbox: BBox,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width = max(1, int(image_size[0] or 1))
    image_height = max(1, int(image_size[1] or 1))
    raw_left, raw_top, raw_right, raw_bottom = (int(value) for value in crop_bbox)
    left, right = sorted((raw_left, raw_right))
    top, bottom = sorted((raw_top, raw_bottom))
    left = min(max(0, left), image_width - 1)
    top = min(max(0, top), image_height - 1)
    right = min(max(left + 1, right), image_width)
    bottom = min(max(top + 1, bottom), image_height)
    return (left, top, right, bottom)


def _expand_span_to_length(
    start: int,
    end: int,
    *,
    total_length: int,
    target_length: int,
) -> tuple[int, int]:
    safe_total = max(1, int(total_length))
    safe_start = min(max(0, int(start)), safe_total - 1)
    safe_end = min(max(safe_start + 1, int(end)), safe_total)
    current_length = safe_end - safe_start
    desired_length = min(safe_total, max(current_length, int(target_length)))
    if current_length >= desired_length:
        return (safe_start, safe_end)

    deficit = desired_length - current_length
    grow_before = deficit // 2
    grow_after = deficit - grow_before
    safe_start = max(0, safe_start - grow_before)
    safe_end = min(safe_total, safe_end + grow_after)

    remaining = desired_length - (safe_end - safe_start)
    if remaining > 0:
        shift_left = min(safe_start, remaining)
        safe_start -= shift_left
        remaining -= shift_left
    if remaining > 0:
        safe_end = min(safe_total, safe_end + remaining)
        remaining = desired_length - (safe_end - safe_start)
    if remaining > 0:
        safe_start = max(0, safe_start - remaining)
    return (safe_start, safe_end)


def _pad_image_to_safe_aspect_ratio(
    image: Any,
    *,
    image_module: Any,
    max_aspect_ratio: float = QWEN_PROCESSOR_SAFE_ASPECT_RATIO,
) -> Any:
    width, height = image.size
    safe_width = max(1, int(width or 1))
    safe_height = max(1, int(height or 1))
    if max(safe_width, safe_height) / float(min(safe_width, safe_height)) < float(max_aspect_ratio):
        return image

    if safe_width < safe_height:
        padded_width = max(
            safe_width + 1,
            int(math.ceil(float(safe_height) / float(max_aspect_ratio))),
        )
        padded = image_module.new(image.mode, (padded_width, safe_height), color="white")
        padded.paste(image, ((padded_width - safe_width) // 2, 0))
        return padded

    padded_height = max(
        safe_height + 1,
        int(math.ceil(float(safe_width) / float(max_aspect_ratio))),
    )
    padded = image_module.new(image.mode, (safe_width, padded_height), color="white")
    padded.paste(image, (0, (padded_height - safe_height) // 2))
    return padded


def _extract_json_object(output_text: str) -> dict[str, Any] | None:
    stripped = output_text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    match = ACTION_JSON_RE.search(stripped)
    if match is not None:
        candidates.append(match.group(0))
    repaired = _repair_truncated_json_object(stripped)
    if repaired is not None:
        candidates.append(repaired)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


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


def _payload_to_action(payload: Mapping[str, Any]) -> CanonicalAction:
    bbox = payload.get("bbox")
    return CanonicalAction(
        action_type=str(payload.get("action_type") or "").upper(),
        argument=_coerce_optional_text(payload.get("argument")),
        bbox=_coerce_bbox(bbox),
        app=_coerce_optional_text(payload.get("app")),
        direction=_coerce_optional_text(payload.get("direction")),
    )


def _action_from_call(action_type: str, payload: str | None) -> CanonicalAction:
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
        bbox=_coerce_bbox(bbox),
        app=None,
        direction=_coerce_optional_text(direction),
    )


def _coerce_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


# ---------------------------------------------------------------------------
# Shared helpers consumed by service_backend.py
# ---------------------------------------------------------------------------

def _is_reduced_request_reason(reason: str) -> bool:
    """Return True when the request reason indicates a reduced (skill-solve) problem."""
    return str(reason).startswith(("reduced_", "witness_"))


def _resolved_request_valid_labels(request: Any) -> tuple[str, ...]:
    """Return the tuple of valid labels that constrain a reduced request."""
    return tuple(getattr(request, "reduced_valid_labels", ()))


def _resolved_request_reduced_label_action_map(request: Any) -> dict[str, Any]:
    """Return the label → CanonicalAction mapping for a reduced request."""
    return dict(getattr(request, "reduced_label_action_map", ()))


def _format_valid_action_label_block(valid_labels: Sequence[str]) -> str:
    """Format a list of valid action labels into a prompt block."""
    if not valid_labels:
        return ""
    return "Valid action labels: " + ", ".join(str(label) for label in valid_labels)


def _select_valid_label_from_output(output_text: str, *, valid_labels: Sequence[str]) -> str | None:
    """Select and return the first matching valid label from model output text."""
    stripped = _normalize_label_output(output_text)
    payload = _extract_json_object(stripped)
    if payload is not None:
        for key in ("label", "selected_label", "action_label", "choice"):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = _normalize_label_output(value)
                if normalized in valid_labels:
                    return normalized
    if stripped in valid_labels:
        return stripped
    compact = _normalize_label_output(stripped.splitlines()[0]) if stripped else ""
    return compact if compact in valid_labels else None


__all__ = [
    "ACTION_CALL_RE",
    "ACTION_JSON_RE",
    "AdapterCompatibilityReport",
    "DependencyUnavailableError",
    "QWEN_PROCESSOR_SAFE_ASPECT_RATIO",
    "QwenBackendConfig",
    "QwenLoRABackend",
    "_action_from_call",
    "_extract_json_object",
    "_format_valid_action_label_block",
    "_is_action_semantically_complete",
    "_is_reduced_request_reason",
    "_load_image",
    "_normalize_crop_bbox_for_processor",
    "_normalize_label_output",
    "_pad_image_to_safe_aspect_ratio",
    "_payload_to_action",
    "_resolve_single_device_map",
    "_resolved_request_reduced_label_action_map",
    "_resolved_request_valid_labels",
    "_select_valid_label_from_output",
    "build_qwen_backend",
    "inspect_lora_adapter_compatibility",
    "resolve_visible_cuda_devices",
]
