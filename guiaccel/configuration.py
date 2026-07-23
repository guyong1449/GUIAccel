"""Configuration helpers for the quotient-controller pipeline.

Supports both legacy `backend` section (local transformers) and the new
`model` + `service` section format used by the vLLM serving path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from guiaccel.calibration import ThresholdSelectionConfig
from guiaccel.evaluation.orchestrator import EvaluationMicrobatchConfig, EvaluationRunConfig
from guiaccel.model import ModelServiceConfig
from guiaccel.routing import FallbackModelConfig
from guiaccel.routing.fallback import ModelRuntimeSpec
from guiaccel.utils.env import load_repo_env
from guiaccel.utils.project_memory import get_repo_root

_SETUP_PLACEHOLDER = "__SET_BY_SETUP_SH__"


def default_config_path(benchmark: str) -> Path:
    normalized = str(benchmark).strip().lower()
    if normalized == "learngui":
        return get_repo_root() / "configs" / "learngui" / "default.json"
    if normalized == "androidcontrol":
        return get_repo_root() / "configs" / "androidcontrol" / "default.json"
    if normalized == "androidworld":
        return get_repo_root() / "configs" / "androidworld" / "default.json"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_json_config(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).resolve()
    payload = json.loads(resolved_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Config at {resolved_path} must decode to a JSON object.")
    return payload


def load_benchmark_config(
    *,
    benchmark: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if config_path is None and benchmark is None:
        raise ValueError("Either benchmark or config_path must be provided.")
    load_repo_env()
    resolved_path = Path(config_path).resolve() if config_path is not None else default_config_path(str(benchmark))
    payload = load_json_config(resolved_path)
    if benchmark is not None and str(payload.get("benchmark")) != str(benchmark):
        raise ValueError(
            f"Config benchmark mismatch: expected {benchmark}, found {payload.get('benchmark')} in {resolved_path}."
        )
    payload.setdefault("_config_path", str(resolved_path))
    payload = _resolve_config_paths(payload)
    # Propagate dataset manifest paths as env vars so deep code (e.g.
    # AndroidControlDataset() with no args) can locate datasets without
    # requiring explicit threading of manifest_path through every call site.
    _apply_dataset_env_vars(payload)
    return payload


def _dataset_manifest_env_var(config: Mapping[str, Any]) -> str:
    benchmark = str(config.get("benchmark") or "").strip()
    if benchmark == "LearnGUI":
        return "SKILLREUSE_LEARNGUI_DATASET_MANIFEST"
    return "SKILLREUSE_ANDROIDCONTROL_DATASET_MANIFEST"


def _resolve_config_path_value(
    value: Any,
    *,
    env_var: str | None = None,
    repo_root: Path | None = None,
) -> str | None:
    env_override = os.environ.get(env_var) if env_var else None
    if env_override:
        return str(Path(env_override).expanduser().resolve())

    if value in {None, ""}:
        return None

    normalized = str(value).strip()
    if normalized == _SETUP_PLACEHOLDER:
        raise ValueError(
            f"Config path placeholder {_SETUP_PLACEHOLDER!r} is unset. "
            f"Run scripts/setup/setup.sh or set {env_var or 'the matching SKILLREUSE_* env var'}."
        )

    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root or get_repo_root()) / candidate
    return str(candidate.resolve())


def _resolve_config_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved_config = dict(config)
    repo_root = get_repo_root()
    manifest_env_var = _dataset_manifest_env_var(resolved_config)

    paths = dict(_mapping(resolved_config.get("paths")))
    if "dataset_manifest" in paths:
        paths["dataset_manifest"] = _resolve_config_path_value(
            paths.get("dataset_manifest"),
            env_var=manifest_env_var,
            repo_root=repo_root,
        )
    if "base_model_path" in paths:
        paths["base_model_path"] = _resolve_config_path_value(
            paths.get("base_model_path"),
            env_var="SKILLREUSE_BASE_MODEL_PATH",
            repo_root=repo_root,
        )
    if "lora_adapter_path" in paths:
        paths["lora_adapter_path"] = _resolve_config_path_value(
            paths.get("lora_adapter_path"),
            env_var=None,
            repo_root=repo_root,
        )
    resolved_config["paths"] = paths

    model = dict(_mapping(resolved_config.get("model")))
    if "source" in model:
        model["source"] = _resolve_config_path_value(
            model.get("source"),
            env_var="SKILLREUSE_BASE_MODEL_PATH",
            repo_root=repo_root,
        )
    if "base_model_path" in model:
        model["base_model_path"] = _resolve_config_path_value(
            model.get("base_model_path"),
            env_var="SKILLREUSE_BASE_MODEL_PATH",
            repo_root=repo_root,
        )
    if "lora_adapter_path" in model:
        model["lora_adapter_path"] = _resolve_config_path_value(
            model.get("lora_adapter_path"),
            env_var=None,
            repo_root=repo_root,
        )
    if model:
        resolved_config["model"] = model

    return resolved_config


def _apply_dataset_env_vars(config: Mapping[str, Any]) -> None:
    """Set SKILLREUSE_*_DATASET_MANIFEST env vars from the config paths section.

    This allows deep code (e.g. AndroidControlDataset() called with no args)
    to locate the dataset without explicit manifest_path threading at every call site.
    """
    manifest = resolve_dataset_manifest_path(config)
    if not manifest or not Path(manifest).exists():
        return
    benchmark = str(config.get("benchmark") or "").strip()
    if benchmark == "AndroidControl" or "android_control" in manifest.lower():
        os.environ.setdefault("SKILLREUSE_ANDROIDCONTROL_DATASET_MANIFEST", manifest)
    elif benchmark == "LearnGUI" or "learngui" in manifest.lower():
        os.environ.setdefault("SKILLREUSE_LEARNGUI_DATASET_MANIFEST", manifest)
    else:
        # Unknown benchmark — set both to be safe
        os.environ.setdefault("SKILLREUSE_ANDROIDCONTROL_DATASET_MANIFEST", manifest)


def apply_runtime_overrides(
    config: Mapping[str, Any],
    *,
    model_path: str | Path | None = None,
    model_name: str | None = None,
    served_model_name: str | None = None,
    api_base: str | None = None,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge CLI-level overrides into the config dict, updating paths/model/service sub-dicts."""
    resolved_config = dict(config)
    paths = dict(_mapping(resolved_config.get("paths")))
    model = dict(_mapping(resolved_config.get("model")))
    service = dict(_mapping(resolved_config.get("service")))

    if model_path is not None:
        normalized_model_source = _normalize_model_source(model_path)
        paths["base_model_path"] = normalized_model_source
        model["source"] = normalized_model_source
    if model_name is not None:
        model["model_name"] = str(model_name)
    if served_model_name is not None:
        normalized_served_model_name = str(served_model_name)
        model["served_model_name"] = normalized_served_model_name
        service["served_model_name"] = normalized_served_model_name
    elif model_name is not None:
        # An explicit request-model override should not remain pinned to a stale served-model default.
        model.pop("served_model_name", None)
        service.pop("served_model_name", None)
    if api_base is not None:
        service["api_base"] = str(api_base)
        service.pop("api_bases", None)
    if adapter_path is not None:
        resolved_adapter_path = str(Path(adapter_path).expanduser().resolve())
        paths["lora_adapter_path"] = resolved_adapter_path
        model["lora_adapter_path"] = resolved_adapter_path

    resolved_config["paths"] = paths
    if model:
        resolved_config["model"] = model
    else:
        resolved_config.pop("model", None)
    if service:
        resolved_config["service"] = service
    else:
        resolved_config.pop("service", None)
    return resolved_config


def resolve_model_spec(
    config: Mapping[str, Any],
    *,
    benchmark: str | None = None,
) -> ModelRuntimeSpec:
    """Build a ModelRuntimeSpec from config, reading model/service/paths sub-dicts."""
    model = _mapping(config.get("model"))
    service = _mapping(config.get("service"))
    paths = _mapping(config.get("paths"))
    backend = str(
        model.get("backend")
        or model.get("kind")
        or service.get("backend")
        or service.get("kind")
        or _mapping(config.get("backend")).get("kind")
        or _mapping(config.get("backend")).get("backend")
        or "vllm"
    )
    model_name = (
        model.get("model_name")
        or service.get("served_model_name")
        or model.get("served_model_name")
        or model.get("source")
        or paths.get("base_model_path")
        or "Qwen3-VL-8B-Instruct"
    )
    return ModelRuntimeSpec(
        benchmark=str(benchmark or config.get("benchmark") or ""),
        backend=str(backend),
        base_model_path=model.get("source") or model.get("base_model_path") or paths.get("base_model_path"),
        lora_adapter_path=model.get("lora_adapter_path") or paths.get("lora_adapter_path"),
        model_name=str(model_name),
        served_model_name=_optional_str(service.get("served_model_name") or model.get("served_model_name")),
    )


def resolve_backend_config(config: Mapping[str, Any]) -> ModelServiceConfig:
    """Build a ModelServiceConfig from config, merging backend + service sub-dicts."""
    backend = dict(_mapping(config.get("backend")))
    service = dict(_mapping(config.get("service")))
    payload = {**backend, **service}
    # Normalise kind/backend key
    if "kind" not in payload and "backend" in payload:
        payload["kind"] = payload["backend"]
    payload.pop("backend", None)
    # served_model_name lives on ModelRuntimeSpec, not ModelServiceConfig
    payload.pop("served_model_name", None)
    # Coerce tuple fields
    if "api_bases" in payload:
        payload["api_bases"] = _as_str_tuple(payload["api_bases"])
    if "visible_cuda_devices" in payload:
        payload["visible_cuda_devices"] = tuple(int(device) for device in payload["visible_cuda_devices"])
    if "extra_request_headers" in payload:
        payload["extra_request_headers"] = dict(_mapping(payload["extra_request_headers"]))
    extra = _mapping(payload.get("extra"))
    if payload.get("attn_implementation") is None and "attn_implementation" in extra:
        payload["attn_implementation"] = extra["attn_implementation"]
    return ModelServiceConfig(**payload)


def resolve_fallback_config(config: Mapping[str, Any]) -> FallbackModelConfig:
    return FallbackModelConfig(**dict(_mapping(config.get("fallback"))))


def resolve_discovery_config(config: Mapping[str, Any]) -> dict[str, Any]:
    discovery = _mapping(config.get("discovery"))
    return {
        "max_skeletons_per_step": int(discovery.get("max_skeletons_per_step", 16)),
        "min_cluster_size": int(discovery.get("min_cluster_size", 32)),
        "cluster_min_similarity": float(discovery.get("cluster_min_similarity", 0.82)),
        "repository_episode_limit": _optional_int(discovery.get("repository_episode_limit")),
        "calibration_episode_limit": _optional_int(discovery.get("calibration_episode_limit")),
        "repository_task_limit": _optional_int(discovery.get("repository_task_limit")),
        "calibration_task_limit": _optional_int(discovery.get("calibration_task_limit")),
        "repository_batch_size": max(1, int(discovery.get("repository_batch_size", 4))),
        "witness_batch_size": max(1, int(discovery.get("witness_batch_size", 16))),
        "progress_flush_interval_examples": max(1, int(discovery.get("progress_flush_interval_examples", 32))),
        "worker_gpus": tuple(int(item) for item in discovery.get("worker_gpus", ())),
    }


def resolve_threshold_selection_config(config: Mapping[str, Any]) -> ThresholdSelectionConfig:
    calibration = _mapping(config.get("calibration"))
    return ThresholdSelectionConfig(**dict(_mapping(calibration.get("threshold_selection"))))


def resolve_dataset_manifest_path(config: Mapping[str, Any]) -> str | None:
    paths = _mapping(config.get("paths"))
    manifest_path = paths.get("dataset_manifest")
    if manifest_path in {None, ""}:
        return None
    return str(Path(str(manifest_path)).expanduser().resolve())


def resolve_androidworld_config(config_path: str | Path) -> "AndroidWorldRunConfig":
    """Parse an AndroidWorld JSON config file into an AndroidWorldRunConfig."""
    from guiaccel.evaluation.androidworld import AndroidWorldRunConfig

    raw = json.loads(Path(config_path).read_text())

    env_section = _mapping(raw.get("environment"))
    task_section = _mapping(raw.get("tasks"))
    model_section = _mapping(raw.get("model"))
    output_section = _mapping(raw.get("output"))
    evaluation_section = _mapping(raw.get("evaluation"))

    service_config = None
    if "service" in raw or "backend" in raw:
        service_config = resolve_backend_config(raw)

    backend_type = str(model_section.get("backend_type", "local_transformers"))
    if service_config is not None and service_config.kind:
        backend_type = service_config.kind

    raw_endpoints = env_section.get("emulator_endpoints", ())
    emulator_endpoints = tuple(
        (str(ep[0]), int(ep[1])) for ep in raw_endpoints
    )

    return AndroidWorldRunConfig(
        emulator_host=str(env_section.get("emulator_host", "localhost")),
        emulator_port=int(env_section.get("emulator_port", 5554)),
        adb_server_port=int(env_section.get("adb_server_port", 5037)),
        grpc_port=int(env_section.get("grpc_port", 8554)),
        screen_width=int(env_section.get("screen_width", 1080)),
        screen_height=int(env_section.get("screen_height", 2400)),
        perform_emulator_setup=bool(env_section.get("perform_emulator_setup", False)),
        transition_pause=float(env_section.get("transition_pause", 1.0)),
        max_steps_per_task=int(task_section.get("max_steps_per_task", 30)),
        task_timeout_seconds=float(task_section.get("task_timeout_seconds", 600.0)),
        task_names=tuple(str(t) for t in task_section.get("task_names", ())),
        suite_family=str(task_section.get("suite_family", "android_world")),
        fallback=resolve_fallback_config(raw),
        model_path=_optional_str(model_section.get("model_path")),
        adapter_path=_optional_str(model_section.get("adapter_path")),
        api_base=_optional_str(model_section.get("api_base")),
        backend_type=backend_type,
        max_new_tokens=int(model_section.get("max_new_tokens", 2048)),
        model_coordinate_scale=int(model_section.get("coordinate_scale", 999)),
        checkpoint_interval=int(output_section.get("checkpoint_interval", 1)),
        save_screenshots=bool(output_section.get("save_screenshots", True)),
        worker_gpus=tuple(int(g) for g in evaluation_section.get("worker_gpus", ())),
        emulator_endpoints=emulator_endpoints,
        measure_end_to_end_latency=bool(evaluation_section.get("measure_end_to_end_latency", False)),
        task_limit=_optional_int(evaluation_section.get("task_limit")),
        service_config=service_config,
    )


def resolve_evaluation_config(config: Mapping[str, Any]) -> EvaluationRunConfig:
    evaluation = _mapping(config.get("evaluation"))
    microbatch = _mapping(evaluation.get("microbatch"))
    if not microbatch:
        microbatch = _mapping(evaluation.get("microbatch_caps"))
    return EvaluationRunConfig(
        fallback=resolve_fallback_config(config),
        learn_gui_shots=tuple(int(item) for item in evaluation.get("learn_gui_shots", (2,))),
        android_instruction_modes=tuple(str(item) for item in evaluation.get("android_instruction_modes", ("high_level", "low_level"))),
        learn_gui_split=str(evaluation.get("learn_gui_split", "test")),
        android_split=str(evaluation.get("android_split", "test")),
        task_limit=_optional_int(evaluation.get("task_limit")),
        episode_limit=_optional_int(evaluation.get("episode_limit")),
        worker_gpus=tuple(int(item) for item in evaluation.get("worker_gpus", ())),
        microbatch=EvaluationMicrobatchConfig(
            sample_cap_per_gpu=max(1, int(microbatch.get("sample_cap_per_gpu", 6))),
            image_cap_per_gpu=max(1, int(microbatch.get("image_cap_per_gpu", 42))),
            visual_token_cap_per_gpu=max(1, int(microbatch.get("visual_token_cap_per_gpu", 12_288))),
            text_token_cap_per_gpu=max(1, int(microbatch.get("text_token_cap_per_gpu", 12_288))),
            progress_flush_interval_examples=max(1, int(microbatch.get("progress_flush_interval_examples", 16))),
        ),
        measure_end_to_end_latency=bool(evaluation.get("measure_end_to_end_latency", False)),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else tuple()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else tuple()


def _normalize_model_source(value: str | Path) -> str:
    candidate = Path(value).expanduser()
    if str(value).startswith(("/", ".", "~")) or candidate.exists():
        return str(candidate.resolve())
    return str(value)


__all__ = [
    "apply_runtime_overrides",
    "default_config_path",
    "load_benchmark_config",
    "load_json_config",
    "resolve_androidworld_config",
    "resolve_backend_config",
    "resolve_dataset_manifest_path",
    "resolve_discovery_config",
    "resolve_evaluation_config",
    "resolve_fallback_config",
    "resolve_model_spec",
    "resolve_threshold_selection_config",
]
