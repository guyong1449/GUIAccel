"""AndroidWorld interactive evaluation: config, data models, and evaluation loop."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from guiaccel.routing.fallback import FallbackModelConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AndroidWorldRunConfig:
    """Complete configuration for an AndroidWorld interactive evaluation run."""

    # --- emulator connection ---
    emulator_host: str = "localhost"
    emulator_port: int = 5554
    adb_server_port: int = 5037
    grpc_port: int = 8554

    # --- task control ---
    max_steps_per_task: int = 30
    transition_pause: float = 1.0
    task_timeout_seconds: float = 600.0
    task_names: tuple[str, ...] = ()

    # --- screen / coordinate ---
    screen_width: int = 1080
    screen_height: int = 2400
    model_coordinate_scale: int = 999

    # --- model ---
    fallback: FallbackModelConfig = field(default_factory=FallbackModelConfig)
    model_path: str | None = None
    adapter_path: str | None = None
    api_base: str | None = None
    backend_type: str = "local_transformers"
    max_new_tokens: int = 2048

    # --- output ---
    checkpoint_interval: int = 1
    save_screenshots: bool = True

    # --- advanced ---
    perform_emulator_setup: bool = False
    suite_family: str = "android_world"

    # --- multi-GPU parallel ---
    worker_gpus: tuple[int, ...] = ()
    emulator_endpoints: tuple[tuple[str, int], ...] = ()

    # --- timing ---
    measure_end_to_end_latency: bool = False

    # --- task limit (smoke tests) ---
    task_limit: int | None = None

    # --- full backend config (from resolve_backend_config) ---
    service_config: Any = field(default=None, compare=False, hash=False)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AndroidWorldStepTrace:
    """Per-step interaction trace (for analysis, not task success judgement)."""

    step_index: int
    goal: str
    screenshot_path: Path | None
    observation_summary: dict[str, Any]
    model_output_text: str | None
    canonical_action: Any | None
    json_action: Any | None
    routing_mode: str
    latency_ms: float
    token_usage: dict[str, int]
    error: str | None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    vision_encoder_ms: float | None = None
    model_latency_ms: float | None = None
    model_timing: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AndroidWorldTaskResult:
    """Complete evaluation result for one AndroidWorld task."""

    task_name: str
    task_goal: str
    task_params: dict[str, Any]
    success: float
    num_steps: int
    max_steps: int
    steps: tuple[AndroidWorldStepTrace, ...]
    total_latency_ms: float
    total_tokens: dict[str, int]
    error: str | None
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def build_androidworld_report(
    results: Sequence[AndroidWorldTaskResult],
    *,
    config: AndroidWorldRunConfig,
    variant_id: str,
) -> dict[str, Any]:
    """Build an AndroidWorld evaluation report from a sequence of task results."""
    total = len(results)
    successes = sum(1 for r in results if r.success >= 1.0)
    partial_scores = [r.success for r in results]

    per_app_results: dict[str, list[float]] = {}
    for r in results:
        for app in r.metadata.get("app_names", ()):
            per_app_results.setdefault(app, []).append(r.success)

    return {
        "benchmark": "AndroidWorld",
        "variant_id": variant_id,
        "task_count": total,
        "task_success_rate": successes / total if total else 0.0,
        "mean_partial_score": sum(partial_scores) / total if total else 0.0,
        "per_app_success_rate": {
            app: sum(scores) / len(scores)
            for app, scores in sorted(per_app_results.items())
        },
        "aggregate_steps": sum(r.num_steps for r in results),
        "aggregate_latency_ms": sum(r.total_latency_ms for r in results),
        "aggregate_tokens": _sum_token_dicts(r.total_tokens for r in results),
        "config": {
            "emulator_host": config.emulator_host,
            "max_steps_per_task": config.max_steps_per_task,
            "transition_pause": config.transition_pause,
            "screen_width": config.screen_width,
            "screen_height": config.screen_height,
        },
        "task_results": [_serialize_task_result(r) for r in results],
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_androidworld_evaluation(
    *,
    config: AndroidWorldRunConfig,
    variant_id: str = "V0",
    skill_library_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    task_filter: Sequence[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """AndroidWorld evaluation main entry point."""
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    if config.worker_gpus:
        return _run_androidworld_parallel(
            config=config,
            variant_id=variant_id,
            skill_library_path=skill_library_path,
            output_dir=output_dir,
            task_filter=task_filter,
            resume=resume,
        )

    from guiaccel.evaluation.androidworld_agent import SkillReuseAndroidWorldAgent

    completed_task_names: set[str] = set()
    previously_completed: list[AndroidWorldTaskResult] = []
    if resume and output_path is not None:
        completed_task_names = _load_completed_task_names(output_path)
        previously_completed = _load_completed_results(output_path)
        log.info("Resuming: %d tasks already completed.", len(completed_task_names))

    env = _create_androidworld_env(config)
    tasks = _load_androidworld_tasks(config, task_filter=task_filter)
    if config.task_limit is not None:
        tasks = tasks[: config.task_limit]

    model, model_spec = _build_fallback_model_and_spec(config)

    agent = SkillReuseAndroidWorldAgent(
        env=env,
        config=config,
        variant_id=variant_id,
        fallback_model=model,
        model_spec=model_spec,
        skill_library_path=skill_library_path,
        output_dir=output_path,
    )

    results: list[AndroidWorldTaskResult] = list(previously_completed)
    for task_eval in tasks:
        if task_eval.name in completed_task_names:
            log.info("Skipping already-completed task: %s", task_eval.name)
            continue
        try:
            result = _run_single_task(env, agent, task_eval, config=config)
        except Exception as exc:
            log.exception("Task %s failed with error: %s", task_eval.name, exc)
            result = AndroidWorldTaskResult(
                task_name=task_eval.name,
                task_goal=getattr(task_eval, "goal", ""),
                task_params=dict(getattr(task_eval, "params", {})),
                success=0.0,
                num_steps=0,
                max_steps=config.max_steps_per_task,
                steps=(),
                total_latency_ms=0.0,
                total_tokens={},
                error=str(exc),
                metadata={},
            )
        results.append(result)
        if output_path is not None:
            _checkpoint_result(result, output_path)

    try:
        env.close()
    except Exception:
        pass

    report = build_androidworld_report(results, config=config, variant_id=variant_id)
    if output_path is not None:
        _save_androidworld_report(report, output_path)
        if config.measure_end_to_end_latency:
            _write_androidworld_timing_summary(results, output_path)
    return report


def _run_single_task(
    env: Any,
    agent: Any,
    task_eval: Any,
    *,
    config: AndroidWorldRunConfig,
) -> AndroidWorldTaskResult:
    """Run one task: initialize → step loop → is_successful → tear_down."""
    max_steps = _compute_max_steps(getattr(task_eval, "complexity", 1), config.max_steps_per_task)
    task_start = time.monotonic()

    task_eval.initialize_task(env)
    agent.reset()
    agent.set_task_name(getattr(task_eval, "name", "unknown"))
    agent.set_max_steps(max_steps)

    step_traces: list[AndroidWorldStepTrace] = []
    goal = str(getattr(task_eval, "goal", ""))
    deadline = task_start + config.task_timeout_seconds

    for step_idx in range(max_steps):
        if time.monotonic() > deadline:
            log.warning("Task %s exceeded timeout of %.0fs", task_eval.name, config.task_timeout_seconds)
            step_traces.append(AndroidWorldStepTrace(
                step_index=step_idx,
                goal=goal,
                screenshot_path=None,
                observation_summary={},
                model_output_text=None,
                canonical_action=None,
                json_action=None,
                routing_mode="timeout",
                latency_ms=0.0,
                token_usage={},
                error=f"Task exceeded timeout of {config.task_timeout_seconds}s",
            ))
            break
        try:
            interaction = agent.step(goal)
        except Exception as exc:
            log.warning("Step %d of task %s raised: %s", step_idx, task_eval.name, exc)
            step_traces.append(AndroidWorldStepTrace(
                step_index=step_idx,
                goal=goal,
                screenshot_path=None,
                observation_summary={},
                model_output_text=None,
                canonical_action=None,
                json_action=None,
                routing_mode="error",
                latency_ms=0.0,
                token_usage={},
                error=str(exc),
            ))
            break
        trace = interaction.data.get("step_trace")
        if trace is not None:
            step_traces.append(trace)
        if interaction.done:
            break

    try:
        success = float(task_eval.is_successful(env))
    except Exception as exc:
        log.warning("is_successful for %s raised: %s", task_eval.name, exc)
        success = 0.0

    try:
        task_eval.tear_down(env)
    except Exception:
        pass

    total_latency_ms = (time.monotonic() - task_start) * 1000.0
    total_tokens = _sum_token_dicts(trace.token_usage for trace in step_traces)

    return AndroidWorldTaskResult(
        task_name=task_eval.name,
        task_goal=goal,
        task_params=dict(getattr(task_eval, "params", {})),
        success=success,
        num_steps=len(step_traces),
        max_steps=max_steps,
        steps=tuple(step_traces),
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
        error=None,
        metadata={
            "app_names": list(getattr(task_eval, "app_names", [])),
            "complexity": getattr(task_eval, "complexity", None),
        },
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _checkpoint_result(result: AndroidWorldTaskResult, output_dir: Path) -> None:
    safe_name = result.task_name.replace("/", "_").replace(" ", "_")
    checkpoint_path = output_dir / f"task_{safe_name}.json"
    checkpoint_path.write_text(
        json.dumps(_serialize_task_result(result), indent=2, default=str)
    )


def _load_completed_task_names(output_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in output_dir.glob("task_*.json"):
        try:
            data = json.loads(path.read_text())
            name = data.get("task_name")
            if name:
                names.add(str(name))
        except Exception:
            pass
    return names


def _load_checkpoint_result(path: Path) -> AndroidWorldTaskResult | None:
    """Load a single checkpoint JSON and reconstruct an AndroidWorldTaskResult."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not data.get("task_name"):
        return None
    steps = tuple(
        AndroidWorldStepTrace(
            step_index=s.get("step_index", i),
            goal=s.get("goal", ""),
            screenshot_path=None,
            observation_summary={},
            model_output_text=s.get("model_output_text"),
            canonical_action=None,
            json_action=None,
            routing_mode=s.get("routing_mode", "unknown"),
            latency_ms=s.get("latency_ms", 0.0),
            token_usage=s.get("token_usage", {}),
            error=s.get("error"),
            prefill_ms=s.get("prefill_ms"),
            decode_ms=s.get("decode_ms"),
            vision_encoder_ms=s.get("vision_encoder_ms"),
            model_latency_ms=s.get("model_latency_ms"),
            model_timing=dict(s.get("model_timing", {})),
        )
        for i, s in enumerate(data.get("steps", []))
    )
    return AndroidWorldTaskResult(
        task_name=data["task_name"],
        task_goal=data.get("task_goal", ""),
        task_params=data.get("task_params", {}),
        success=float(data.get("success", 0.0)),
        num_steps=int(data.get("num_steps", len(steps))),
        max_steps=int(data.get("max_steps", 0)),
        steps=steps,
        total_latency_ms=float(data.get("total_latency_ms", 0.0)),
        total_tokens=data.get("total_tokens", {}),
        error=data.get("error"),
        metadata=data.get("metadata", {}),
    )


def _load_completed_results(output_dir: Path) -> list[AndroidWorldTaskResult]:
    """Load all checkpoint results from existing task JSON files."""
    results: list[AndroidWorldTaskResult] = []
    for path in sorted(output_dir.glob("task_*.json")):
        result = _load_checkpoint_result(path)
        if result is not None:
            results.append(result)
    return results


def _save_androidworld_report(report: dict[str, Any], output_dir: Path) -> None:
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Environment and task loading
# ---------------------------------------------------------------------------


def _console_port_for_grpc(grpc_port: int, *, base_grpc: int = 8554, base_console: int = 5554) -> int:
    return base_console + 2 * (grpc_port - base_grpc)


def _create_androidworld_env(config: AndroidWorldRunConfig) -> Any:
    try:
        from android_env import loader
        from android_env.components import config_classes
        from android_world.env import android_world_controller, env_launcher
    except ImportError as exc:
        raise ImportError(
            "android_world is required for AndroidWorld evaluation. "
            "Install via: pip install -r requirements-androidworld.txt"
        ) from exc

    console_port = config.emulator_port
    if console_port == 5554 and config.grpc_port != 8554:
        console_port = _console_port_for_grpc(config.grpc_port)

    adb_path = str(Path(__file__).resolve().parents[2] / "scripts/androidworld/adb_relay_wrapper.sh")
    env_config = config_classes.AndroidEnvConfig(
        task=config_classes.FilesystemTaskConfig(
            path=android_world_controller._write_default_task_proto()
        ),
        simulator=config_classes.EmulatorConfig(
            emulator_launcher=config_classes.EmulatorLauncherConfig(
                emulator_console_port=console_port,
                adb_port=console_port + 1,
                grpc_port=config.grpc_port,
            ),
            adb_controller=config_classes.AdbControllerConfig(
                adb_path=adb_path,
                adb_server_port=config.adb_server_port,
                default_timeout=60.0,
            ),
        ),
    )
    from android_world.env import interface

    android_env_instance = loader.load(env_config)
    controller = android_world_controller.AndroidWorldController(
        android_env_instance,
        install_a11y_forwarding_app=False,
    )
    async_env = interface.AsyncAndroidEnv(controller)
    env_launcher.setup_env(
        async_env,
        config.perform_emulator_setup,
        freeze_datetime=True,
    )
    return async_env


def _load_androidworld_tasks(
    config: AndroidWorldRunConfig,
    *,
    task_filter: Sequence[str] | None = None,
) -> list[Any]:
    try:
        from android_world import registry, suite_utils
    except ImportError as exc:
        raise ImportError(
            "android_world is required for AndroidWorld evaluation. "
            "Install via: pip install -r requirements-androidworld.txt"
        ) from exc

    task_registry = registry.TaskRegistry().get_registry(config.suite_family)

    selected: list[str] | None = None
    if config.task_names and task_filter:
        selected = sorted(set(config.task_names) & set(task_filter))
    elif config.task_names:
        selected = list(config.task_names)
    elif task_filter:
        selected = list(task_filter)

    suite = suite_utils.create_suite(
        task_registry, n_task_combinations=1, tasks=selected
    )
    return [task for instances in suite.values() for task in instances]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def _build_fallback_model_and_spec(
    config: AndroidWorldRunConfig,
    *,
    gpu_id: int | None = None,
) -> tuple[Any, Any]:
    from dataclasses import replace as dc_replace

    from guiaccel.model import ModelServiceConfig, build_backend
    from guiaccel.routing.fallback import ModelRuntimeSpec

    model_spec = ModelRuntimeSpec(
        benchmark="AndroidControl",
        backend=config.backend_type,
        base_model_path=config.model_path,
        lora_adapter_path=config.adapter_path,
    )

    if config.service_config is not None:
        service_config = config.service_config
        if gpu_id is not None:
            service_config = dc_replace(
                service_config,
                visible_cuda_devices=(gpu_id,),
                device_map=0,
            )
    else:
        service_config = ModelServiceConfig(
            kind=config.backend_type,
            api_base=config.api_base or "http://127.0.0.1:8000/v1",
        )

    model = build_backend(model_spec, config=service_config, eager_load=True)
    return model, model_spec


# ---------------------------------------------------------------------------
# Parallel evaluation helpers
# ---------------------------------------------------------------------------


def _plan_androidworld_task_assignments(
    tasks: Sequence[Any],
    worker_gpus: Sequence[int],
) -> list[tuple[int, list[str]]]:
    """Distribute tasks to GPU workers by estimated load (complexity * 10), greedy."""
    loads: dict[int, int] = {gpu: 0 for gpu in worker_gpus}
    assignments: dict[int, list[str]] = {gpu: [] for gpu in worker_gpus}
    sorted_tasks = sorted(tasks, key=lambda t: getattr(t, "complexity", 1), reverse=True)
    for task in sorted_tasks:
        target_gpu = min(worker_gpus, key=lambda g: (loads[g], g))
        assignments[target_gpu].append(task.name)
        loads[target_gpu] += max(1, int(getattr(task, "complexity", 1))) * 10
    return [(gpu, names) for gpu, names in assignments.items() if names]


def _resolve_emulator_endpoints(
    config: AndroidWorldRunConfig,
) -> list[tuple[str, int]]:
    """Resolve or auto-generate emulator endpoints for parallel workers."""
    if config.emulator_endpoints:
        return list(config.emulator_endpoints)
    host = config.emulator_host
    base_port = config.grpc_port
    n = max(1, len(config.worker_gpus))
    return [(host, base_port + i) for i in range(n)]


def _parallel_androidworld_worker(
    gpu_id: int,
    emulator_host: str,
    emulator_grpc_port: int,
    task_names: tuple[str, ...],
    config: AndroidWorldRunConfig,
    variant_id: str,
    skill_library_path: str | None,
    worker_output_dir: str | None,
    resume: bool,
    result_queue: Any,
) -> None:
    """Per-GPU worker: independent model + emulator + assigned task subset."""
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from dataclasses import replace as dc_replace

        from guiaccel.evaluation.androidworld_agent import SkillReuseAndroidWorldAgent

        worker_config = dc_replace(
            config,
            emulator_host=emulator_host,
            grpc_port=emulator_grpc_port,
        )

        model, model_spec = _build_fallback_model_and_spec(worker_config, gpu_id=gpu_id)
        env = _create_androidworld_env(worker_config)
        tasks = _load_androidworld_tasks(worker_config, task_filter=list(task_names))

        output_path = Path(worker_output_dir) if worker_output_dir else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        completed_task_names: set[str] = set()
        previously_completed: list[AndroidWorldTaskResult] = []
        if resume and output_path is not None:
            completed_task_names = _load_completed_task_names(output_path)
            previously_completed = _load_completed_results(output_path)
            log.info("Worker GPU %d: resuming with %d completed tasks.", gpu_id, len(completed_task_names))

        agent = SkillReuseAndroidWorldAgent(
            env=env,
            config=worker_config,
            variant_id=variant_id,
            fallback_model=model,
            model_spec=model_spec,
            skill_library_path=skill_library_path,
            output_dir=output_path,
        )

        results: list[AndroidWorldTaskResult] = list(previously_completed)
        for task_eval in tasks:
            if task_eval.name in completed_task_names:
                log.info("Worker GPU %d: skipping completed task %s", gpu_id, task_eval.name)
                continue
            try:
                result = _run_single_task(env, agent, task_eval, config=worker_config)
            except Exception as exc:
                log.exception("Worker GPU %d: task %s failed: %s", gpu_id, task_eval.name, exc)
                result = AndroidWorldTaskResult(
                    task_name=task_eval.name,
                    task_goal=getattr(task_eval, "goal", ""),
                    task_params=dict(getattr(task_eval, "params", {})),
                    success=0.0,
                    num_steps=0,
                    max_steps=worker_config.max_steps_per_task,
                    steps=(),
                    total_latency_ms=0.0,
                    total_tokens={},
                    error=str(exc),
                    metadata={},
                )
            results.append(result)
            if output_path is not None:
                _checkpoint_result(result, output_path)

        try:
            env.close()
        except Exception:
            log.warning("Worker GPU %d: env.close() raised.", gpu_id, exc_info=True)

        result_queue.put((gpu_id, tuple(results), None))
    except Exception:
        import traceback
        result_queue.put((gpu_id, tuple(), traceback.format_exc()))


def _run_androidworld_parallel(
    *,
    config: AndroidWorldRunConfig,
    variant_id: str,
    skill_library_path: str | Path | None,
    output_dir: str | Path | None,
    task_filter: Sequence[str] | None,
    resume: bool,
) -> dict[str, Any]:
    """Multi-GPU parallel AndroidWorld evaluation."""
    import multiprocessing as mp
    import queue
    import time

    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    from guiaccel.model.service_backend import require_local_worker_backend
    from guiaccel.routing.fallback import ModelRuntimeSpec

    model_spec = ModelRuntimeSpec(
        benchmark="AndroidControl",
        backend=config.backend_type,
        base_model_path=config.model_path,
        lora_adapter_path=config.adapter_path,
    )
    require_local_worker_backend(
        model_spec,
        config=config.service_config,
        context="AndroidWorld parallel evaluation",
    )

    tasks = _load_androidworld_tasks(config, task_filter=task_filter)
    if config.task_limit is not None:
        tasks = tasks[: config.task_limit]

    assignments = _plan_androidworld_task_assignments(tasks, config.worker_gpus)
    endpoints = _resolve_emulator_endpoints(config)

    if len(endpoints) < len(config.worker_gpus):
        raise ValueError(
            f"Need at least {len(config.worker_gpus)} emulator endpoints "
            f"for {len(config.worker_gpus)} GPU workers, got {len(endpoints)}."
        )

    resolved_skill_library_path = (
        str(Path(skill_library_path).resolve()) if skill_library_path else None
    )

    if output_path is not None:
        progress_marker = output_path / "parallel_evaluation_progress.json"
        progress_marker.write_text(json.dumps({
            "worker_gpus": list(config.worker_gpus),
            "assignments": [
                {"gpu_id": gpu_id, "task_count": len(task_names)}
                for gpu_id, task_names in assignments
            ],
        }, indent=2))

    gpu_to_endpoint = {
        gpu: endpoints[i] for i, gpu in enumerate(config.worker_gpus)
    }

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    workers: list[tuple[int, Any]] = []

    for i, (gpu_id, task_names) in enumerate(assignments):
        if i > 0:
            time.sleep(3)
        host, port = gpu_to_endpoint[gpu_id]
        worker_output = str(output_path / f"worker_gpu{gpu_id}") if output_path else None

        process = context.Process(
            target=_parallel_androidworld_worker,
            args=(
                gpu_id,
                host,
                port,
                tuple(task_names),
                config,
                variant_id,
                resolved_skill_library_path,
                worker_output,
                resume,
                result_queue,
            ),
        )
        process.start()
        workers.append((gpu_id, process))
        log.info("Spawned worker GPU %d: %d tasks, emulator %s:%d", gpu_id, len(task_names), host, port)

    collected: dict[int, tuple[AndroidWorldTaskResult, ...]] = {}
    pending_gpus = {int(gpu_id) for gpu_id, _ in workers}

    while pending_gpus:
        try:
            gpu_id, results, error_text = result_queue.get(timeout=5.0)
        except queue.Empty:
            dead = [
                (int(gid), int(proc.exitcode))
                for gid, proc in workers
                if int(gid) in pending_gpus and proc.exitcode is not None and proc.exitcode != 0
            ]
            for dead_gpu, exit_code in dead:
                log.error("Worker GPU %d died with exit code %d", dead_gpu, exit_code)
                pending_gpus.discard(dead_gpu)
            if not dead:
                alive = any(proc.is_alive() for gid, proc in workers if int(gid) in pending_gpus)
                if not alive:
                    log.error("All pending workers dead; aborting collection.")
                    break
            continue

        if int(gpu_id) not in pending_gpus:
            continue
        pending_gpus.discard(int(gpu_id))

        if error_text is not None:
            log.error("Worker GPU %d failed:\n%s", gpu_id, error_text)
        else:
            collected[int(gpu_id)] = results
            log.info("Worker GPU %d completed: %d tasks", gpu_id, len(results))

    for _, process in workers:
        process.join(timeout=30)

    all_results: list[AndroidWorldTaskResult] = []
    for gpu_id, _ in assignments:
        all_results.extend(collected.get(gpu_id, ()))
    all_results.sort(key=lambda r: r.task_name)

    if not all_results and collected:
        log.warning("No results collected from any worker.")

    report = build_androidworld_report(all_results, config=config, variant_id=variant_id)
    if output_path is not None:
        _save_androidworld_report(report, output_path)
        if config.measure_end_to_end_latency:
            _write_androidworld_timing_summary(all_results, output_path)

    return report


def _build_androidworld_timing_summary(
    results: Sequence[AndroidWorldTaskResult],
) -> dict[str, dict[str, dict[str, float]]]:
    """Build timing summary in the same schema as AndroidControl partial_timing_summary."""
    from guiaccel.evaluation.partial_report import PhaseStats

    baseline: dict[str, PhaseStats] = {}
    for r in results:
        for s in r.steps:
            for key, value in s.model_timing.items():
                if value is not None:
                    baseline.setdefault(key, PhaseStats()).extend((float(value),))
    return {
        "baseline": {k: v.to_summary_entry() for k, v in sorted(baseline.items())},
        "hybrid": {},
    }


def _write_androidworld_timing_summary(
    results: Sequence[AndroidWorldTaskResult],
    output_dir: Path,
) -> None:
    """Write partial_timing_summary.json compatible with compare_prefill_timing.py."""
    import time as time_mod

    summary = _build_androidworld_timing_summary(results)
    step_count = sum(r.num_steps for r in results)
    path = output_dir / "partial_timing_summary.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", time_mod.gmtime()),
                "result_count": step_count,
                "timing_summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    log.info("Timing summary written to %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_max_steps(complexity: Any, max_steps_per_task: int) -> int:
    try:
        return max(1, int(complexity) * 10)
    except (TypeError, ValueError):
        return int(max_steps_per_task)


def _sum_token_dicts(dicts: Iterable[dict[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for d in dicts:
        for key, value in d.items():
            result[key] = result.get(key, 0) + int(value)
    return result


def _serialize_task_result(result: AndroidWorldTaskResult) -> dict[str, Any]:
    return {
        "task_name": result.task_name,
        "task_goal": result.task_goal,
        "task_params": result.task_params,
        "success": result.success,
        "num_steps": result.num_steps,
        "max_steps": result.max_steps,
        "total_latency_ms": result.total_latency_ms,
        "total_tokens": result.total_tokens,
        "error": result.error,
        "metadata": result.metadata,
        "steps": [
            {
                "step_index": s.step_index,
                "goal": s.goal,
                "routing_mode": s.routing_mode,
                "latency_ms": s.latency_ms,
                "token_usage": s.token_usage,
                "model_output_text": s.model_output_text,
                "error": s.error,
                "prefill_ms": s.prefill_ms,
                "decode_ms": s.decode_ms,
                "vision_encoder_ms": s.vision_encoder_ms,
                "model_latency_ms": s.model_latency_ms,
                "model_timing": s.model_timing,
            }
            for s in result.steps
        ],
    }


__all__ = [
    "AndroidWorldRunConfig",
    "AndroidWorldStepTrace",
    "AndroidWorldTaskResult",
    "build_androidworld_report",
    "run_androidworld_evaluation",
    "_build_fallback_model_and_spec",
]
