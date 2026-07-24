"""Unit tests for parallel evaluation resume helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from skillreuse.evaluation import orchestrator
from skillreuse.evaluation.models import EvaluationExample, StepEvaluationResult, StepJudgement
from skillreuse.evaluation.orchestrator import (
    _evaluation_example_key,
    _load_parallel_evaluation_chunk_results,
    _load_parallel_evaluation_resume_state,
    _make_parallel_evaluation_progress_dir,
    _parallel_evaluation_chunk_dir_path,
    _prepare_parallel_evaluation_output_dir,
)
from skillreuse.routing import RouteDecision, RoutedActionResult
from skillreuse.routing.common import TokenUsage
from skillreuse.routing.fallback import FallbackResult, FullModelRequest, FullModelResponse
from skillreuse.types import CanonicalAction, ScreenshotAsset


def _minimal_example(
    *,
    group_id: str,
    episode_id: str,
    step_index: int,
    instruction_mode: str,
) -> EvaluationExample:
    return EvaluationExample(
        benchmark="AndroidControl",
        setting=instruction_mode,
        partition="IDD",
        group_id=group_id,
        episode_id=episode_id,
        step_index=step_index,
        repository_example=SimpleNamespace(),
        instruction_mode=instruction_mode,
    )


def _minimal_result(
    *,
    group_id: str,
    episode_id: str,
    step_index: int,
    instruction_mode: str,
) -> StepEvaluationResult:
    action = CanonicalAction("CLICK", None, None, None, None)
    judgement = StepJudgement(action=action, metrics={"step_match": True}, primary_correct=True)
    example = _minimal_example(
        group_id=group_id,
        episode_id=episode_id,
        step_index=step_index,
        instruction_mode=instruction_mode,
    )
    request = FullModelRequest(
        observation_id="obs-1",
        reason="baseline_eval",
        benchmark="AndroidControl",
        screenshot=ScreenshotAsset(png_bytes=b"png"),
        prompt_text="prompt",
        history_length=0,
        support_context={},
        model_spec=None,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        image_max_pixels=1024,
        estimated_token_usage=TokenUsage(),
    )
    response = FullModelResponse(action=action, token_usage=TokenUsage(), latency_ms=1.0)
    fallback = FallbackResult(request=request, response=response)
    routed = RoutedActionResult(
        action=action,
        final_mode="fallback",
        route_decision=RouteDecision(
            selected_skill=None,
            selected_budget=None,
            selected_score=None,
            execution_result=None,
            ood_rejections=0,
            reason="baseline_eval",
        ),
        token_usage=TokenUsage(),
        latency_ms=1.0,
        fallback_result=fallback,
    )
    return StepEvaluationResult(
        example=example,
        ground_truth_action=action,
        baseline=judgement,
        hybrid=judgement,
        baseline_fallback=fallback,
        hybrid_routed=routed,
    )


def test_evaluation_example_key():
    example = _minimal_example(
        group_id="ep-1",
        episode_id="ep-1",
        step_index=3,
        instruction_mode="high_level",
    )
    assert _evaluation_example_key(example) == ("ep-1", "ep-1", 3)


def test_load_resume_state_empty(tmp_path: Path):
    latest_progress, states = _load_parallel_evaluation_resume_state(tmp_path, worker_gpus=(0, 1))
    assert latest_progress == {}
    assert states == {}


def test_load_resume_state_synthetic(tmp_path: Path):
    progress_dir = tmp_path / "parallel_evaluation_progress"
    chunks_dir = tmp_path / "parallel_evaluation_chunks"
    progress_dir.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)

    completed = _minimal_result(
        group_id="ep-1",
        episode_id="ep-1",
        step_index=0,
        instruction_mode="high_level",
    )
    chunk_path = chunks_dir / "eval_results_gpu0_chunk0001.pkl"
    with chunk_path.open("wb") as handle:
        pickle.dump({"results": (completed,)}, handle)

    (progress_dir / "progress_gpu0.json").write_text(
        json.dumps(
            {
                "gpu_id": 0,
                "phase": "batched_runtime",
                "status": "running",
                "completed_examples": 1,
                "total_examples": 4,
            }
        )
    )

    latest_progress, states = _load_parallel_evaluation_resume_state(tmp_path, worker_gpus=(0, 1))
    assert latest_progress[0]["phase"] == "batched_runtime"
    assert set(states) == {0}
    state = states[0]
    assert state.gpu_id == 0
    assert state.completed_examples == 1
    assert state.total_examples == 4
    assert state.chunk_paths == (str(chunk_path.resolve()),)

    loaded = _load_parallel_evaluation_chunk_results(state.chunk_paths)
    assert len(loaded) == 1
    assert _evaluation_example_key(loaded[0].example) == _evaluation_example_key(completed.example)


def test_make_checkpoint_dirs_fresh_wipes(tmp_path: Path):
    progress_dir = tmp_path / "parallel_evaluation_progress"
    chunks_dir = tmp_path / "parallel_evaluation_chunks"
    report_path = tmp_path / "androidcontrol_test_v0_evaluation.json"
    progress_dir.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)
    report_path.write_text("{}")
    (progress_dir / "progress_gpu0.json").write_text("{}")
    (chunks_dir / "eval_results_gpu0_chunk0001.pkl").write_bytes(b"stale")

    resolved_output_dir = _prepare_parallel_evaluation_output_dir(
        tmp_path,
        benchmark="AndroidControl",
        evaluation_split="test",
        variant_id="V0",
        resume=False,
    )

    assert resolved_output_dir == tmp_path.resolve()
    assert not progress_dir.exists()
    assert not chunks_dir.exists()
    assert not report_path.exists()


def test_make_checkpoint_dirs_resume_preserves(tmp_path: Path):
    output_dir = tmp_path / "rerun_output"
    progress_dir = output_dir / "parallel_evaluation_progress"
    progress_dir.mkdir(parents=True)
    progress_file = progress_dir / "progress_gpu0.json"
    progress_file.write_text('{"phase": "batched_runtime"}')

    made_progress = _make_parallel_evaluation_progress_dir(output_dir, resume=True)

    assert made_progress == progress_dir
    assert progress_file.read_text() == '{"phase": "batched_runtime"}'
    assert _parallel_evaluation_chunk_dir_path(output_dir) == output_dir / "parallel_evaluation_chunks"
    shutil.rmtree(output_dir, ignore_errors=True)


def test_load_resume_state_reads_resume_artifacts(tmp_path: Path):
    output_dir = tmp_path / "rerun_output"
    progress_dir = output_dir / "parallel_evaluation_progress"
    chunks_dir = output_dir / "parallel_evaluation_chunks"
    progress_dir.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)

    completed = _minimal_result(
        group_id="ep-1",
        episode_id="ep-1",
        step_index=0,
        instruction_mode="high_level",
    )
    aggregate_progress_path = output_dir / "parallel_evaluation_progress.json"
    progress_path = progress_dir / "progress_gpu0.json"
    chunk_path = chunks_dir / "eval_results_gpu0_chunk0001.pkl"
    aggregate_progress_path.write_text(
        json.dumps({"worker_progress": {"0": {"phase": "batched_runtime", "status": "running"}}}),
        encoding="utf-8",
    )
    progress_path.write_text(
        json.dumps({"phase": "batched_runtime", "status": "running", "completed_examples": 1, "total_examples": 4}),
        encoding="utf-8",
    )
    with chunk_path.open("wb") as handle:
        pickle.dump({"results": (completed,)}, handle)

    latest_progress, states = _load_parallel_evaluation_resume_state(output_dir, worker_gpus=(0,))

    assert latest_progress[0]["phase"] == "batched_runtime"
    assert states[0].completed_examples == 1
    assert states[0].chunk_paths == (str(chunk_path.resolve()),)
    shutil.rmtree(output_dir, ignore_errors=True)


def test_short_circuit_only_on_completed_resume(tmp_path: Path):
    helper = getattr(orchestrator, "_should_short_circuit_evaluation", None)
    assert helper is not None

    progress_path = tmp_path / "parallel_evaluation_progress.json"
    report_path = tmp_path / "androidcontrol_test_v0_evaluation.json"
    report_path.write_text(json.dumps({"score": 1.0}), encoding="utf-8")
    progress_path.write_text(json.dumps({"phase": "running", "status": "running"}), encoding="utf-8")

    assert not helper(
        resume_from_output_dir=False,
        output_dir=tmp_path,
        benchmark="AndroidControl",
        variant="V0",
        split="test",
    )
    assert not helper(
        resume_from_output_dir=True,
        output_dir=tmp_path,
        benchmark="AndroidControl",
        variant="V0",
        split="test",
    )

    progress_path.write_text(json.dumps({"phase": "complete", "status": "completed"}), encoding="utf-8")
    assert helper(
        resume_from_output_dir=True,
        output_dir=tmp_path,
        benchmark="AndroidControl",
        variant="V0",
        split="test",
    )


def test_load_completed_parallel_report(tmp_path: Path):
    helper = getattr(orchestrator, "_load_completed_parallel_evaluation_report", None)
    assert helper is not None

    output_dir = tmp_path / "rerun_output"
    output_dir.mkdir()
    progress_path = output_dir / "parallel_evaluation_progress.json"
    report_path = output_dir / "androidcontrol_test_v0_evaluation.json"
    progress_path.write_text(json.dumps({"phase": "complete", "status": "completed"}), encoding="utf-8")
    report_path.write_text(json.dumps({"score": 0.5}), encoding="utf-8")

    loaded = helper(
        output_dir,
        benchmark="AndroidControl",
        evaluation_split="test",
        variant_id="V0",
    )

    assert loaded == {"score": 0.5, "output_path": str(report_path.resolve())}
    shutil.rmtree(output_dir, ignore_errors=True)
