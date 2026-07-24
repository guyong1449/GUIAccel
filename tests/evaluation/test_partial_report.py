"""Unit tests for incremental partial evaluation reports."""

from __future__ import annotations

from types import SimpleNamespace

from skillreuse.evaluation.controller_metrics import compute_controller_metrics
from skillreuse.evaluation.models import StepEvaluationResult, StepJudgement
from skillreuse.evaluation.partial_report import (
    PartialReportContext,
    TimingSummaryAccumulator,
    build_worker_partial_report,
    merge_controller_metrics,
    merge_timing_summaries,
    write_aggregated_partial_reports,
)
from skillreuse.evaluation.timing import compute_timing_summary
from skillreuse.routing import RouteDecision, RoutedActionResult
from skillreuse.routing.common import TokenUsage
from skillreuse.routing.fallback import FallbackResult, FullModelRequest, FullModelResponse
from skillreuse.types import CanonicalAction, ScreenshotAsset


def _result(
    *,
    episode_id: str,
    step_index: int,
    total_e2e_ms: float,
    prompt_tokens: int = 100,
    visual_tokens: int = 50,
    generated_tokens: int = 20,
) -> StepEvaluationResult:
    action = CanonicalAction("CLICK", None, None, None, None)
    judgement = StepJudgement(action=action, metrics={"step_match": True}, primary_correct=True)
    example = SimpleNamespace(
        benchmark="AndroidControl",
        setting="high_level",
        partition="IDD",
        group_id=episode_id,
        episode_id=episode_id,
        step_index=step_index,
        repository_example=SimpleNamespace(dataset_step=SimpleNamespace(goal="goal")),
        instruction_mode="high_level",
    )
    token_usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        visual_tokens=visual_tokens,
        generated_tokens=generated_tokens,
        full_model_calls=1,
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
        estimated_token_usage=token_usage,
    )
    response = FullModelResponse(
        action=action,
        token_usage=token_usage,
        latency_ms=total_e2e_ms,
        raw_output="{}",
        model_name="test-model",
        timing={"total_e2e_step_ms": total_e2e_ms, "vision_encoder_ms": total_e2e_ms * 0.4},
    )
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
            reason="no_safe_skill",
        ),
        token_usage=token_usage,
        latency_ms=total_e2e_ms,
        fallback_result=fallback,
    )
    return StepEvaluationResult(
        example=example,
        ground_truth_action=action,
        baseline=judgement,
        hybrid=judgement,
        baseline_fallback=fallback,
        hybrid_routed=routed,
        baseline_end_to_end_latency_ms=total_e2e_ms,
        hybrid_end_to_end_latency_ms=total_e2e_ms,
        baseline_timing=dict(response.timing),
        hybrid_timing=dict(response.timing),
    )


def test_merge_timing_summaries_matches_full_compute():
    gpu0 = [_result(episode_id="1", step_index=0, total_e2e_ms=100.0)]
    gpu1 = [_result(episode_id="2", step_index=0, total_e2e_ms=300.0)]
    partial0 = build_worker_partial_report(gpu0)
    partial1 = build_worker_partial_report(gpu1)
    merged = merge_timing_summaries([partial0["timing_summary"], partial1["timing_summary"]])
    expected = compute_timing_summary(gpu0 + gpu1)
    assert merged["baseline"]["total_e2e_step_ms"]["mean_ms"] == expected["baseline"]["total_e2e_step_ms"]["mean_ms"]
    assert merged["baseline"]["total_e2e_step_ms"]["count"] == 2.0


def test_merge_controller_metrics_matches_full_compute():
    gpu0 = [_result(episode_id="1", step_index=0, total_e2e_ms=100.0, prompt_tokens=80, visual_tokens=40, generated_tokens=10)]
    gpu1 = [_result(episode_id="2", step_index=0, total_e2e_ms=200.0, prompt_tokens=120, visual_tokens=60, generated_tokens=30)]
    partial0 = build_worker_partial_report(gpu0)
    partial1 = build_worker_partial_report(gpu1)
    merged = merge_controller_metrics([partial0["controller_metrics"], partial1["controller_metrics"]])
    expected = compute_controller_metrics(gpu0 + gpu1)
    assert merged["baseline_tokens_total"] == expected["baseline_tokens_total"]
    assert merged["hybrid_end_to_end_latency_ms_total"] == expected["hybrid_end_to_end_latency_ms_total"]


def test_timing_summary_accumulator_incremental():
    first = [_result(episode_id="1", step_index=0, total_e2e_ms=100.0)]
    second = [_result(episode_id="2", step_index=0, total_e2e_ms=300.0)]
    accum = TimingSummaryAccumulator()
    accum.extend_from_results(first)
    accum.extend_from_results(second)
    summary = accum.to_summary()
    assert summary["baseline"]["total_e2e_step_ms"]["count"] == 2.0
    assert summary["baseline"]["total_e2e_step_ms"]["mean_ms"] == 200.0


def test_write_aggregated_partial_reports(tmp_path):
    gpu0 = build_worker_partial_report([_result(episode_id="1", step_index=0, total_e2e_ms=100.0)])
    gpu1 = build_worker_partial_report([_result(episode_id="2", step_index=0, total_e2e_ms=200.0)])
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    (progress_dir / "partial_report_gpu0.json").write_text(
        __import__("json").dumps({"gpu_id": 0, **gpu0}),
        encoding="utf-8",
    )
    (progress_dir / "partial_report_gpu1.json").write_text(
        __import__("json").dumps({"gpu_id": 1, **gpu1}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    context = PartialReportContext(
        benchmark="AndroidControl",
        evaluation_split="test",
        variant_id="V0",
        variant_description="Full baseline only",
    )
    written = write_aggregated_partial_reports(
        progress_dir=progress_dir,
        output_dir=output_dir,
        benchmark=context.benchmark,
        evaluation_split=context.evaluation_split,
        variant_id=context.variant_id,
        variant_description=context.variant_description,
    )
    assert "partial_timing_summary" in written
    assert "partial_controller_metrics" in written
    assert "partial_detailed_results" in written
    assert "partial_log_summary" in written
    payload = __import__("json").loads(written["partial_timing_summary"].read_text(encoding="utf-8"))
    assert payload["result_count"] == 2
    assert payload["timing_summary"]["baseline"]["total_e2e_step_ms"]["count"] == 2.0
