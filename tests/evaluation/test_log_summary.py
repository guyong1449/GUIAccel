"""Unit tests for evaluation log summary extraction and formatting."""

from __future__ import annotations

from types import SimpleNamespace

from skillreuse.evaluation.log_summary import (
    SUMMARY_FOOTER,
    SUMMARY_HEADER,
    attach_log_summary_to_report,
    build_eval_log_parse_prompt,
    build_structured_log_summary,
    compute_mode_benchmark_metrics,
    format_evaluation_summary,
    format_mode_benchmark_progress_line,
    format_partial_progress_stdout,
    inspect_eval_json_path,
)
from skillreuse.routing.fallback import ModelRuntimeSpec


def _synthetic_android_report() -> dict:
    return {
        "benchmark": "AndroidControl",
        "evaluation_split": "test",
        "variant_id": "V0",
        "variant_description": "baseline only",
        "result_count": 4,
        "official_metrics": {
            "in_distribution": {
                "baseline": {
                    "high_level_step_accuracy": 0.5,
                    "low_level_step_accuracy": 0.75,
                    "high_level_episode_accuracy": 0.5,
                },
                "hybrid": {
                    "high_level_step_accuracy": 0.5,
                    "low_level_step_accuracy": 0.75,
                    "high_level_episode_accuracy": 0.5,
                },
            },
            "overall": {
                "baseline": {
                    "high_level_step_accuracy": 0.5,
                    "low_level_step_accuracy": 0.75,
                    "high_level_episode_accuracy": 0.5,
                },
                "hybrid": {
                    "high_level_step_accuracy": 0.5,
                    "low_level_step_accuracy": 0.75,
                    "high_level_episode_accuracy": 0.5,
                },
            },
        },
        "controller_metrics": {
            "baseline_tokens_total": 1000,
            "hybrid_tokens_total": 400,
            "baseline_tokens_visual": 800,
            "hybrid_tokens_visual": 200,
            "baseline_tokens_input_text": 100,
            "hybrid_tokens_input_text": 100,
            "baseline_tokens_output": 100,
            "hybrid_tokens_output": 100,
            "total_token_reduction": 0.6,
            "visual_token_reduction": 0.75,
            "input_text_token_reduction": 0.0,
            "output_token_reduction": 0.0,
            "baseline_end_to_end_latency_ms_total": 4000.0,
            "hybrid_end_to_end_latency_ms_total": 2000.0,
            "latency_reduction": 0.5,
        },
        "null_action_diagnostics": {
            "baseline_null_action_count": 1,
            "baseline_null_action_rate": 0.25,
            "hybrid_null_action_count": 0,
            "hybrid_null_action_rate": 0.0,
        },
    }


def test_build_log_summary_android_includes_metadata_and_metrics() -> None:
    report = _synthetic_android_report()
    backend_config = SimpleNamespace(
        kind="divprune",
        keep_ratio=0.098,
        attn_implementation={"vision_config": "eager", "text_config": "sdpa"},
    )
    model_spec = ModelRuntimeSpec(
        benchmark="AndroidControl",
        backend="divprune",
        base_model_path="/models/qwen3-vl",
        model_name="Qwen3-VL-8B-Instruct",
    )

    summary = build_structured_log_summary(
        report,
        output_dir="/tmp/eval_out",
        eval_json_path="/tmp/eval_out/androidcontrol_test_v0_evaluation.json",
        backend_config=backend_config,
        model_spec=model_spec,
    )

    metadata = summary["metadata"]
    assert metadata["benchmark"] == "AndroidControl"
    assert metadata["evaluation_split"] == "test"
    assert metadata["variant_id"] == "V0"
    assert metadata["backend_kind"] == "divprune"
    assert metadata["vit_attn_implementation"] == "eager"
    assert metadata["llm_attn_implementation"] == "sdpa"
    assert metadata["keep_ratio"] == 0.098
    assert summary["accuracy"]["by_subset"]["in_distribution"]["hybrid"]["low_level_step_accuracy"] == 0.75
    assert summary["tokens"]["mean_hybrid_tokens_total_per_step"] == 100.0
    assert summary["latency"]["speedup_vs_baseline"] == 2.0


def test_attach_log_summary_to_report_mutates_report() -> None:
    report = _synthetic_android_report()
    summary = attach_log_summary_to_report(report, output_dir="/tmp/out")
    assert report["log_summary"] is summary
    assert "metadata" in report["log_summary"]


def test_format_partial_progress_stdout_includes_tokens_and_latency() -> None:
    rendered = format_partial_progress_stdout(
        {
            "completed": 1000,
            "target": 18412,
            "hl_step_accuracy": 0.444,
            "ll_step_accuracy": 0.547,
            "null_action_rate": 0.038,
            "steps_per_second": 0.05,
            "progress_percent": 5.4,
            "elapsed_s": 19953.0,
        },
        controller_metrics={
            "hybrid_tokens_total": 1894900,
            "hybrid_tokens_visual": 689700,
            "hybrid_tokens_output": 214900,
            "hybrid_end_to_end_latency_ms_total": 3828100.0,
        },
        gpu_id=0,
    )
    assert "GPU0" in rendered
    assert "HL=44.4%" in rendered
    assert "LL=54.7%" in rendered
    assert "tokens_total=1894900" in rendered
    assert "output=214900" in rendered
    assert "mean_step_lat_ms=3828.1" in rendered
    assert "elapsed=5h32m33s" in rendered


def test_build_eval_log_parse_prompt_uses_runtime_example_line() -> None:
    prompt = build_eval_log_parse_prompt(log_path="/tmp/slurm.out")
    assert "Log file: /tmp/slurm.out" in prompt
    assert "[Partial metrics @" in prompt
    assert "tokens_total=1894900" in prompt
    assert "elapsed=5h32m33s" in prompt
    assert SUMMARY_HEADER in prompt
    assert "grep '[Partial metrics @'" in prompt


def test_format_elapsed_compact() -> None:
    from skillreuse.evaluation.log_summary import _format_elapsed

    assert _format_elapsed(45.0) == "45s"
    assert _format_elapsed(125.0) == "2m05s"
    assert _format_elapsed(19953.0) == "5h32m33s"


def test_format_evaluation_summary_is_bounded_and_grep_friendly() -> None:
    report = _synthetic_android_report()
    summary = build_structured_log_summary(report, output_dir="/tmp/out")
    rendered = format_evaluation_summary(summary)

    assert rendered.startswith(SUMMARY_HEADER)
    assert SUMMARY_FOOTER in rendered
    assert "benchmark=AndroidControl" in rendered
    assert "in_distribution:" in rendered
    assert "hybrid_total=400" in rendered
    assert "speedup_vs_baseline=2.0000" in rendered
    assert rendered.count("\n") < 40


def test_inspect_eval_json_path_missing_and_readable(tmp_path) -> None:
    missing = inspect_eval_json_path(tmp_path / "missing.json")
    assert missing["exists"] is False
    assert missing["readable"] is False

    valid = tmp_path / "report.json"
    valid.write_text('{"benchmark": "AndroidControl"}', encoding="utf-8")
    readable = inspect_eval_json_path(valid)
    assert readable["exists"] is True
    assert readable["readable"] is True

    broken = tmp_path / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    unreadable = inspect_eval_json_path(broken)
    assert unreadable["exists"] is True
    assert unreadable["readable"] is False


def test_compute_mode_benchmark_metrics_splits_hl_and_ll() -> None:
    from tests.evaluation.test_partial_report import _result

    hl_ep1 = _result(
        episode_id="hl-1",
        step_index=0,
        total_e2e_ms=1000.0,
        prompt_tokens=100,
        visual_tokens=50,
        generated_tokens=20,
    )
    hl_ep1.example.instruction_mode = "high_level"
    hl_ep2 = _result(
        episode_id="hl-2",
        step_index=0,
        total_e2e_ms=2000.0,
        prompt_tokens=200,
        visual_tokens=100,
        generated_tokens=40,
    )
    hl_ep2.example.instruction_mode = "high_level"
    ll_ep = _result(
        episode_id="ll-1",
        step_index=0,
        total_e2e_ms=3000.0,
        prompt_tokens=300,
        visual_tokens=150,
        generated_tokens=60,
    )
    ll_ep.example.instruction_mode = "low_level"

    metrics = compute_mode_benchmark_metrics((hl_ep1, hl_ep2, ll_ep))
    assert metrics["high_level"]["step_accuracy"] == 1.0
    assert metrics["high_level"]["avg_input_tokens_per_episode"] == 225.0
    assert metrics["high_level"]["avg_output_tokens_per_episode"] == 30.0
    assert metrics["high_level"]["step_latency_s"] == 1.5
    assert metrics["high_level"]["e2e_latency_s"] == 1.5
    assert metrics["low_level"]["step_accuracy"] == 1.0
    assert metrics["low_level"]["avg_input_tokens_per_episode"] == 450.0
    assert metrics["low_level"]["avg_output_tokens_per_episode"] == 60.0
    assert metrics["low_level"]["step_latency_s"] == 3.0
    assert metrics["low_level"]["e2e_latency_s"] == 3.0


def test_format_mode_benchmark_progress_line_matches_table_columns() -> None:
    rendered = format_mode_benchmark_progress_line(
        "high_level",
        {
            "step_accuracy": 0.5527,
            "avg_input_tokens_per_episode": 16182.0,
            "avg_output_tokens_per_episode": 652.0,
            "step_latency_s": 4.963,
            "e2e_latency_s": 27.88,
        },
        completed=1000,
        target=18412,
    )
    assert "[Partial metrics @ 1000/18412 AGG] high_level" in rendered
    assert "step_acc=0.5527" in rendered
    assert "avg_in_tok/ep=16182" in rendered
    assert "avg_out_tok/ep=652" in rendered
    assert "step_lat_s=4.963" in rendered
    assert "episode_lat_s=27.88" in rendered
