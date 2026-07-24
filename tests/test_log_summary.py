"""Tests for evaluation log summary helpers."""

from __future__ import annotations

from skillreuse.evaluation.log_summary import (
    SUMMARY_HEADER,
    build_log_summary,
    format_log_summary,
)


def _android_report() -> dict:
    return {
        "benchmark": "AndroidControl",
        "evaluation_split": "test",
        "variant_id": "V0",
        "variant_description": "baseline only",
        "result_count": 100,
        "dataset_manifest": "manifest.json",
        "official_metrics": {
            "overall": {
                "hybrid": {
                    "high_level_step_accuracy": 0.4435,
                    "low_level_step_accuracy": 0.5469,
                    "high_level_episode_accuracy": 0.0,
                }
            },
            "in_distribution": {
                "hybrid": {
                    "high_level_step_accuracy": 0.4744,
                    "low_level_step_accuracy": 0.5694,
                }
            },
            "app_unseen": {
                "hybrid": {
                    "high_level_step_accuracy": 0.4198,
                    "low_level_step_accuracy": 0.5299,
                }
            },
        },
        "controller_metrics": {
            "baseline_tokens_total": 51516801,
            "hybrid_tokens_total": 34889667,
            "total_token_reduction": 0.323,
            "baseline_tokens_visual": 23871290,
            "hybrid_tokens_visual": 12699596,
            "visual_token_reduction": 0.468,
            "baseline_tokens_input_text": 25506702,
            "hybrid_tokens_input_text": 20040258,
            "baseline_tokens_output": 2138809,
            "hybrid_tokens_output": 2149813,
            "baseline_end_to_end_latency_ms_total": 71021376.0,
            "hybrid_end_to_end_latency_ms_total": 70483676.0,
            "latency_reduction": 0.0076,
        },
        "null_action_diagnostics": {
            "hybrid_null_action_rate": 0.0381,
            "hybrid_null_action_count": 702,
        },
        "output_path": "/tmp/androidcontrol_test_v0_evaluation.json",
    }


def _learngui_report() -> dict:
    return {
        "benchmark": "LearnGUI",
        "evaluation_split": "test",
        "variant_id": "V0",
        "result_count": 20,
        "official_metrics": {
            "1_shot": {
                "hybrid": {
                    "action_type_accuracy": 0.75,
                    "action_match_accuracy": 0.60,
                }
            }
        },
        "controller_metrics": {
            "baseline_tokens_total": 1000,
            "hybrid_tokens_total": 800,
            "total_token_reduction": 0.2,
            "baseline_tokens_visual": 400,
            "hybrid_tokens_visual": 300,
            "visual_token_reduction": 0.25,
            "baseline_end_to_end_latency_ms_total": 2000.0,
            "hybrid_end_to_end_latency_ms_total": 1500.0,
            "latency_reduction": 0.25,
        },
        "null_action_diagnostics": {},
    }


def test_build_log_summary_android_includes_core_sections() -> None:
    summary = build_log_summary(_android_report())

    assert summary["run_metadata"]["benchmark"] == "AndroidControl"
    assert summary["overall_accuracy"]["hl_step_accuracy_hybrid"] == "44.35%"
    assert summary["per_subset_accuracy"][0]["subset"] == "in_distribution"
    assert summary["per_subset_accuracy"][0]["metrics"]["ll_step_hybrid"] == "56.94%"
    assert summary["tokens_compression"]["visual_token_compression"] == "46.80%"
    assert summary["latency_speedup"]["hybrid_speedup_vs_baseline"] == "1.01x"
    assert summary["eval_json"]["path"].endswith("androidcontrol_test_v0_evaluation.json")


def test_build_log_summary_learngui_uses_shot_groups() -> None:
    summary = build_log_summary(_learngui_report())

    assert summary["overall_accuracy"]["1_shot_action_match_hybrid"] == "60.00%"
    assert summary["per_subset_accuracy"][0]["subset"] == "1_shot"
    assert summary["per_subset_accuracy"][0]["metrics"]["action_type_hybrid"] == "75.00%"


def test_build_log_summary_handles_missing_metrics() -> None:
    summary = build_log_summary({"benchmark": "AndroidControl", "result_count": 0})

    assert summary["overall_accuracy"]["hl_step_accuracy_hybrid"] == "N/A"
    assert summary["per_subset_accuracy"] == []
    assert summary["tokens_compression"]["total_tokens_baseline"] == 0
    assert summary["eval_json"]["path"] is None


def test_format_log_summary_starts_with_header_and_includes_sections() -> None:
    rendered = format_log_summary(build_log_summary(_android_report()))

    assert rendered.startswith(SUMMARY_HEADER)
    assert "[Run Metadata]" in rendered
    assert "[Overall Accuracy]" in rendered
    assert "[Per-Subset Accuracy]" in rendered
    assert "[Tokens & Compression]" in rendered
    assert "[Latency & Speedup]" in rendered
    assert "[Eval JSON Source]" in rendered
