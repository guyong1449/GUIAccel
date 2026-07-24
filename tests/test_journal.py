from __future__ import annotations

import json
from pathlib import Path

from skillreuse.journal import (
    EVALUATION_SUMMARY_SECTION_END,
    EVALUATION_SUMMARY_SECTION_START,
    append_evaluation_summary_section,
    persist_evaluation_summary_artifacts,
    prepare_journal,
    update_pointers_with_evaluation,
    update_status,
    write_evaluation_summary,
)


def test_prepare_journal_writes_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260628_020525"
    config_path = tmp_path / "config.json"
    run_dir.mkdir()
    config_path.write_text('{"benchmark": "AndroidControl"}', encoding="utf-8")

    prepare_journal(
        run_dir,
        config_path=str(config_path),
        output_dir=str(tmp_path / "outputs"),
        mode="divprune_eval",
        comment="rerun",
    )

    assert (run_dir / "config_snapshot.json").exists()
    assert (run_dir / "eval_log_parse_prompt.md").exists()
    assert "[Partial metrics @" in (run_dir / "eval_log_parse_prompt.md").read_text(encoding="utf-8")
    pointers = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
    assert pointers["status"] == "started"
    assert (run_dir / "run_summary.md").read_text(encoding="utf-8").find("- status: started") >= 0


def test_write_evaluation_summary_and_pointers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260628_020525"
    run_dir.mkdir()
    log_summary = {
        "metadata": {"benchmark": "AndroidControl", "variant_id": "V0"},
        "accuracy": {"overall": {}},
    }
    summary_path = write_evaluation_summary(run_dir, log_summary)
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["benchmark"] == "AndroidControl"

    pointers_path = update_pointers_with_evaluation(
        run_dir,
        eval_json_path="/tmp/out/report.json",
        evaluation_summary_path=str(summary_path),
    )
    pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    assert pointers["eval_json_path"] == "/tmp/out/report.json"
    assert pointers["evaluation_summary_path"] == str(summary_path)


def test_append_evaluation_summary_section_replaces_existing_block(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260628_020525"
    run_dir.mkdir()
    summary_path = run_dir / "run_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# SkillReuse Run Summary",
                "",
                "- status: started",
                "",
                "## Evaluation Summary",
                "",
                EVALUATION_SUMMARY_SECTION_START,
                "old summary",
                EVALUATION_SUMMARY_SECTION_END,
                "",
            ]
        ),
        encoding="utf-8",
    )

    append_evaluation_summary_section(run_dir, "=== SkillReuse Evaluation Summary ===\nnew summary")
    text = summary_path.read_text(encoding="utf-8")
    assert "old summary" not in text
    assert "new summary" in text
    assert text.count(EVALUATION_SUMMARY_SECTION_START) == 1


def test_persist_evaluation_summary_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260628_020525"
    run_dir.mkdir()
    (run_dir / "pointers.json").write_text(json.dumps({"status": "started"}), encoding="utf-8")
    (run_dir / "run_summary.md").write_text("# SkillReuse Run Summary\n\n- status: started\n", encoding="utf-8")

    report = {
        "output_path": str(tmp_path / "androidcontrol_test_v0_evaluation.json"),
        "log_summary": {
            "metadata": {
                "benchmark": "AndroidControl",
                "evaluation_split": "test",
                "variant_id": "V0",
                "result_count": 1,
                "eval_json": {"exists": True, "readable": True, "error": None},
            },
            "accuracy": {
                "overall": {"hybrid": {"low_level_step_accuracy": 1.0}},
                "by_subset": {},
                "null_actions": {"hybrid_null_action_count": 0, "hybrid_null_action_rate": 0.0},
            },
            "tokens": {
                "hybrid_tokens_total": 10,
                "hybrid_tokens_visual": 5,
                "hybrid_tokens_input_text": 3,
                "hybrid_tokens_output": 2,
                "mean_hybrid_tokens_total_per_step": 10.0,
                "mean_hybrid_tokens_visual_per_step": 5.0,
                "mean_hybrid_tokens_input_text_per_step": 3.0,
                "mean_hybrid_tokens_output_per_step": 2.0,
                "total_token_reduction": 0.0,
                "visual_token_reduction": 0.0,
            },
            "latency": {
                "hybrid_end_to_end_latency_ms_total": 100.0,
                "mean_step_lat_ms": 100.0,
                "latency_reduction": 0.0,
                "speedup_vs_baseline": None,
            },
        },
    }

    artifacts = persist_evaluation_summary_artifacts(run_dir, report)
    assert artifacts["evaluation_summary_json"].exists()
    pointers = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
    assert pointers["eval_json_path"] == report["output_path"]
    assert "=== SkillReuse Evaluation Summary ===" in (run_dir / "run_summary.md").read_text(
        encoding="utf-8"
    )


def test_update_status_updates_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260628_020525"
    run_dir.mkdir()
    (run_dir / "pointers.json").write_text(json.dumps({"status": "started"}), encoding="utf-8")
    (run_dir / "run_summary.md").write_text(
        "# SkillReuse Run Summary\n\n- status: started\n",
        encoding="utf-8",
    )

    update_status(run_dir, "failed")

    pointers = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
    assert pointers["status"] == "failed"
    assert "- status: failed" in (run_dir / "run_summary.md").read_text(encoding="utf-8")
