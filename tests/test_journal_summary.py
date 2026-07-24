"""Tests for journal evaluation summary persistence."""

from __future__ import annotations

import json
from pathlib import Path

from skillreuse.evaluation.log_summary import build_log_summary
from skillreuse.journal import get_job_context, resolve_journal_run_dir, write_evaluation_summary


def test_get_job_context_reads_visionzip_submitted_at(monkeypatch) -> None:
    monkeypatch.delenv("SKILLREUSE_JOB_SUBMITTED_AT", raising=False)
    monkeypatch.setenv("VISIONZIP_JOB_SUBMITTED_AT", "20260628_120000")

    ctx = get_job_context()

    assert ctx["submitted_at"] == "20260628_120000"


def test_write_evaluation_summary_updates_journal_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "journal_run"
    run_dir.mkdir()
    (run_dir / "pointers.json").write_text(
        json.dumps({"status": "started", "mode": "eval"}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_summary.md").write_text("# SkillReuse Run Summary\n\n- status: started\n", encoding="utf-8")

    report = {
        "benchmark": "AndroidControl",
        "evaluation_split": "test",
        "variant_id": "V0",
        "result_count": 10,
        "official_metrics": {
            "overall": {
                "hybrid": {
                    "high_level_step_accuracy": 0.5,
                    "low_level_step_accuracy": 0.6,
                    "high_level_episode_accuracy": 0.0,
                }
            }
        },
        "controller_metrics": {},
        "null_action_diagnostics": {},
        "output_path": str(tmp_path / "androidcontrol_test_v0_evaluation.json"),
    }
    summary = build_log_summary(report)

    summary_path = write_evaluation_summary(
        run_dir,
        summary,
        report_path=report["output_path"],
    )

    assert summary_path.exists()
    pointers = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
    assert pointers["evaluation_summary_path"] == str(summary_path)
    assert pointers["evaluation_report_path"] == report["output_path"]
    assert pointers["evaluation_summary"]["run_metadata"]["benchmark"] == "AndroidControl"

    run_summary = (run_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "## Evaluation Summary" in run_summary
    assert report["output_path"] in run_summary


def test_write_evaluation_summary_replaces_existing_section(tmp_path: Path) -> None:
    run_dir = tmp_path / "journal_run"
    run_dir.mkdir()
    (run_dir / "run_summary.md").write_text(
        "# SkillReuse Run Summary\n\n## Evaluation Summary\n\n- old\n",
        encoding="utf-8",
    )
    summary = build_log_summary({"benchmark": "LearnGUI", "result_count": 1})

    write_evaluation_summary(run_dir, summary)

    run_summary = (run_dir / "run_summary.md").read_text(encoding="utf-8")
    assert run_summary.count("## Evaluation Summary") == 1
    assert "- old" not in run_summary
    assert "LearnGUI" in run_summary


def test_resolve_journal_run_dir_prefers_visionzip_env(monkeypatch, tmp_path: Path) -> None:
    visionzip_dir = tmp_path / "visionzip_journal"
    baseline_dir = tmp_path / "baseline_journal"
    monkeypatch.setenv("VISIONZIP_JOURNAL_RUN_DIR", str(visionzip_dir))
    monkeypatch.setenv("BASELINE_JOURNAL_RUN_DIR", str(baseline_dir))

    assert resolve_journal_run_dir() == visionzip_dir


def test_resolve_journal_run_dir_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("VISIONZIP_JOURNAL_RUN_DIR", raising=False)
    monkeypatch.delenv("BASELINE_JOURNAL_RUN_DIR", raising=False)
    monkeypatch.delenv("SKILLREUSE_JOURNAL_RUN_DIR", raising=False)

    assert resolve_journal_run_dir() is None
