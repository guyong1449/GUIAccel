"""Journal utilities: config snapshot, pointers.json, run_summary.md for SLURM runs."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_EVAL_SUMMARY_SECTION = "## Evaluation Summary"

EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"
EVALUATION_SUMMARY_SECTION_START = "<!-- skillreuse:evaluation-summary:start -->"
EVALUATION_SUMMARY_SECTION_END = "<!-- skillreuse:evaluation-summary:end -->"


def write_eval_log_parse_prompt(run_dir: Path, *, log_path: str | Path | None = None) -> Path:
    """Write eval_log_parse_prompt.md beside slurm.out for agent consumption."""
    from guiaccel.evaluation.log_summary import (
        EVAL_LOG_PARSE_PROMPT_FILENAME,
        build_eval_log_parse_prompt,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_log = Path(log_path) if log_path is not None else run_dir / "slurm.out"
    out_path = run_dir / EVAL_LOG_PARSE_PROMPT_FILENAME
    out_path.write_text(
        build_eval_log_parse_prompt(log_path=resolved_log),
        encoding="utf-8",
    )
    return out_path


def get_job_context() -> dict[str, Any]:
    """Read SLURM environment variables into a structured dict."""
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_PARTITION",
        "SLURM_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_ACCOUNT",
    ]
    ctx: dict[str, Any] = {}
    for key in keys:
        val = os.environ.get(key)
        if val is not None:
            short = key.replace("SLURM_JOB_", "").replace("SLURM_", "").lower()
            ctx[short] = val
    ctx["hostname"] = platform.node()
    ctx["submitted_at"] = (
        os.environ.get("GUIACCEL_JOB_SUBMITTED_AT")
        or os.environ.get("SKILLREUSE_JOB_SUBMITTED_AT")
        or os.environ.get("VISIONZIP_JOB_SUBMITTED_AT")
        or os.environ.get("BASELINE_JOB_SUBMITTED_AT")
        or ""
    )
    return ctx


def resolve_journal_run_dir() -> Path | None:
    """Return the active journal run directory from known launcher env vars."""
    for env_key in (
        "GUIACCEL_JOURNAL_RUN_DIR",
        "VISIONZIP_JOURNAL_RUN_DIR",
        "BASELINE_JOURNAL_RUN_DIR",
        "SKILLREUSE_JOURNAL_RUN_DIR",
    ):
        value = os.environ.get(env_key, "").strip()
        if value:
            return Path(value)
    return None


def write_config_snapshot(run_dir: Path, config_path: str | Path) -> Path:
    """Copy the JSON config used for this run into the journal directory."""
    config_path = Path(config_path)
    snapshot_path = run_dir / "config_snapshot.json"
    if config_path.exists():
        snapshot_path.write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        snapshot_path.write_text(
            json.dumps({"error": f"config not found: {config_path}"}, indent=2),
            encoding="utf-8",
        )
    return snapshot_path


def write_pointers_file(
    run_dir: Path,
    *,
    config_path: str,
    output_dir: str,
    mode: str,
    vllm_topology: dict[str, Any] | None = None,
    status: str = "started",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write pointers.json with paths, job context, vLLM topology, and status."""
    pointers: dict[str, Any] = {
        "config_path": config_path,
        "output_dir": output_dir,
        "mode": mode,
        "status": status,
        "job_context": get_job_context(),
    }
    if vllm_topology:
        pointers["vllm_topology"] = vllm_topology
    if extra:
        pointers.update(extra)
    out = run_dir / "pointers.json"
    out.write_text(json.dumps(pointers, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_run_summary(
    run_dir: Path,
    *,
    mode: str,
    config_path: str,
    status: str = "started",
    comment: str = "",
) -> Path:
    """Write a human-readable run_summary.md."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_ctx = get_job_context()
    lines = [
        "# GUIAccel Run Summary",
        "",
        f"- mode: {mode}",
        f"- status: {status}",
        f"- config: {config_path}",
        f"- started: {now}",
        f"- slurm_job_id: {job_ctx.get('id', 'N/A')}",
        f"- partition: {job_ctx.get('partition', 'N/A')}",
        f"- node: {job_ctx.get('hostname', 'N/A')}",
    ]
    if comment:
        lines.append(f"- comment: {comment}")
    lines.append("")
    out = run_dir / "run_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def prepare_journal(
    run_dir: Path,
    *,
    config_path: str,
    output_dir: str,
    mode: str,
    comment: str = "",
    vllm_topology: dict[str, Any] | None = None,
) -> None:
    """Create all journal artifacts in the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(run_dir, config_path)
    write_eval_log_parse_prompt(run_dir)
    write_pointers_file(
        run_dir,
        config_path=config_path,
        output_dir=output_dir,
        mode=mode,
        vllm_topology=vllm_topology,
    )
    write_run_summary(run_dir, mode=mode, config_path=config_path, comment=comment)


def write_evaluation_summary(
    run_dir: Path,
    summary: Mapping[str, Any],
    *,
    report_path: str | None = None,
) -> Path:
    """Persist evaluation_summary.json and update journal pointers/summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / EVALUATION_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pointers_path = run_dir / "pointers.json"
    if pointers_path.exists():
        pointers = json.loads(pointers_path.read_text(encoding="utf-8"))
    else:
        pointers = {"job_context": get_job_context()}
    pointers["evaluation_summary_path"] = str(summary_path)
    if report_path:
        pointers["evaluation_report_path"] = report_path
    pointers["evaluation_summary"] = dict(summary)
    pointers_path.write_text(
        json.dumps(pointers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _append_evaluation_summary_section(run_dir / "run_summary.md", summary, report_path=report_path)
    return summary_path


def persist_evaluation_summary_to_journal(
    summary: Mapping[str, Any],
    *,
    report_path: str | None = None,
) -> Path | None:
    """Write journal evaluation artifacts when a journal run dir env var is set."""
    run_dir = resolve_journal_run_dir()
    if run_dir is None:
        return None
    return write_evaluation_summary(run_dir, summary, report_path=report_path)


def _append_evaluation_summary_section(
    summary_path: Path,
    summary: Mapping[str, Any],
    *,
    report_path: str | None,
) -> None:
    section_lines = [_EVAL_SUMMARY_SECTION, ""]
    if report_path:
        section_lines.append(f"- eval_report: {report_path}")
    run_metadata = summary.get("run_metadata")
    if isinstance(run_metadata, Mapping):
        for key in ("benchmark", "evaluation_split", "variant_id", "result_count", "backend"):
            value = run_metadata.get(key)
            if value not in (None, ""):
                section_lines.append(f"- {key}: {value}")
    overall = summary.get("overall_accuracy")
    if isinstance(overall, Mapping):
        for key, value in overall.items():
            section_lines.append(f"- {key}: {value}")
    eval_json = summary.get("eval_json")
    if isinstance(eval_json, Mapping):
        section_lines.append(
            f"- eval_json: path={eval_json.get('path') or 'N/A'}, "
            f"exists={eval_json.get('exists', False)}, readable={eval_json.get('readable', False)}"
        )
    section_lines.append("")
    section_text = "\n".join(section_lines)

    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
        marker_index = existing.find(_EVAL_SUMMARY_SECTION)
        if marker_index >= 0:
            existing = existing[:marker_index].rstrip() + "\n\n"
        else:
            existing = existing.rstrip() + "\n\n"
        summary_path.write_text(existing + section_text, encoding="utf-8")
    else:
        summary_path.write_text(section_text, encoding="utf-8")


def append_evaluation_summary_section(run_dir: Path, formatted_summary: str) -> Path:
    """Insert or replace a bounded evaluation summary section in run_summary.md."""
    summary_path = run_dir / "run_summary.md"
    section_lines = [
        "",
        "## Evaluation Summary",
        "",
        EVALUATION_SUMMARY_SECTION_START,
        formatted_summary.rstrip(),
        EVALUATION_SUMMARY_SECTION_END,
        "",
    ]
    section = "\n".join(section_lines)
    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8")
        start = text.find(EVALUATION_SUMMARY_SECTION_START)
        end = text.find(EVALUATION_SUMMARY_SECTION_END)
        if start >= 0 and end > start:
            end += len(EVALUATION_SUMMARY_SECTION_END)
            updated = text[:start].rstrip() + section + text[end:].lstrip("\n")
            if updated and not updated.endswith("\n"):
                updated += "\n"
            summary_path.write_text(updated, encoding="utf-8")
            return summary_path
        existing = text.rstrip()
        summary_path.write_text(existing + section, encoding="utf-8")
        return summary_path

    summary_path.write_text("# GUIAccel Run Summary\n" + section, encoding="utf-8")
    return summary_path


def update_pointers_with_evaluation(
    run_dir: Path,
    *,
    eval_json_path: str | None = None,
    evaluation_summary_path: str | None = None,
) -> Path:
    """Update pointers.json with evaluation artifact paths."""
    pointers_path = run_dir / "pointers.json"
    if pointers_path.exists():
        data = json.loads(pointers_path.read_text(encoding="utf-8"))
    else:
        data = {}
    if eval_json_path:
        data["eval_json_path"] = eval_json_path
    if evaluation_summary_path:
        data["evaluation_summary_path"] = evaluation_summary_path
    pointers_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return pointers_path


def persist_evaluation_summary_artifacts(
    run_dir: Path,
    report: Mapping[str, Any],
    *,
    formatted_summary: str | None = None,
) -> dict[str, Path]:
    """Write evaluation summary JSON, markdown section, and pointer updates."""
    from guiaccel.evaluation.log_summary import format_evaluation_summary

    log_summary = dict(report.get("log_summary") or {})
    if not log_summary:
        raise ValueError("report does not contain log_summary")
    rendered = formatted_summary or format_evaluation_summary(log_summary)
    summary_json_path = run_dir / EVALUATION_SUMMARY_FILENAME
    summary_json_path.write_text(
        json.dumps(log_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_md_path = append_evaluation_summary_section(run_dir, rendered)
    pointers_path = update_pointers_with_evaluation(
        run_dir,
        eval_json_path=str(report.get("output_path") or log_summary.get("metadata", {}).get("eval_json_path") or ""),
        evaluation_summary_path=str(summary_json_path),
    )
    return {
        "evaluation_summary_json": summary_json_path,
        "run_summary_md": summary_md_path,
        "pointers_json": pointers_path,
    }


def update_status(run_dir: Path, status: str) -> None:
    """Update status in pointers.json and run_summary.md."""
    pointers_path = run_dir / "pointers.json"
    summary_path = run_dir / "run_summary.md"
    if pointers_path.exists():
        data = json.loads(pointers_path.read_text(encoding="utf-8"))
        data["status"] = status
        pointers_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("- status:"):
                lines[i] = f"- status: {status}"
                break
        summary_path.write_text("\n".join(lines), encoding="utf-8")
