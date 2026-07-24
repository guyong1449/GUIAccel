#!/usr/bin/env python
"""Scan training_journal and optionally delete junk/incomplete runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    _lib = _parent / "_lib"
    if (_lib / "repo_path.py").is_file():
        sys.path.insert(0, str(_lib))
        break
else:
    raise RuntimeError(f"Could not locate scripts/_lib from {__file__}")
from repo_path import bootstrap

REPO_ROOT = bootstrap(Path(__file__))


BAD_STATUSES = {"failed", "failed_pre_promote", "aborted_pre_promote"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan training_journal and optionally delete junk runs."
    )
    parser.add_argument(
        "--journal-root",
        default=str(REPO_ROOT / "training_journal"),
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually delete selected runs"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Delete without interactive confirmation"
    )
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Remove empty directories under training_journal (including empty run_* shells)",
    )
    return parser.parse_args()


def remove_empty_dirs(journal_root: Path, *, apply: bool, yes: bool) -> list[Path]:
    empty_dirs = [
        path
        for path in sorted(journal_root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        if path.is_dir() and not any(path.iterdir())
    ]
    if journal_root.exists() and journal_root.is_dir() and not any(journal_root.iterdir()):
        empty_dirs.append(journal_root)

    if not empty_dirs:
        print("No empty directories found under training_journal.")
        return []

    print("Empty directories:")
    for path in empty_dirs:
        print(f"  {path}")
    print(f"\nTotal: {len(empty_dirs)} empty dir(s)")

    if not apply:
        print("\nDry run only. Re-run with --remove-empty-dirs --apply to delete.")
        return empty_dirs

    for path in empty_dirs:
        if not yes:
            answer = input(f"Delete empty dir {path}? [y/N] ").strip().lower()
            if answer != "y":
                continue
        path.rmdir()
        print(f"  Removed: {path}")
    return empty_dirs


def read_status(run_dir: Path) -> str:
    pointers_path = run_dir / "pointers.json"
    if pointers_path.exists():
        try:
            data = json.loads(pointers_path.read_text(encoding="utf-8"))
            return data.get("status", "unknown")
        except Exception:
            pass
    summary_path = run_dir / "run_summary.md"
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- status:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def is_run_active(run_dir: Path) -> bool:
    pointers_path = run_dir / "pointers.json"
    if not pointers_path.exists():
        return False
    try:
        payload = json.loads(pointers_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    job_id = (payload.get("job_context") or {}).get("id")
    if not job_id:
        return False
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", str(job_id)],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except Exception:
        return False
    return bool(out.strip())


def main():
    args = parse_args()
    journal_root = Path(args.journal_root)

    if args.remove_empty_dirs:
        remove_empty_dirs(journal_root, apply=args.apply, yes=args.yes)

    if not journal_root.exists():
        if not args.remove_empty_dirs:
            print(f"Journal root does not exist: {journal_root}")
        return

    candidates: list[tuple[Path, str]] = []

    run_dirs = sorted(journal_root.glob("20*/run_*"))
    tentative_dir = journal_root / ".tentative"
    if tentative_dir.exists():
        run_dirs.extend(sorted(tentative_dir.glob("run_*")))

    for run_dir in run_dirs:
        if not any(run_dir.iterdir()):
            candidates.append((run_dir, "empty"))
            continue
        status = read_status(run_dir)
        active = is_run_active(run_dir)
        if active:
            continue
        if status in BAD_STATUSES or status == "unknown":
            candidates.append((run_dir, status))

    if not candidates:
        print("No candidate runs found for cleanup.")
        return

    print("Candidates for deletion:")
    for run_dir, status in candidates:
        print(f"  {run_dir}  [status={status}]")
    print(f"\nTotal: {len(candidates)} run(s)")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete.")
        return

    for run_dir, status in candidates:
        if not args.yes:
            answer = (
                input(f"Delete {run_dir.name} (status={status})? [y/N] ").strip().lower()
            )
            if answer != "y":
                continue
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"  Deleted: {run_dir}")


if __name__ == "__main__":
    main()
