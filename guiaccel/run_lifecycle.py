"""Run lifecycle management: tentative → promote / abort for journal runs."""

import os
import shutil
import signal
from pathlib import Path


INTERRUPT_SIGNALS = (signal.SIGINT, signal.SIGTERM)

JOURNAL_DIR_NAME = "training_journal"
ENV_PREFIX = "SKILLREUSE"


def tentative_run_dir(repo_root: Path, run_stamp: str) -> Path:
    return repo_root / JOURNAL_DIR_NAME / ".tentative" / f"run_{run_stamp}"


def final_run_dir(repo_root: Path, run_date: str, run_stamp: str) -> Path:
    return repo_root / JOURNAL_DIR_NAME / run_date / f"run_{run_stamp}"


def update_env_run_dir(run_dir: Path):
    os.environ[f"{ENV_PREFIX}_JOURNAL_RUN_DIR"] = str(run_dir)


class RunLifecycle:
    """Manages a journal run directory through tentative → promoted or aborted states."""

    def __init__(self, tentative_dir: Path, final_dir: Path):
        self.tentative_dir = tentative_dir
        self.final_dir = final_dir
        self.current_dir = tentative_dir
        self.promoted = False
        self.tentative_dir.mkdir(parents=True, exist_ok=True)
        update_env_run_dir(self.current_dir)

    def promote(self) -> Path:
        if self.promoted:
            return self.current_dir
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.tentative_dir, self.final_dir)
        self.current_dir = self.final_dir
        self.promoted = True
        update_env_run_dir(self.current_dir)
        return self.current_dir

    def abort_cleanup(self, *, preserve_diagnostics: bool = True):
        if preserve_diagnostics and self.current_dir.exists():
            aborted_dir = (
                self.tentative_dir.parent.parent / ".aborted" / self.current_dir.name
            )
            aborted_dir.mkdir(parents=True, exist_ok=True)
            for name in [
                "slurm.out",
                "slurm.err",
                "terminal.log",
                "config_snapshot.json",
                "pointers.json",
                "run_summary.md",
                "vllm_health.json",
            ]:
                src = self.current_dir / name
                if src.exists():
                    shutil.copy2(src, aborted_dir / name)
        shutil.rmtree(self.current_dir, ignore_errors=True)


class SignalScope:
    """Context manager that captures SIGINT/SIGTERM and raises KeyboardInterrupt."""

    def __init__(self):
        self._previous: dict[int, object] = {}
        self.triggered_signal: int | None = None

    def _handler(self, signum, frame):
        self.triggered_signal = signum
        raise KeyboardInterrupt(f"Received signal {signum}")

    def __enter__(self):
        for signum in INTERRUPT_SIGNALS:
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        return False
