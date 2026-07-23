#!/usr/bin/env python
"""Run an arbitrary command and tee stdout/stderr to a log file with timestamps."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR = "logs"


def run_with_log(
    command: list[str] | str,
    log_file: str | None = None,
    append: bool = False,
    realtime: bool = True,
) -> int:
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{DEFAULT_LOG_DIR}/run_{timestamp}.log"

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as log_f:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write(
            f"Command: {' '.join(command) if isinstance(command, list) else command}\n"
        )
        log_f.write(f"Start: {start_time}\n")
        log_f.write("=" * 80 + "\n\n")
        log_f.flush()

        process = subprocess.Popen(
            command if isinstance(command, list) else command,
            shell=isinstance(command, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        try:
            for line in process.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                log_f.write(line + "\n")
                log_f.flush()
                if realtime:
                    print(line)

            return_code = process.wait()
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"End: {end_time}\n")
            log_f.write(f"Return code: {return_code}\n")
            log_f.write("=" * 80 + "\n")
            log_f.flush()
            return return_code
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            log_f.write("\n[Interrupted]\n")
            log_f.flush()
            return 130


def main():
    parser = argparse.ArgumentParser(
        description="Run a command with stdout/stderr tee to a log file."
    )
    parser.add_argument("--log", "-l", type=str, default=None, help="Log file path")
    parser.add_argument("--append", "-a", action="store_true", help="Append to log")
    parser.add_argument(
        "--no-realtime", action="store_true", help="Suppress stdout echo"
    )
    args, unknown = parser.parse_known_args()

    if not unknown:
        print("Error: no command specified", file=sys.stderr)
        sys.exit(1)

    command: list[str] | str = [a for a in unknown if a != "--"]
    if len(command) == 1 and " " in command[0]:
        command = command[0]

    code = run_with_log(
        command,
        log_file=args.log,
        append=args.append,
        realtime=not args.no_realtime,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
