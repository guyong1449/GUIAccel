"""Download AndroidControl from the official Google Research bucket with verification.

This script is intentionally stdlib-only so it can run in a minimal environment.
It downloads split metadata plus all TFRecord shards into a local directory,
keeps in-progress bytes in ``.part`` files, and only promotes a file to its
final name after the downloaded size matches the official object size.

cd SkillReuse

python scripts/run_with_log.py \
  --log logs/download_androidcontrol.log \
  -- "conda run -p .conda/maiui-vllm python scripts/download_androidcontrol.py \
  --output-dir data/androidcontrol --workers 8 --retries 10"
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BUCKET_PREFIX = "https://storage.googleapis.com/gresearch/android_control/"
BUCKET_LIST_URL = "https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=android_control/"
for _parent in Path(__file__).resolve().parents:
    _lib = _parent / "_lib"
    if (_lib / "repo_path.py").is_file():
        sys.path.insert(0, str(_lib))
        break
else:
    raise RuntimeError(f"Could not locate scripts/_lib from {__file__}")
from repo_path import bootstrap

REPO_ROOT = bootstrap(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "androidcontrol"
CHUNK_SIZE = 1024 * 1024


def fetch_bucket_manifest() -> dict[str, int]:
    with urlopen(BUCKET_LIST_URL) as response:
        payload = json.load(response)
    manifest: dict[str, int] = {}
    for item in payload.get("items", []):
        name = str(item.get("name") or "")
        if not name.startswith("android_control/"):
            continue
        basename = name.split("/", 1)[1]
        if not basename or basename == "":
            continue
        if basename == "":
            continue
        manifest[basename] = int(item["size"])
    return manifest


def build_download_plan(bucket_manifest: dict[str, int]) -> list[tuple[str, int]]:
    wanted = []
    for name, size in sorted(bucket_manifest.items()):
        if name == "":
            continue
        if name == "android_control":
            continue
        wanted.append((name, int(size)))
    return wanted


def write_manifest(output_dir: Path) -> None:
    payload = {
        "benchmark": "AndroidControl",
        "prepared_path": str(output_dir.resolve()),
        "source": "official_google_research_gcs",
        "source_url": BUCKET_PREFIX,
        "notes": "Root contains AndroidControl TFRecord shards plus split metadata; prepared_images is optional.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_one_file(output_dir: Path, name: str, expected_size: int, retries: int) -> str:
    final_path = output_dir / name
    tmp_path = output_dir / f"{name}.part"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and final_path.stat().st_size == expected_size:
        return f"SKIP {name}"
    if final_path.exists():
        final_path.unlink()
    if tmp_path.exists() and tmp_path.stat().st_size > expected_size:
        tmp_path.unlink()

    attempt = 0
    while True:
        attempt += 1
        try:
            _resume_http_download(
                url=f"{BUCKET_PREFIX}{name}",
                tmp_path=tmp_path,
                expected_size=expected_size,
            )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt > retries:
                raise RuntimeError(f"Failed downloading {name}: {exc}") from exc
            time.sleep(min(10.0, 1.0 * attempt))
            continue

        actual_size = tmp_path.stat().st_size if tmp_path.exists() else 0
        if actual_size == expected_size:
            os.replace(tmp_path, final_path)
            return f"DONE {name}"
        if actual_size > expected_size:
            tmp_path.unlink(missing_ok=True)
        if attempt > retries:
            raise RuntimeError(
                f"Downloaded size mismatch for {name}: expected {expected_size}, got {actual_size}."
            )
        time.sleep(min(10.0, 1.0 * attempt))


def _resume_http_download(*, url: str, tmp_path: Path, expected_size: int) -> None:
    start = tmp_path.stat().st_size if tmp_path.exists() else 0
    if start >= expected_size:
        return
    request = Request(url)
    if start > 0:
        request.add_header("Range", f"bytes={start}-")
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if start > 0 and status == 200:
            tmp_path.unlink(missing_ok=True)
            start = 0
        mode = "ab" if start > 0 else "wb"
        with tmp_path.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)


def summarize(output_dir: Path, plan: Iterable[tuple[str, int]]) -> str:
    ok = 0
    pending = 0
    for name, size in plan:
        final_path = output_dir / name
        if final_path.exists() and final_path.stat().st_size == size:
            ok += 1
        else:
            pending += 1
    return f"verified={ok} pending={pending}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AndroidControl with size verification.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_manifest = fetch_bucket_manifest()
    plan = build_download_plan(bucket_manifest)
    write_manifest(output_dir)

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_name: dict[Future[str], str] = {
            executor.submit(download_one_file, output_dir, name, size, int(args.retries)): name
            for name, size in plan
        }
        while future_to_name:
            done, _ = wait(tuple(future_to_name.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                name = future_to_name.pop(future)
                result = future.result()
                print(result, flush=True)
                print(summarize(output_dir, plan), flush=True)

    print("COMPLETE", summarize(output_dir, plan), flush=True)


if __name__ == "__main__":
    main()
