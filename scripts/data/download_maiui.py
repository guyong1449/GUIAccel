"""Download Tongyi-MAI/MAI-UI-8B with official Hugging Face Hub APIs.

The helper only fetches the files needed for local serving:
- safetensors weights and any safetensors index
- tokenizer assets
- model and processor configs
- the chat template

Repeated runs are safe: huggingface_hub reuses cached blobs and resumes partial
downloads when the installed library version supports it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any

for _parent in Path(__file__).resolve().parents:
    _lib = _parent / "_lib"
    if (_lib / "repo_path.py").is_file():
        sys.path.insert(0, str(_lib))
        break
else:
    raise RuntimeError(f"Could not locate scripts/_lib from {__file__}")
from repo_path import bootstrap

REPO_ID = "Tongyi-MAI/MAI-UI-8B"
REPO_ROOT = bootstrap(Path(__file__))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "MAI-UI-8B"
DEFAULT_REVISION = "main"

CONFIG_BASENAMES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
}
TOKENIZER_BASENAMES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
TEMPLATE_BASENAMES = {
    "chat_template.jinja",
    "chat_template.json",
}
PRIMARY_SUMMARY_BASENAMES = {
    "chat_template.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
}


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Tongyi-MAI/MAI-UI-8B into a local serving directory."
    )
    parser.add_argument(
        "--output-dir",
        "--local-dir",
        dest="output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Local target directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Model revision to resolve on the Hub. Default: main",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory override.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to the local HF login/HF_TOKEN.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max concurrent downloads when snapshot download is available. Default: 8",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-fetch files even if they already exist in the cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download plan without downloading files.",
    )
    return parser.parse_args()


def load_huggingface_hub() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
        ) from exc
    return HfApi, hf_hub_download, snapshot_download


def supports_argument(func: Any, name: str) -> bool:
    return name in inspect.signature(func).parameters


def fetch_repo_files(api: Any, *, repo_id: str, revision: str, token: str | None) -> tuple[str, list[RepoFile]]:
    info = api.model_info(
        repo_id=repo_id,
        revision=revision,
        files_metadata=True,
        token=token,
    )
    repo_files: list[RepoFile] = []
    for sibling in info.siblings or []:
        path = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if not path:
            continue
        size = getattr(sibling, "size", None)
        repo_files.append(RepoFile(path=str(path), size=int(size) if size is not None else None))
    return getattr(info, "sha", None) or revision, sorted(repo_files, key=lambda item: item.path)


def build_download_plan(repo_files: list[RepoFile]) -> list[RepoFile]:
    selected: list[RepoFile] = []
    basenames: set[str] = set()
    for repo_file in repo_files:
        basename = PurePosixPath(repo_file.path).name
        if repo_file.path.endswith(".safetensors") or repo_file.path.endswith(".safetensors.index.json"):
            selected.append(repo_file)
            basenames.add(basename)
            continue
        if basename in CONFIG_BASENAMES or basename in TOKENIZER_BASENAMES or basename in TEMPLATE_BASENAMES:
            selected.append(repo_file)
            basenames.add(basename)

    if not any(item.path.endswith(".safetensors") for item in selected):
        raise RuntimeError(f"No safetensors weights were found in {REPO_ID}.")
    if "config.json" not in basenames:
        raise RuntimeError(f"{REPO_ID} is missing config.json.")
    if not ({"chat_template.jinja", "chat_template.json"} & basenames):
        raise RuntimeError(f"{REPO_ID} is missing a chat template file.")
    if "tokenizer_config.json" not in basenames:
        raise RuntimeError(f"{REPO_ID} is missing tokenizer_config.json.")
    if not (
        "tokenizer.json" in basenames
        or "tokenizer.model" in basenames
        or {"vocab.json", "merges.txt"} <= basenames
    ):
        raise RuntimeError(f"{REPO_ID} is missing the expected tokenizer assets.")
    return sorted(selected, key=lambda item: item.path)


def download_plan(
    snapshot_download: Any,
    hf_hub_download: Any,
    *,
    plan: list[RepoFile],
    output_dir: Path,
    revision: str,
    cache_dir: Path | None,
    token: str | None,
    workers: int,
    force_download: bool,
) -> None:
    allow_patterns = [item.path for item in plan]
    snapshot_kwargs: dict[str, Any] = {
        "repo_id": REPO_ID,
        "revision": revision,
        "allow_patterns": allow_patterns,
        "token": token,
    }
    if cache_dir is not None:
        snapshot_kwargs["cache_dir"] = str(cache_dir)
    if supports_argument(snapshot_download, "force_download"):
        snapshot_kwargs["force_download"] = force_download
    if supports_argument(snapshot_download, "resume_download"):
        snapshot_kwargs["resume_download"] = True
    if supports_argument(snapshot_download, "max_workers"):
        snapshot_kwargs["max_workers"] = max(1, workers)
    if supports_argument(snapshot_download, "local_dir"):
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_kwargs["local_dir"] = str(output_dir)
        if supports_argument(snapshot_download, "local_dir_use_symlinks"):
            snapshot_kwargs["local_dir_use_symlinks"] = False
        snapshot_download(**snapshot_kwargs)
        return

    # Older huggingface_hub builds lack snapshot_download(local_dir=...).
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        file_kwargs: dict[str, Any] = {
            "repo_id": REPO_ID,
            "filename": item.path,
            "revision": revision,
            "token": token,
        }
        if cache_dir is not None:
            file_kwargs["cache_dir"] = str(cache_dir)
        if supports_argument(hf_hub_download, "force_download"):
            file_kwargs["force_download"] = force_download
        if supports_argument(hf_hub_download, "resume_download"):
            file_kwargs["resume_download"] = True
        use_local_dir = supports_argument(hf_hub_download, "local_dir")
        if use_local_dir:
            file_kwargs["local_dir"] = str(output_dir)
            if supports_argument(hf_hub_download, "local_dir_use_symlinks"):
                file_kwargs["local_dir_use_symlinks"] = False
        downloaded = Path(hf_hub_download(**file_kwargs))
        if use_local_dir:
            continue
        target_path = output_dir / item.path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.stat().st_size == downloaded.stat().st_size:
            continue
        shutil.copy2(downloaded, target_path)


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def verify_download(output_dir: Path, plan: list[RepoFile], resolved_revision: str) -> None:
    missing: list[str] = []
    mismatched: list[str] = []
    verified: list[tuple[str, int]] = []
    total_bytes = 0
    for item in plan:
        local_path = output_dir / item.path
        if not local_path.exists():
            missing.append(item.path)
            continue
        local_size = local_path.stat().st_size
        if item.size is not None and local_size != item.size:
            mismatched.append(
                f"{item.path} expected={format_bytes(item.size)} actual={format_bytes(local_size)}"
            )
            continue
        verified.append((item.path, local_size))
        total_bytes += local_size

    if missing or mismatched:
        problems = []
        if missing:
            problems.append("missing=" + ", ".join(missing))
        if mismatched:
            problems.append("mismatch=" + "; ".join(mismatched))
        raise RuntimeError("Verification failed: " + " | ".join(problems))

    weight_files = [path for path, _ in verified if path.endswith(".safetensors")]
    support_files = [path for path, _ in verified if not path.endswith(".safetensors")]
    print(
        f"VERIFY repo={REPO_ID} revision={resolved_revision} dir={output_dir} "
        f"files={len(verified)} weights={len(weight_files)} support={len(support_files)} "
        f"total={format_bytes(total_bytes)}",
        flush=True,
    )

    summary_paths = []
    for path, size in verified:
        basename = PurePosixPath(path).name
        if path.endswith(".safetensors") or basename in PRIMARY_SUMMARY_BASENAMES:
            summary_paths.append((path, size))
    for path, size in summary_paths:
        print(f"OK {path} {format_bytes(size)}", flush=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    HfApi, hf_hub_download, snapshot_download = load_huggingface_hub()
    api = HfApi()
    resolved_revision, repo_files = fetch_repo_files(
        api,
        repo_id=REPO_ID,
        revision=args.revision,
        token=args.token,
    )
    plan = build_download_plan(repo_files)

    if args.dry_run:
        total_bytes = sum(item.size or 0 for item in plan)
        print(
            f"PLAN repo={REPO_ID} revision={resolved_revision} files={len(plan)} "
            f"total={format_bytes(total_bytes)} dir={output_dir}",
            flush=True,
        )
        for item in plan:
            size_text = format_bytes(item.size) if item.size is not None else "unknown"
            print(f"WANT {item.path} {size_text}", flush=True)
        return 0

    download_plan(
        snapshot_download,
        hf_hub_download,
        plan=plan,
        output_dir=output_dir,
        revision=args.revision,
        cache_dir=cache_dir,
        token=args.token,
        workers=max(1, int(args.workers)),
        force_download=bool(args.force_download),
    )
    verify_download(output_dir, plan, resolved_revision)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
