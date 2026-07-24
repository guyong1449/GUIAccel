"""Download LearnGUI dataset from HuggingFace and generate manifest.json.

cd SkillReuse

python scripts/run_with_log.py \\
  --log logs/download_learngui.log \\
  -- "conda run -p .conda/envs/skillreuse python scripts/download_learngui.py \\
  --output-dir data/learngui --workers 8"
"""

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "learngui"
HF_REPO_ID = "lgy0404/LearnGUI"
HF_REPO_TYPE = "dataset"


def write_manifest(output_dir: Path) -> None:
    payload = {
        "benchmark": "LearnGUI",
        "notes": "Root contains LearnGUI offline assets: screenshots, element annotations, instruction annotations, task splits.",
        "prepared_path": str(output_dir.resolve()),
        "source": "learngui_official_release",
        "source_url": f"https://huggingface.co/datasets/{HF_REPO_ID}",
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_structure(output_dir: Path) -> list[str]:
    """Return list of missing expected paths (checked after ZIP extraction)."""
    offline = output_dir / "offline"
    expected = [
        offline / "screenshots",
        offline / "element_anno" / "element_anno",
        offline / "instruction_anno" / "instruction_anno",
        offline / "low_level_instructions.json",
        offline / "task_split.json",
    ]
    return [str(p) for p in expected if not p.exists()]


def _extract_zip(zip_path: Path, dest_dir: Path, *, label: str) -> None:
    """Extract a (possibly multi-part) ZIP archive into *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    unzip_exe = shutil.which("unzip")
    if unzip_exe is None:
        import zipfile
        print(f"  Extracting {label} with Python zipfile (single-part) …", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    else:
        print(f"  Extracting {label} with system unzip …", flush=True)
        cmd = [unzip_exe, "-q", "-o", str(zip_path), "-d", str(dest_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode not in (0, 1):  # unzip returns 1 for warnings
            print(f"  unzip stderr: {result.stderr.strip()}", flush=True)
            raise RuntimeError(f"unzip failed (rc={result.returncode}) for {zip_path}")


def extract_offline_zips(offline_dir: Path) -> None:
    """Extract ZIP archives under *offline_dir* if not already extracted."""
    print("Extracting LearnGUI offline archives …", flush=True)

    # Screenshots (multi-part: screenshot.zip + .z01-.z05); extract into offline/ → offline/screenshots/
    screenshots_dir = offline_dir / "screenshots"
    screenshot_zip = offline_dir / "screenshot.zip"
    if screenshots_dir.exists() and any(screenshots_dir.iterdir()):
        print("  screenshots: already extracted", flush=True)
    elif screenshot_zip.exists():
        _extract_zip(screenshot_zip, offline_dir, label="screenshots")
    else:
        print(f"  WARNING: screenshot.zip not found at {screenshot_zip}", flush=True)

    # Element annotations: element_anno.zip → offline/element_anno/element_anno/
    element_anno_dir = offline_dir / "element_anno" / "element_anno"
    element_anno_zip = offline_dir / "element_anno.zip"
    if element_anno_dir.exists() and any(element_anno_dir.iterdir()):
        print("  element_anno: already extracted", flush=True)
    elif element_anno_zip.exists():
        _extract_zip(element_anno_zip, element_anno_dir.parent, label="element_anno")
    else:
        print(f"  WARNING: element_anno.zip not found at {element_anno_zip}", flush=True)

    # Instruction annotations: instruction_anno.zip → offline/instruction_anno/instruction_anno/
    instruction_anno_dir = offline_dir / "instruction_anno" / "instruction_anno"
    instruction_anno_zip = offline_dir / "instruction_anno.zip"
    if instruction_anno_dir.exists() and any(instruction_anno_dir.iterdir()):
        print("  instruction_anno: already extracted", flush=True)
    elif instruction_anno_zip.exists():
        _extract_zip(instruction_anno_zip, instruction_anno_dir.parent, label="instruction_anno")
    else:
        print(f"  WARNING: instruction_anno.zip not found at {instruction_anno_zip}", flush=True)

    # Fix upstream typo: task_spilit.json → task_split.json
    spilit_path = offline_dir / "task_spilit.json"
    split_path = offline_dir / "task_split.json"
    if spilit_path.exists() and not split_path.exists():
        spilit_path.rename(split_path)
        print("  Renamed task_spilit.json → task_split.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LearnGUI dataset from HuggingFace.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers passed to snapshot_download.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    offline_dir = output_dir / "offline"

    missing = verify_structure(output_dir)
    if not missing:
        print(f"All expected assets already present in {output_dir}. Skipping download.", flush=True)
        write_manifest(output_dir)
        print("Manifest written.", flush=True)
        return

    print(f"Downloading {HF_REPO_ID} → {output_dir}", flush=True)
    print(f"Missing paths: {missing}", flush=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    # local_dir=output_dir so the HF repo's offline/ subtree lands at output_dir/offline/
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(output_dir),
        max_workers=max(1, int(args.workers)),
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print("Download complete.", flush=True)

    extract_offline_zips(offline_dir)

    missing_after = verify_structure(output_dir)
    if missing_after:
        print(f"ERROR: expected paths still missing after extraction: {missing_after}", flush=True)
        sys.exit(1)

    print("Structure verified.", flush=True)
    write_manifest(output_dir)
    print(f"Manifest written to {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
