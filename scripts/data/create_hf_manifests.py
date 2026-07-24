"""Create manifest.json files for HuggingFace-downloaded datasets.

Usage (run on the training server with the HuggingFace data already cloned):

    python scripts/create_hf_manifests.py

This script:
1.  Writes  manifest.json  into the AndroidControl dataset root so that
    AndroidControlDataset can locate it via the configs.
2.  Extracts the LearnGUI ZIP archives (screenshots, element_anno,
    instruction_anno) if not already extracted, then writes  manifest.json.

Both manifest paths are the ones already hard-coded in every configs/*.json
file:
  AndroidControl:
    .../reece124/android_control/manifest.json
  LearnGUI:
    .../lgy0404/LearnGUI/manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default HuggingFace dataset roots (matching configs/*.json)
# ---------------------------------------------------------------------------
HF_BASE = Path(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search"
    "/ai-search/wangjunpeng06/reuse/data/huggingface.co/datasets"
)
AC_ROOT = HF_BASE / "reece124" / "android_control"
LG_ROOT = HF_BASE / "lgy0404" / "LearnGUI"


# ---------------------------------------------------------------------------
# AndroidControl manifest
# ---------------------------------------------------------------------------

def create_androidcontrol_manifest(ac_root: Path) -> Path:
    """Write manifest.json to *ac_root* and return its path."""
    shards = sorted(ac_root.glob("android_control-*"))
    splits_ok = (ac_root / "splits.json").exists()
    test_subsplits_ok = (ac_root / "test_subsplits.json").exists()

    if not shards:
        print(f"  WARNING: no android_control-* shard files found in {ac_root}",
              file=sys.stderr)
    if not splits_ok:
        print(f"  WARNING: splits.json not found in {ac_root}", file=sys.stderr)
    if not test_subsplits_ok:
        print(f"  WARNING: test_subsplits.json not found in {ac_root}", file=sys.stderr)

    payload = {
        "benchmark": "AndroidControl",
        "prepared_path": str(ac_root.resolve()),
        "source": "huggingface_mirror",
        "source_url": "https://huggingface.co/datasets/reece124/android_control",
        "shard_count": len(shards),
        "notes": (
            "Shards may be gzip-compressed TFRecord (auto-detected by "
            "iter_tfrecord_records in skillreuse/utils/tfrecord.py)."
        ),
    }
    manifest_path = ac_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  ✓ AndroidControl manifest written: {manifest_path}  "
          f"({len(shards)} shards, splits={'ok' if splits_ok else 'MISSING'}, "
          f"test_subsplits={'ok' if test_subsplits_ok else 'MISSING'})")
    return manifest_path


# ---------------------------------------------------------------------------
# LearnGUI helpers
# ---------------------------------------------------------------------------

def _extract_zip(zip_path: Path, dest_dir: Path, *, label: str) -> None:
    """Extract a (possibly multi-part) ZIP archive into *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    unzip_exe = shutil.which("unzip")
    if unzip_exe is None:
        # Fall back to Python's zipfile (single-part only)
        import zipfile
        print(f"    Extracting {label} with Python zipfile (single-part) …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    else:
        print(f"    Extracting {label} with system unzip …")
        cmd = [unzip_exe, "-q", "-o", str(zip_path), "-d", str(dest_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode not in (0, 1):  # unzip returns 1 for warnings
            print(f"    unzip stderr: {result.stderr.strip()}", file=sys.stderr)
            raise RuntimeError(f"unzip failed (rc={result.returncode}) for {zip_path}")


def _check_or_extract(archive_path: Path, expected_dir: Path, *, label: str) -> None:
    """Extract *archive_path* into *expected_dir.parent* if *expected_dir* absent."""
    if expected_dir.exists() and any(expected_dir.iterdir()):
        print(f"    {label}: already extracted at {expected_dir}")
        return
    if not archive_path.exists():
        print(f"    WARNING: {label} archive not found at {archive_path}", file=sys.stderr)
        return
    print(f"    {label}: extracting {archive_path.name} → {expected_dir.parent} …")
    _extract_zip(archive_path, expected_dir.parent, label=label)
    if expected_dir.exists():
        print(f"    ✓ {label} extracted")
    else:
        print(f"    WARNING: expected directory {expected_dir} still missing after extraction",
              file=sys.stderr)


def prepare_learngui_offline(offline_root: Path) -> None:
    """Extract LearnGUI ZIP archives under *offline_root* if not already done."""
    print("  Checking LearnGUI offline extractions …")

    # Screenshots (multi-part: screenshot.zip + .z01-.z05)
    screenshots_dir = offline_root / "screenshots"
    screenshot_zip = offline_root / "screenshot.zip"
    # For multi-part ZIPs, system unzip reassembles automatically when given the .zip part.
    _check_or_extract(screenshot_zip, screenshots_dir, label="screenshots")

    # Element annotations → offline/element_anno/element_anno/
    element_anno_dir = offline_root / "element_anno" / "element_anno"
    element_anno_zip = offline_root / "element_anno.zip"
    _check_or_extract(element_anno_zip, element_anno_dir, label="element_anno")

    # Instruction annotations → offline/instruction_anno/instruction_anno/
    instruction_anno_dir = offline_root / "instruction_anno" / "instruction_anno"
    instruction_anno_zip = offline_root / "instruction_anno.zip"
    _check_or_extract(instruction_anno_zip, instruction_anno_dir, label="instruction_anno")


def create_learngui_manifest(lg_root: Path) -> Path:
    """Prepare ZIP extractions (if needed) and write manifest.json."""
    offline_root = lg_root / "offline"
    if not offline_root.exists():
        print(f"  WARNING: offline/ directory not found under {lg_root}", file=sys.stderr)
    else:
        prepare_learngui_offline(offline_root)

    # Sanity checks
    screenshots_dir = offline_root / "screenshots"
    low_level_path = offline_root / "low_level_instructions.json"
    task_split_path = offline_root / "task_split.json"

    for path, label in [
        (screenshots_dir, "screenshots/"),
        (low_level_path, "low_level_instructions.json"),
        (task_split_path, "task_split.json"),
    ]:
        if path.exists():
            print(f"    ✓ {label} present")
        else:
            print(f"    WARNING: {label} not found at {path}", file=sys.stderr)

    payload = {
        "benchmark": "LearnGUI",
        "prepared_path": str(lg_root.resolve()),
        "source": "huggingface_mirror",
        "source_url": "https://huggingface.co/datasets/lgy0404/LearnGUI",
        "notes": (
            "Run this script once after cloning to extract screenshots and "
            "annotation ZIPs from offline/."
        ),
    }
    manifest_path = lg_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  ✓ LearnGUI manifest written: {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create manifest.json files for HuggingFace-downloaded datasets."
    )
    parser.add_argument(
        "--ac-root",
        default=str(AC_ROOT),
        metavar="DIR",
        help=f"AndroidControl dataset root (default: {AC_ROOT})",
    )
    parser.add_argument(
        "--lg-root",
        default=str(LG_ROOT),
        metavar="DIR",
        help=f"LearnGUI dataset root (default: {LG_ROOT})",
    )
    parser.add_argument(
        "--skip-androidcontrol",
        action="store_true",
        help="Skip AndroidControl manifest creation.",
    )
    parser.add_argument(
        "--skip-learngui",
        action="store_true",
        help="Skip LearnGUI manifest creation.",
    )
    args = parser.parse_args()

    if not args.skip_androidcontrol:
        ac_root = Path(args.ac_root)
        print(f"\n[AndroidControl]  root={ac_root}")
        if not ac_root.exists():
            print(f"  ERROR: directory does not exist: {ac_root}", file=sys.stderr)
        else:
            create_androidcontrol_manifest(ac_root)

    if not args.skip_learngui:
        lg_root = Path(args.lg_root)
        print(f"\n[LearnGUI]  root={lg_root}")
        if not lg_root.exists():
            print(f"  ERROR: directory does not exist: {lg_root}", file=sys.stderr)
        else:
            create_learngui_manifest(lg_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
