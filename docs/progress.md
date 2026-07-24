# Current Progress

## Objective

Compare Qwen2-VL-7B target-only greedy decoding with Qwen2-VL-7B + MSD on the
offline AndroidControl dataset. Qwen3-VL migration is explicitly out of scope
until this comparison is reproducible.

## Current state

- AndroidWorld modules, launchers, tests, configuration, and canonicalization
  branches have been removed.
- The required Qwen2-VL MSD inference core from `/dkucc/home/rw335/MSD` commit
  `fd76a5e` is vendored under `guiaccel/model/msd/`; runtime `sys.path`
  dependency on the separate MSD checkout has been removed.
- `MSDQwen2VLBackend` exposes GUIAccel's benchmark-faithful request/response
  interface and switches between `naivegenerate` and `msdgenerate`.
- `scripts/androidcontrol/eval_msd.py` evaluates high-level and low-level
  AndroidControl instructions and reports step accuracy, episode accuracy,
  parse failures, latency, and accepted tokens per verification round.
- AndroidControl TFRecord dry-run and the new unit tests pass.

## Blockers

- Conda environment `msd-androidcontrol` has the pinned torch, transformers,
  and accelerate versions, and vendored `EaModel` imports successfully.
- Runtime dependencies are complete: torch 2.1.2+cu121, torchvision
  0.16.2+cu121, qwen-vl-utils, transformers 4.48.3, and accelerate 1.3.0 import
  successfully.
- Qwen2-VL target and MSD draft weights are not present locally.
- The existing GUIAccel test suite still has legacy `skillreuse.*` imports.
- Codex managed commands inherit proxy variables but cannot see the interactive
  shell's `127.0.0.1:17890`; the user verified proxy access in their own shell.

## Next steps

1. From the user's proxy-capable shell, run
   `scripts/setup/download_msd_models_home.sh`.
2. Verify both Hugging Face snapshots under
   `/dkucc/home/rw335/.cache/huggingface/hub`.
3. Resolve the cached snapshots in an offline Slurm allocation.
4. Run paired 16-step naive/MSD smokes with identical samples and settings.
5. Require greedy output equivalence, stable action parsing, and positive
   latency benefit before scaling to the full test split.
