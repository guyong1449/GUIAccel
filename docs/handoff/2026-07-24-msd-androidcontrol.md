# Handoff Prompt: Qwen2-VL MSD on AndroidControl

You are continuing work in `/dkucc/home/rw335/GUIAccel`.

## Required startup

1. Read `/dkucc/home/rw335/AI_PROFILE/agent-core.md` if it exists. It was
   missing during the previous session.
2. Read `/dkucc/home/rw335/.codex/AGENTS.md`.
3. Read repository instructions in `agents.md`.
4. Read `docs/progress.md` and `docs/memory/2026-07-24.md`.
5. Preserve all unrelated user changes in this very dirty worktree.

## Approved scope

Reproduce and compare two Qwen2-VL-7B conditions on the AndroidControl test
split:

1. Target-only greedy decoding through MSD's `naivegenerate`.
2. The same target plus MSD draft through `msdgenerate`.

Use the same target model, samples, prompts, order, temperature=0, token limit,
image preprocessing, and scoring. Do not implement Qwen3-VL yet.

Report:

- high-level and low-level step accuracy;
- high-level episode accuracy;
- action parse-failure rate;
- end-to-end latency and speedup;
- peak GPU memory;
- generated tokens;
- MSD verification rounds and accepted tokens per round.

Start with a paired 16-step smoke. Scale only after both paths are correct.

## What has been implemented

- MSD Qwen2-VL inference core is vendored under `guiaccel/model/msd/`.
- Source: `/dkucc/home/rw335/MSD`, commit
  `fd76a5ee2bd107a5a04f05afd0651aab7ba6fab4`.
- Attribution: `guiaccel/model/msd/NOTICE.md` and `LICENSE.EAGLE`.
- Backend: `guiaccel/model/msd_qwen2vl_backend.py`.
- Evaluation CLI: `scripts/androidcontrol/eval_msd.py`.
- Config: `configs/androidcontrol/msd_qwen2vl.json`.
- Requirements: `requirements-msd-qwen2vl.txt`.
- Setup: `scripts/setup/setup_msd_androidcontrol.sh`.
- Slurm launcher:
  `run_scripts/androidcontrol/eval/eval_msd_qwen2vl_slurm.sh`.
- Tests: `tests/test_msd_qwen2vl_backend.py`.
- AndroidWorld source/config/script/test references were removed.

## Current verified state

- AndroidControl TFRecord dry-run successfully read test episode 140 and built
  both instruction modes.
- `pytest -q tests/test_msd_qwen2vl_backend.py`: 2 passed.
- Slurm launcher shell syntax and dry-run passed.
- Resource snapshot is stored in `.claude_resources.json`; current login shell
  has no GPU.

## Immediate blocker

`msd-androidcontrol` exists but is not usable:

```text
conda run -n msd-androidcontrol python -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

The user can access `http://127.0.0.1:17890` from their interactive shell.
Codex managed commands inherit proxy variables but cannot connect to that
loopback port. Do not claim the user's proxy is broken.

Ask the user to finish, or continue if dependencies are now present:

```bash
cd /dkucc/home/rw335/GUIAccel
bash scripts/setup/setup_msd_androidcontrol.sh
```

Then verify:

```bash
conda run -n msd-androidcontrol python -c \
  "import torch,transformers,accelerate; print(torch.__version__,transformers.__version__,accelerate.__version__)"

conda run -n msd-androidcontrol python -c \
  "from guiaccel.model.msd.core.ea_model import EaModel; print('ok')"
```

Expected pinned versions are torch 2.1.2, transformers 4.48.3, and accelerate
1.3.0.

## Model weights and GPU execution

The target and draft are not cached yet:

- `Qwen/Qwen2-VL-7B-Instruct`
- `lucylyn/MSD-Qwen2VL-7B-Instruct`

Download/cache large weights under `/work/rw335`, not home. Compute-node jobs
cannot use the login node's loopback proxy, so download on the proxy-capable
login shell first and run Slurm jobs offline from shared storage.

Before submit, verify Slurm association and partitions. Use one A40 for the
smoke. Run paired conditions with separate new output directories:

```bash
bash run_scripts/androidcontrol/eval/eval_msd_qwen2vl_slurm.sh

MSD_BASELINE=true \
bash run_scripts/androidcontrol/eval/eval_msd_qwen2vl_slurm.sh
```

Do not submit a full test run until the paired smoke demonstrates valid action
parsing, comparable greedy outputs, no cache contamination across requests,
and measurable MSD acceleration.

## Important caveats

- The worktree contains many pre-existing modified/untracked files. Do not
  reset, clean, or overwrite unrelated work.
- Existing tests such as `tests/test_qwen_action_parsing.py` still import the
  old `skillreuse.*` package and fail during collection; this predates MSD.
- Do not run GPU inference on the login node.
- Keep output artifacts under `/work/rw335` and logs nested under
  `logs/msd_androidcontrol/`.
