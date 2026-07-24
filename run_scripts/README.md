# run_scripts/ — SLURM 提交脚本索引

按 **数据集 × 测试类型** 分类。请使用下表中的 canonical 路径提交作业。

## 共享库 `_lib/`

| 文件 | 用途 |
|------|------|
| `slurm_helpers.sh` | 分区 / GRES / sbatch 辅助 |
| `journal_helpers.sh` | `training_journal/` 记录与 promote |
| `monitor_job.sh` | 作业监控 |

## AndroidControl `androidcontrol/`

### 正式评测 `eval/`

| 脚本 | 方法 / 后端 |
|------|-------------|
| `eval_baseline_transformers_slurm.sh` | Baseline transformers |
| `eval_divprune_slurm.sh` | DivPrune |
| `eval_visionzip_slurm.sh` | VisionZip |
| `eval_msd_qwen2vl_slurm.sh` | MSD-Qwen2-VL / naive 对照 |

```bash
bash run_scripts/androidcontrol/eval/eval_visionzip_slurm.sh [--resume] [--output-dir PATH]
bash run_scripts/androidcontrol/eval/eval_baseline_transformers_slurm.sh [--resume]
bash run_scripts/androidcontrol/eval/eval_divprune_slurm.sh [--resume]
bash run_scripts/androidcontrol/eval/eval_msd_qwen2vl_slurm.sh
```

### Timing 评测 `timing/`

| 脚本 | 说明 |
|------|------|
| `slurm_timing_baseline.sh` | 全量 baseline timing |
| `slurm_timing_baseline_smoke2pct.sh` | 2% smoke baseline |
| `slurm_timing_visionzip.sh` | 全量 VisionZip timing |
| `slurm_timing_visionzip_smoke2pct.sh` | 2% smoke VisionZip |
| `slurm_timing_visionzip_smoke2pct_qwen50.sh` | qwen50 变体 |
| `slurm_timing_visionzip_smoke2pct_paper56.sh` | paper56 变体 |

内部调用 `scripts/core/run_evaluation.py`，日志写入 `training_journal/`。详见 [`docs/commands.md`](../docs/commands.md)。

## LearnGUI `learngui/eval/`

| 脚本 | 说明 |
|------|------|
| `eval_learngui_slurm.sh` | Baseline 评测 |
| `eval_learngui_visionzip_slurm.sh` | VisionZip 评测 |

## SFT `sft/`

| 脚本 | 说明 |
|------|------|
| `sft_phase_a_job.sh` | Phase A SFT sbatch job |
| `sft_phase_a_supervisor.sh` | Login-node 自动 resume supervisor |

## Regression Head `regression_head/`

| 脚本 | 分区习惯 | 说明 |
|------|----------|------|
| `extract_4gpu.sh` | GPU ×4 | Hidden-state 提取 |
| `extract_smoke_1gpu.sh` | GPU ×1 smoke | 短 smoke |
| `train_cpu.sh` | **`common`** 多 CPU | 小 MLP，禁止 GPU |
| `eval_decode_4gpu.sh` | GPU ×4 | AR vs CoordHead decode eval |

Slurm `.out/.err` → `logs/regression_head/<script_stem>/`（见 `logs/README.md`）。

## 通用 Pipeline `pipeline/`

| 脚本 | 说明 |
|------|------|
| `submit_slurm.sh` | Discovery / eval / e2e 通用 SLURM 提交 |

## 目录结构

```
run_scripts/
├── _lib/                  # 共享 helper
├── androidcontrol/
│   ├── eval/              # 正式 long eval
│   └── timing/            # prefill/decode timing
├── learngui/eval/
├── regression_head/       # CoordHead extract / train_cpu / decode eval
├── sft/
└── pipeline/
```
