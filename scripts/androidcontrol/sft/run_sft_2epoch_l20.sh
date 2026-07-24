#!/bin/bash
# =============================================================================
# run_sft_2epoch_l20.sh — 2-epoch full SFT + LoRA merge on L20 (Phase A)
#
# Purpose: Run SFT training (2 epochs, full data) + merge only.
# Expected time: ~6-10h on 4× L20 (256G)
#
# Usage (no --time; cluster default partition limit):
#   srun --gres=gpu:4 --partition=l20-gpu --cpus-per-task=16 --mem=256G \
#       bash /dkucc/home/rw335/SkillReuse/scripts/androidcontrol/sft/run_sft_2epoch_l20.sh
#
# Monitor:
#   tail -f /dkucc/home/rw335/SkillReuse/logs/sft_2epoch_l20.log
#   tail -f /dkucc/home/rw335/SkillReuse/logs/sft_2epoch.log
# =============================================================================
set -euo pipefail

PROJ=/dkucc/home/rw335/SkillReuse
PYTHON=/dkucc/home/rw335/.conda/envs/skillreuse/bin/python
SWIFT_BIN=/dkucc/home/rw335/.conda/envs/skillreuse/bin/swift
LOG="${PROJ}/logs/sft_2epoch_l20.log"

cd "$PROJ"
mkdir -p "${PROJ}/logs" "${PROJ}/outputs"
exec >> "$LOG" 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

timing_start() { TIMING_STAGE=$1; TIMING_T0=$(date +%s); log "[TIMING] stage=$TIMING_STAGE start"; }
timing_end() { log "[TIMING] stage=$TIMING_STAGE end elapsed=$(($(date +%s) - TIMING_T0))s"; }

log "========================================================"
log "SFT 2-Epoch L20 START (Phase A)"
log "PROJ=$PROJ  LOG=$LOG"
log "========================================================"

# ── [0] 验证前置条件 ──────────────────────────────────────────────────────────
BASE_MODEL_PATH=$("$PYTHON" -c "
import json, sys
d = json.load(open('configs/androidcontrol/default.json'))
p = d.get('model', {}).get('source', '')
if not p or p == '__SET_BY_SETUP_SH__':
    print('ERROR: run bash scripts/setup/setup.sh first', file=sys.stderr); sys.exit(1)
print(p)
")
CONFIG="$PROJ/configs/androidcontrol/default.json"
MERGED_PATH="$PROJ/outputs/sft_merged"
SWIFT_JSONL="$PROJ/outputs/sft_data/androidcontrol_train.jsonl"

log "Base model: $BASE_MODEL_PATH"

# ── [1] 准备全量 SFT 数据（并行 20 shards，~5-10 分钟）──────────────────────
if [ -f "$SWIFT_JSONL" ]; then
    _JSONL_N=$(wc -l < "${SWIFT_JSONL}")
    log "[1/4] SFT 数据已存在 (${_JSONL_N} 条)，跳过"
    timing_start 1
    timing_end
else
    log "=== [1/4] 准备 SFT 数据（全量，并行处理）==="
    timing_start 1
    "$PYTHON" scripts/data/convert_androidcontrol_to_swift.py \
        --config "$CONFIG" \
        --output-dir "$PROJ/outputs/sft_data" \
        --workers 20 \
        --instruction-modes high_level low_level
    _JSONL_N=$(wc -l < "${SWIFT_JSONL}")
    log "SFT 数据完成: ${_JSONL_N} 条"
    timing_end
fi

# ── [1b] SFT 前确保 GPU 空闲（杀掉所有 vLLM / setsid 残留进程）────────────
SFT_DONE="$PROJ/outputs/sft_output/.sft_done"
if [ ! -f "$SFT_DONE" ]; then
    log "=== [1b/4] 释放 GPU（杀 vLLM 进程）==="
    timing_start 1b
    # 从 pid 文件杀（smoke 或上一次 full 残留）
    for pid_file in "$PROJ/logs/vllm_smoke.pid" "$PROJ/logs/vllm_full.pid" "$PROJ/logs/vllm_2epoch_eval.pid"; do
        if [ -f "$pid_file" ]; then
            OLD_PID=$(cat "$pid_file" 2>/dev/null || true)
            if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
                log "  kill vLLM PID $OLD_PID (from $pid_file)"
                kill "$OLD_PID" 2>/dev/null || true
            fi
        fi
    done
    # 兜底：杀掉所有包含 vllm.entrypoints 的进程
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    # 等 GPU 显存完全释放（最多 60 秒）
    log "  等待 GPU 显存释放…"
    for i in $(seq 1 12); do
        VLLM_PROCS=$(pgrep -fc "vllm.entrypoints" 2>/dev/null) || VLLM_PROCS=0
        if [ "$VLLM_PROCS" -eq 0 ]; then
            log "  GPU 已空闲"
            break
        fi
        [ "$i" -eq 12 ] && log "  WARNING: 60s 后仍有 vLLM 进程，强制继续"
        sleep 5
    done
    sleep 3  # 额外等 CUDA context 释放
    timing_end
else
    log "[1b/4] SFT 已完成，跳过 GPU 清理"
    timing_start 1b
    timing_end
fi

# ── [2] Swift SFT（4× L20，2 epochs，使用验证格式）──────────────────────────
if [ -f "$SFT_DONE" ]; then
    log "[2/4] SFT 已完成，跳过"
    timing_start 2
    timing_end
else
    log "=== [2/4] Swift SFT (4× L20，2 epochs) ==="
    timing_start 2
    # Use SLURM-provided CUDA_VISIBLE_DEVICES when present; only fill if unset.
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
            export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
        else
            export CUDA_VISIBLE_DEVICES=0,1,2,3
        fi
    fi
    _ACTUAL_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())")
    if [[ "$_ACTUAL_GPUS" -lt 1 ]]; then
        log "ERROR: no CUDA GPUs visible (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
        exit 1
    fi
    export NPROC_PER_NODE="$_ACTUAL_GPUS"
    log "SFT GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
    RESUME_ARGS=()
    if [ -n "${RESUME_CKPT:-}" ]; then
        if [ ! -d "$RESUME_CKPT" ]; then
            log "ERROR: RESUME_CKPT set but path does not exist: $RESUME_CKPT"
            exit 1
        fi
        if [ ! -f "$RESUME_CKPT/adapter_config.json" ]; then
            log "ERROR: RESUME_CKPT missing adapter_config.json: $RESUME_CKPT"
            exit 1
        fi
        log "Resuming SFT from checkpoint: $RESUME_CKPT"
        RESUME_ARGS=(--resume_from_checkpoint "$RESUME_CKPT" --load_args true)
    fi
    "$SWIFT_BIN" sft \
        --model "$BASE_MODEL_PATH" \
        --dataset "$SWIFT_JSONL" \
        --output_dir "$PROJ/outputs/sft_output" \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --lora_rank 64 \
        --lora_alpha 16 \
        --bf16 true \
        --gradient_checkpointing true \
        --deepspeed zero2 \
        --dataloader_num_workers 4 \
        --save_strategy steps \
        --save_steps 500 \
        --logging_steps 20 \
        "${RESUME_ARGS[@]}" \
        2>&1 | tee "$PROJ/logs/sft_2epoch.log"
    touch "$SFT_DONE"
    log "SFT 完成"
    timing_end
fi

# ── [3] 找最新 checkpoint 并合并 LoRA ────────────────────────────────────────
if [ -f "$MERGED_PATH/config.json" ]; then
    log "[3/4] 合并模型已存在，跳过: $MERGED_PATH"
    timing_start 3
    timing_end
else
    log "=== [3/4] 合并 LoRA ==="
    timing_start 3
    LAST_CKPT=$("$PYTHON" -c "
from pathlib import Path
import sys
ckpts = list(Path('$PROJ/outputs/sft_output').rglob('adapter_config.json'))
if not ckpts:
    print('ERROR: no checkpoint found after SFT', file=sys.stderr); sys.exit(1)
latest = max(ckpts, key=lambda p: p.stat().st_mtime)
print(str(latest.parent))
")
    log "Last checkpoint: $LAST_CKPT"
    CUDA_VISIBLE_DEVICES=0 \
    "$SWIFT_BIN" export \
        --model "$BASE_MODEL_PATH" \
        --adapters "$LAST_CKPT" \
        --merge_lora true \
        --output_dir "$MERGED_PATH" \
        2>&1 | tee "$PROJ/logs/merge_2epoch.log"
    log "合并完成: $MERGED_PATH"
    timing_end
fi

# ── 汇总 ─────────────────────────────────────────────────────────────────────
log "========================================================"
log "SFT 2-Epoch L20 DONE (Phase A)"
log "SFT 模型: $MERGED_PATH"
log "SFT training log: $PROJ/logs/sft_2epoch.log"
log "Merge log: $PROJ/logs/merge_2epoch.log"
log "完整日志: $LOG"
log "========================================================"
