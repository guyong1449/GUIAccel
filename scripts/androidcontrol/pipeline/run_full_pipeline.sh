#!/bin/bash
# =============================================================================
# run_full_pipeline.sh — 完整流水线（SFT → V0 → Discovery → V1/V2/V3）
#
# 预期耗时：SFT ~6h，Discovery ~4-6h，Eval ~1-2h × 4，总计 ~15-20h
# 各步骤已有产物（.done 标记）则自动跳过，支持断点续跑。
#
# 启动方式（Jupyter cell，脱离终端防断连）：
#   source .env
#   nohup bash scripts/androidcontrol/pipeline/run_full_pipeline.sh &
#   echo "Full Pipeline PID: $!"
#
# 监控：
#   tail -f logs/full_pipeline.log   # 主进度
#   tail -f logs/sft_full.log        # SFT loss
#   tail -f logs/vllm_full.log       # vLLM
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_lib/load_env.sh"
skillreuse_load_env "${REPO_DIR}"

PROJ="${SKILLREUSE_REPO_ROOT}"
CONDA_PREFIX="${SKILLREUSE_CONDA_PREFIX:?Set SKILLREUSE_CONDA_PREFIX in .env}"
PYTHON="${CONDA_PREFIX}/bin/python"
SWIFT="${CONDA_PREFIX}/bin/swift"
LOG="$PROJ/logs/full_pipeline.log"

# ── 第一步：用绝对路径建目录，然后自重定向 ──────────────────────────────────
mkdir -p "$PROJ/logs" "$PROJ/outputs"
exec >> "$LOG" 2>&1   # 此后所有 stdout/stderr 都写入 LOG

cd "$PROJ"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "========================================================"
log "QwenReuse Full Pipeline START"
log "PROJ=$PROJ  LOG=$LOG"
log "========================================================"

# ── [0] 验证前置条件 ──────────────────────────────────────────────────────────
BASE_MODEL_PATH=$("$PYTHON" -c "
import sys
from skillreuse.configuration import load_benchmark_config
config = load_benchmark_config(config_path='configs/androidcontrol/default.json')
p = config.get('model', {}).get('source', '')
if not p:
    print('ERROR: model.source missing; check configs/androidcontrol/default.json and .env', file=sys.stderr)
    sys.exit(1)
print(p)
")
CONFIG="$PROJ/configs/androidcontrol/default.json"
MERGED_PATH="$PROJ/outputs/sft_merged"
SWIFT_JSONL="$PROJ/outputs/sft_data/androidcontrol_train.jsonl"
SKILL_LIB="$PROJ/outputs/androidcontrol_discovery/androidcontrol_quotient_skill_library.pkl"

log "Base model: $BASE_MODEL_PATH"

# ── [1] 准备全量 SFT 数据（并行 20 shards，~5-10 分钟）──────────────────────
if [ -f "$SWIFT_JSONL" ]; then
    log "[1/8] SFT 数据已存在 ($(wc -l < "$SWIFT_JSONL") 条)，跳过"
else
    log "=== [1/8] 准备 SFT 数据（全量，并行处理）==="
    "$PYTHON" scripts/data/convert_androidcontrol_to_swift.py \
        --config "$CONFIG" \
        --output-dir "$PROJ/outputs/sft_data" \
        --workers 20 \
        --instruction-modes high_level low_level
    log "SFT 数据完成: $(wc -l < "$SWIFT_JSONL") 条"
fi

# ── [1b] SFT 前确保 GPU 空闲（杀掉所有 vLLM / setsid 残留进程）────────────
SFT_DONE="$PROJ/outputs/sft_output/.sft_done"
if [ ! -f "$SFT_DONE" ]; then
    log "=== [1b/8] 释放 GPU（杀 vLLM 进程）==="
    # 从 pid 文件杀（smoke 或上一次 full 残留）
    for pid_file in "$PROJ/logs/vllm_smoke.pid" "$PROJ/logs/vllm_full.pid"; do
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
        VLLM_PROCS=$(pgrep -f "vllm.entrypoints" 2>/dev/null | wc -l)
        if [ "$VLLM_PROCS" -eq 0 ]; then
            log "  GPU 已空闲"
            break
        fi
        [ "$i" -eq 12 ] && log "  WARNING: 60s 后仍有 vLLM 进程，强制继续"
        sleep 5
    done
    sleep 3  # 额外等 CUDA context 释放
fi

# ── [2] Swift SFT（4× H20，2 epochs，使用验证格式）──────────────────────────
if [ -f "$SFT_DONE" ]; then
    log "[2/8] SFT 已完成，跳过"
else
    log "=== [2/8] Swift SFT (4× H20，2 epochs) ==="
    # Detect GPUs: SLURM or fallback to all visible
    if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    fi
    export NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    "$SWIFT" sft \
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
        2>&1 | tee "$PROJ/logs/sft_full.log"
    touch "$SFT_DONE"
    log "SFT 完成"
fi

# ── [3] 找最新 checkpoint 并合并 LoRA ────────────────────────────────────────
if [ -f "$MERGED_PATH/config.json" ]; then
    log "[3/8] 合并模型已存在，跳过: $MERGED_PATH"
else
    log "=== [3/8] 合并 LoRA ==="
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
    "$SWIFT" export \
        --model "$BASE_MODEL_PATH" \
        --adapters "$LAST_CKPT" \
        --merge_lora true \
        --output_dir "$MERGED_PATH" \
        2>&1 | tee "$PROJ/logs/merge.log"
    log "合并完成: $MERGED_PATH"
fi

# ── [4] 更新 config 指向合并后模型 ──────────────────────────────────────────
log "=== [4/8] 更新 config ==="
"$PYTHON" - <<PYEOF
import json
with open("$CONFIG") as f:
    d = json.load(f)
d["paths"]["base_model_path"] = "$MERGED_PATH"
d["model"]["source"] = "$MERGED_PATH"
with open("$CONFIG", "w") as f:
    json.dump(d, f, indent=2)
print("Config updated -> $MERGED_PATH")
PYEOF

# ── [5] 启动 vLLM（4× H20，SFT 合并模型）────────────────────────────────────
log "=== [5/8] 启动 vLLM ==="
# 停掉可能残留的旧进程
if [ -f "$PROJ/logs/vllm_full.pid" ]; then
    OLD_PID=$(cat "$PROJ/logs/vllm_full.pid" 2>/dev/null || true)
    [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null && \
        { log "停止旧 vLLM PID $OLD_PID"; kill "$OLD_PID" 2>/dev/null || true; sleep 8; } || true
fi

setsid "$PYTHON" scripts/core/start_vllm_replicas.py \
    --model "$MERGED_PATH" \
    --served-model-name Qwen3-VL-8B \
    --gpus "$CUDA_VISIBLE_DEVICES" \
    --front-port 8000 --base-port 8100 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 32 \
>> "$PROJ/logs/vllm_full.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$PROJ/logs/vllm_full.pid"
log "vLLM PID: $VLLM_PID  (log: $PROJ/logs/vllm_full.log)"

log "等待 vLLM proxy 就绪（最多 5 分钟）…"
READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
        READY=1; log "vLLM proxy 就绪"; break
    fi
    sleep 5
done
[ "$READY" -eq 0 ] && { log "ERROR: vLLM 未就绪，查看 $PROJ/logs/vllm_full.log"; exit 1; }

# ── [6] V0 Baseline（SFT 模型，全量 test set）───────────────────────────────
V0_DONE="$PROJ/outputs/v0_eval/.eval_done"
if [ -f "$V0_DONE" ]; then
    log "[6/8] V0 已完成，跳过"
else
    log "=== [6/8] V0 Baseline（全量）==="
    "$PYTHON" scripts/core/run_evaluation.py \
        --benchmark AndroidControl \
        --config "$CONFIG" \
        --variant V0 \
        --output-dir "$PROJ/outputs/v0_eval"
    touch "$V0_DONE"
    log "V0 完成"
fi

# ── [7] Discovery（从零归纳 skill library）───────────────────────────────────
DISC_DONE="$PROJ/outputs/androidcontrol_discovery/.discovery_done"
if [ -f "$DISC_DONE" ] && [ -f "$SKILL_LIB" ]; then
    log "[7/8] Discovery 已完成，跳过"
else
    log "=== [7/8] Discovery（~4-6h，支持 --resume 续跑）==="
    "$PYTHON" scripts/core/run_discovery.py \
        --benchmark AndroidControl \
        --config "$CONFIG" \
        --output-dir "$PROJ/outputs/androidcontrol_discovery" \
        --resume
    touch "$DISC_DONE"
    log "Discovery 完成"
fi

# 打印 discovery 结果
"$PYTHON" - <<PYEOF
import json
from pathlib import Path
p = Path("$PROJ/outputs/androidcontrol_discovery/androidcontrol_discovery_summary.json")
if p.exists():
    s = json.loads(p.read_text())
    print(f"  inducted skills:   {s.get('skill_count', '?')}")
    print(f"  calibrated skills: {s.get('calibrated_skill_count', '?')}")
PYEOF

# ── [8] V1 / V2 / V3 Evaluation ──────────────────────────────────────────────
for VARIANT in V1 V2 V3; do
    VDIR="$PROJ/outputs/${VARIANT,,}_eval"
    VDONE="$VDIR/.eval_done"
    if [ -f "$VDONE" ]; then
        log "[$VARIANT] 已完成，跳过"
        continue
    fi
    log "=== [8/8] $VARIANT Evaluation ==="
    "$PYTHON" scripts/core/run_evaluation.py \
        --benchmark AndroidControl \
        --config "$CONFIG" \
        --skill-library-path "$SKILL_LIB" \
        --variant "$VARIANT" \
        --output-dir "$VDIR"
    touch "$VDONE"
    log "$VARIANT 完成"
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
log "========================================================"
log "Full Pipeline DONE"
log "SFT 模型: $MERGED_PATH"
log "vLLM PID: $(cat "$PROJ/logs/vllm_full.pid" 2>/dev/null || echo '?') (仍在运行)"
log "  停止: kill \$(cat $PROJ/logs/vllm_full.pid)"
log "完整日志: $LOG"
log "========================================================"
