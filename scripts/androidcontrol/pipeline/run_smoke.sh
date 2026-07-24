#!/bin/bash
# =============================================================================
# run_smoke.sh — 冒烟测试流水线（SFT → V0 → Discovery → V1）
#
# 目的：端到端验证整个流水线可以跑通，每步用极小数据量。
# 数据规模：每 shard 5 个 episode (~100 条 SFT 样本)，eval/discovery 限 20 episodes
# 预期耗时：全程约 1-2 小时
#
# 启动方式（Jupyter cell）：
#   PREFIX=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangjunpeng06
#   nohup bash $PREFIX/qwenreuse/scripts/androidcontrol/pipeline/run_smoke.sh &
#   echo "Smoke PID: $!"
#
# 监控：
#   tail -f $PREFIX/qwenreuse/logs/smoke_pipeline.log
# =============================================================================
set -euo pipefail

PREFIX=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangjunpeng06
PROJ=$PREFIX/qwenreuse
PYTHON=$PREFIX/miniconda3/envs/maiui-vllm/bin/python
SWIFT=$PREFIX/miniconda3/envs/maiui-vllm/bin/swift
LOG=$PROJ/logs/smoke_pipeline.log

# ── 第一步：用绝对路径建目录，然后自重定向（nohup 不需要指定输出文件）────────
mkdir -p "$PROJ/logs" "$PROJ/outputs"
exec >> "$LOG" 2>&1   # 此后所有 stdout/stderr 都写入 LOG

cd "$PROJ"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "========================================================"
log "QwenReuse Smoke Pipeline START"
log "PROJ=$PROJ  LOG=$LOG"
log "========================================================"

# ── [0] 验证模型路径 ─────────────────────────────────────────────────────────
BASE_MODEL_PATH=$("$PYTHON" -c "
import json, sys
d = json.load(open('configs/androidcontrol/default.json'))
p = d.get('model', {}).get('source', '')
if not p or p == '__SET_BY_SETUP_SH__':
    print('ERROR: run bash scripts/setup/setup.sh first', file=sys.stderr); sys.exit(1)
print(p)
")
log "Base model: $BASE_MODEL_PATH"

MERGED_PATH="$PROJ/outputs/sft_merged_smoke"
SMOKE_CONFIG="$PROJ/configs/androidcontrol/pipeline_smoke.json"

# ── [1] Smoke SFT 数据准备（每 shard 5 个 episode）───────────────────────────
SMOKE_JSONL="$PROJ/outputs/sft_data_smoke/androidcontrol_train.jsonl"

if [ -f "$SMOKE_JSONL" ]; then
    log "[1/7] SFT 数据已存在 ($(wc -l < "$SMOKE_JSONL") 条)，跳过"
else
    log "=== [1/7] 准备 Smoke SFT 数据 ==="
    "$PYTHON" scripts/data/convert_androidcontrol_to_swift.py \
        --config configs/androidcontrol/default.json \
        --output-dir "$PROJ/outputs/sft_data_smoke" \
        --workers 20 \
        --instruction-modes high_level low_level \
        --limit 5
    log "SFT 数据完成: $(wc -l < "$SMOKE_JSONL") 条"
fi

# ── [2] Smoke Swift SFT（max_steps=50，在最后一步保存）───────────────────────
SFT_DONE="$PROJ/outputs/sft_output_smoke/.sft_done"

if [ -f "$SFT_DONE" ]; then
    log "[2/7] SFT 已完成，跳过"
else
    log "=== [2/7] Swift SFT (smoke: max_steps=50，使用验证格式) ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    "$SWIFT" sft \
        --model "$BASE_MODEL_PATH" \
        --dataset "$SMOKE_JSONL" \
        --output_dir "$PROJ/outputs/sft_output_smoke" \
        --num_train_epochs 1 \
        --max_steps 50 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --lora_rank 64 \
        --lora_alpha 16 \
        --bf16 true \
        --gradient_checkpointing true \
        --deepspeed zero2 \
        --dataloader_num_workers 4 \
        --save_strategy steps \
        --save_steps 50 \
        --logging_steps 5 \
        2>&1 | tee "$PROJ/logs/sft_smoke.log"
    touch "$SFT_DONE"
    log "SFT 完成"
fi

# ── [3] 找 checkpoint 并合并 LoRA ────────────────────────────────────────────
if [ -f "$MERGED_PATH/config.json" ]; then
    log "[3/7] 合并模型已存在，跳过: $MERGED_PATH"
else
    log "=== [3/7] 合并 LoRA ==="
    LAST_CKPT=$("$PYTHON" -c "
from pathlib import Path
import sys
ckpts = list(Path('$PROJ/outputs/sft_output_smoke').rglob('adapter_config.json'))
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
        --output_dir "$MERGED_PATH"
    log "合并完成: $MERGED_PATH"
fi

# ── [4] 生成 Smoke 配置（含 episode 限制和合并模型路径）─────────────────────
log "=== [4/7] 生成 Smoke 配置 ==="
"$PYTHON" - <<PYEOF
import json
with open("$PROJ/configs/androidcontrol/default.json") as f:
    d = json.load(f)
d["paths"]["base_model_path"] = "$MERGED_PATH"
d["model"]["source"] = "$MERGED_PATH"
d.setdefault("evaluation", {})["episode_limit"] = 20
d.setdefault("discovery", {}).update({
    "repository_episode_limit": 20,
    "calibration_episode_limit": 10,
})
with open("$SMOKE_CONFIG", "w") as f:
    json.dump(d, f, indent=2)
print("Smoke config: $SMOKE_CONFIG")
PYEOF

# ── [5] 启动 vLLM ────────────────────────────────────────────────────────────
log "=== [5/7] 启动 vLLM ==="
# 停掉可能残留的旧 vLLM
if [ -f "$PROJ/logs/vllm_smoke.pid" ]; then
    OLD_PID=$(cat "$PROJ/logs/vllm_smoke.pid" 2>/dev/null || true)
    [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null && \
        { log "停止旧 vLLM PID $OLD_PID"; kill "$OLD_PID" 2>/dev/null || true; sleep 5; } || true
fi

setsid "$PYTHON" scripts/core/start_vllm_replicas.py \
    --model "$MERGED_PATH" \
    --served-model-name Qwen3-VL-8B \
    --gpus 0,1,2,3 \
    --front-port 8000 --base-port 8100 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 32 \
>> "$PROJ/logs/vllm_smoke.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$PROJ/logs/vllm_smoke.pid"
log "vLLM PID: $VLLM_PID  (log: $PROJ/logs/vllm_smoke.log)"

log "等待 vLLM proxy 就绪（最多 5 分钟）…"
READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
        READY=1; log "vLLM proxy 就绪"; break
    fi
    sleep 5
done
[ "$READY" -eq 0 ] && { log "ERROR: vLLM 未就绪，查看 $PROJ/logs/vllm_smoke.log"; exit 1; }

# ── [6] Smoke V0 Baseline ────────────────────────────────────────────────────
V0_DONE="$PROJ/outputs/v0_eval_smoke/.eval_done"
if [ -f "$V0_DONE" ]; then
    log "[6a/7] V0 已完成，跳过"
else
    log "=== [6a/7] V0 Baseline (smoke 20 episodes) ==="
    "$PYTHON" scripts/core/run_evaluation.py \
        --benchmark AndroidControl \
        --config "$SMOKE_CONFIG" \
        --variant V0 \
        --output-dir "$PROJ/outputs/v0_eval_smoke"
    touch "$V0_DONE"
    log "V0 完成"
fi

# ── [6b] Smoke Discovery ─────────────────────────────────────────────────────
DISC_DONE="$PROJ/outputs/androidcontrol_discovery_smoke/.discovery_done"
if [ -f "$DISC_DONE" ]; then
    log "[6b/7] Discovery 已完成，跳过"
else
    log "=== [6b/7] Discovery (smoke: repo=20, calib=10 episodes) ==="
    "$PYTHON" scripts/core/run_discovery.py \
        --benchmark AndroidControl \
        --config "$SMOKE_CONFIG" \
        --output-dir "$PROJ/outputs/androidcontrol_discovery_smoke"
    touch "$DISC_DONE"
    log "Discovery 完成"
fi

# ── [7] Smoke V1 ─────────────────────────────────────────────────────────────
SKILL_LIB="$PROJ/outputs/androidcontrol_discovery_smoke/androidcontrol_quotient_skill_library.pkl"
V1_DONE="$PROJ/outputs/v1_eval_smoke/.eval_done"
if [ -f "$V1_DONE" ]; then
    log "[7/7] V1 已完成，跳过"
elif [ ! -f "$SKILL_LIB" ]; then
    log "[7/7] WARNING: skill library 未找到（calibrated skills 可能为 0），跳过 V1"
else
    log "=== [7/7] V1 Evaluation (smoke) ==="
    "$PYTHON" scripts/core/run_evaluation.py \
        --benchmark AndroidControl \
        --config "$SMOKE_CONFIG" \
        --skill-library-path "$SKILL_LIB" \
        --variant V1 \
        --output-dir "$PROJ/outputs/v1_eval_smoke"
    touch "$V1_DONE"
    log "V1 完成"
fi

# ── 汇总 ─────────────────────────────────────────────────────────────────────
log "========================================================"
log "Smoke Pipeline DONE"
log "vLLM 仍在运行 PID=$(cat "$PROJ/logs/vllm_smoke.pid" 2>/dev/null || echo '?')"
log "  停止: kill \$(cat $PROJ/logs/vllm_smoke.pid)"
log "结果: $PROJ/outputs/v0_eval_smoke/ | discovery_smoke/ | v1_eval_smoke/"
log "完整日志: $LOG"
log "========================================================"
