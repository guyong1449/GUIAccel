# agents.md — GUIAccel: Coordinate Regression Head

Technical reference for the GUIAccel coordinate regression head on Qwen3-VL-8B-Instruct.
主攻目标: 将 GUI Agent 的坐标输出从自回归 token 生成替换为单步 MLP 回归。

---

## Model & Serving

| Item | Detail |
|------|--------|
| Backbone | Qwen3-VL-8B-Instruct |
| Architecture | `Qwen3VLForConditionalGeneration` (HuggingFace transformers) |
| Hidden dim | 3584 (Qwen3-VL-8B 的 `hidden_size`) |
| vLLM support | Native (OpenAI-compat API, `/v1/chat/completions`) |
| Serving | 8 replicas, 1 GPU each (H20-141G), proxy on `:8000` |
| Coordinate scale | 0-999 归一化 (Qwen3-VL native mobile_use format) |
| Dataset | AndroidControl (train/validation/test) |
| Conda env | `maiui-vllm` (复用自 SkillReuse) |

---

## 问题定义

### 坐标文本生成的低效性

Qwen3-VL 在 AndroidControl 上的 CLICK 动作输出格式:

```json
<thinking>
We are on the settings page. We need to tap the "Connected devices" option...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [453, 628]}}
</tool_call>
```

其中 `"coordinate": [453, 628]` 的 token 化过程:

```
Token 序列:   [  4  5  3  ,     6  2  8  ]
Decode 步数:  1   2  3  4  5   6  7  8  9
```

每次 CLICK/LONG_PRESS 需要 **6-9 个 decode step** 仅用于生成两个整数。
在 AndroidControl 上 CLICK + LONG_PRESS 占 ~54% 的步骤。

**目标**: 用一个 MLP 回归头替代坐标 token 的自回归生成，将这 6-9 step 压缩为 1 step (单次矩阵乘法)。

---

## 方法: Coordinate Regression Head

### 核心架构

```
Qwen3-VL Backbone (frozen)
    │
    ├── Prefill: process(image, prompt) → KV cache
    │
    └── Decode:
        ├── Step 1..N: autoregressive generate <thinking>...</thinking>
        │               and "action": "click"  ← action_type token
        │
        ├── ★ Trigger Point: 检测到 action_type = CLICK 或 LONG_PRESS
        │     │
        │     └── Extract h_t ∈ R^3584  (action_type token 的 hidden state)
        │           │
        │           ▼
        │     ┌─────────────────────────────┐
        │     │  CoordHead (trainable)      │
        │     │                             │
        │     │  h_t ─→ Linear(3584, 256)   │
        │     │      ─→ ReLU                │
        │     │      ─→ Dropout(0.1)        │
        │     │      ─→ Linear(256, 2)      │
        │     │      ─→ Sigmoid             │
        │     │  output: [x̂, ŷ] ∈ [0,1]²   │
        │     └─────────────────────────────┘
        │           │
        │           ▼
        │     反归一化: x_999 = round(x̂ × 999), y_999 = round(ŷ × 999)
        │           │
        │           ▼
        │     拼回 JSON: {"action":"click","coordinate":[x_999, y_999]}
        │
        └── 跳过坐标 token 的后续自回归生成 (省 6-9 decode step)
```

### 参数量分析

```
Linear(3584, 256):  3584 × 256 + 256 = 917,760 参数
Linear(256, 2):     256 × 2 + 2      = 514 参数
总计:               918,274 参数 (~3.5 MB fp32, ~1.8 MB fp16)
```

相比 Qwen3-VL-8B 的 8.29B 参数，回归头仅增加 **0.011%** 的参数量。

---

## 三阶段流水线

### Phase 1: Hidden State 提取 (`extract`)

```
experiments/regression_head.py --mode extract \
    --output-dir outputs/regression_head/extracted \
    --split train \
    --task-limit 5000
```

**流程**:

```
AndroidControl train split
    │
    ▼
对每个 episode 的每个 step:
    if step.action_type ∈ {CLICK, LONG_PRESS}:
        │
        ├── 运行 Qwen3-VL forward pass (local transformers, 非 vLLM)
        │   启用 output_hidden_states=True
        │
        ├── 定位 action_type token 的位置 (在 output sequence 中)
        │   方法: 在 generated token sequence 中搜索
        │         "click" 或 "long_press" 对应的 token ID
        │
        ├── 提取该位置的 last_hidden_state
        │   h_t = model_output.hidden_states[-1][:, action_type_pos, :]
        │   h_t ∈ R^3584
        │
        └── 记录 ground truth 坐标
            gt_coord = [x/999, y/999]  (归一化到 [0,1])
            │
            ▼
    保存: {
        "hidden_state": h_t,          # torch.Tensor [3584]
        "gt_coord": gt_coord,         # torch.Tensor [2]
        "episode_id": str,
        "step_index": int,
        "screenshot_size": (W, H),
    }
```

**关键实现细节**:

1. **Action type token 定位**: Qwen3-VL 的 tokenizer 将 `"click"` 编码为特定 token ID。需要:
   - 先 tokenize `"click"` → 获取 target token IDs
   - 在 model output 的 token 序列中搜索匹配位置
   - 取 **最后一次出现** (因为 thinking 中可能也提到 "click")

2. **Hidden state 层选择**: 默认使用最后一层 (`hidden_states[-1]`)。消融实验可尝试倒数第 2-4 层或多层拼接。

3. **内存管理**: Qwen3-VL-8B + `output_hidden_states=True` 在 H20-141G 上单图推理约需 ~30 GB VRAM。需单 GPU 运行，batch_size=1。

**输出文件**: `extracted/train_hidden_states.pt` — 包含 N 个 (h_t, gt_coord) 对。

### Phase 2: MLP 训练 (`train`)

```
experiments/regression_head.py --mode train \
    --output-dir outputs/regression_head/trained \
    --hidden-dim 256 \
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 256
```

**训练配置**:

```python
class CoordRegressionHead(nn.Module):
    def __init__(self, input_dim=3584, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # 输出 [0,1]²
        )

    def forward(self, h):
        return self.mlp(h)
```

| 超参数 | 初始值 | 搜索范围 | 备注 |
|--------|--------|---------|------|
| hidden_dim | 256 | {128, 256, 512} | 过大会过拟合 |
| learning_rate | 1e-3 | {5e-4, 1e-3, 2e-3} | AdamW |
| weight_decay | 1e-4 | {0, 1e-4, 1e-3} | 正则化 |
| dropout | 0.1 | {0, 0.1, 0.2} | 输入 h_t 的分布较稳定 |
| batch_size | 256 | {128, 256, 512} | 纯 MLP 训练，GPU 不是瓶颈 |
| epochs | 50 | {30, 50, 100} | early stopping patience=10 |
| loss | Smooth L1 | {MSE, Smooth L1} | Smooth L1 对离群点鲁棒 |

**损失函数**:

```python
loss = F.smooth_l1_loss(pred_coord, gt_coord, beta=0.01)
# beta=0.01: 在 0-1 归一化尺度上, 0.01 对应 ~10 像素 (999尺度)
# L1 region: |err| > 0.01 → 防止大误差主导梯度
# L2 region: |err| ≤ 0.01 → 小误差区域更精细
```

**数据划分**:
- Train: AndroidControl train split 的 80%
- Val: AndroidControl train split 的 20% (用于 early stopping)
- Test: AndroidControl test split (最终评估)

**输出文件**: `trained/coord_head.pth` — MLP 权重。

### Phase 3: 评估 (`eval`)

```
experiments/regression_head.py --mode eval \
    --output-dir outputs/regression_head/eval \
    --split test
```

**评估流程**:

```
对 test split 的每个 CLICK/LONG_PRESS 步骤:
    │
    ├── [Standard] 正常自回归生成完整 output → 解析坐标
    │   记录: latency_standard, coord_standard, tokens_generated
    │
    └── [Regression] 生成到 action_type token 后截断
        取 hidden state → CoordHead → coord_regression
        记录: latency_regression, coord_regression, tokens_saved
    │
    ▼
比较:
    coord_error   = |coord_standard - coord_regression| (per sample)
    latency_ratio = latency_regression / latency_standard
    match_rate    = AndroidControl 标准匹配评分
```

**输出**:
- `eval/results.json`: 逐步骤的详细结果
- `eval/summary.json`: 汇总指标 (MAE, match_rate, latency_speedup, etc.)
- `eval/error_analysis.json`: 误差 > 阈值的案例分析

---

## 坐标系统

Qwen3-VL-8B-Instruct 使用 **0-999 归一化坐标**:

```
x_999 = round(x_pixel / screenshot_width  × 999)
y_999 = round(y_pixel / screenshot_height × 999)
```

回归头内部使用 **[0, 1] 归一化**，输出时乘以 999:

```
[0, 1] 内部表示 ←→ [0, 999] 模型输出 ←→ 绝对像素 (评估用)
      ×999              × W/999 或 × H/999
```

AndroidControl 评估使用 **绝对像素坐标**:
- ground truth bbox 来自 accessibility tree (绝对像素)
- 预测的 (x_999, y_999) 需要反归一化到绝对像素
- 匹配判据: 预测中心点落在 GT bbox 内

---

## Action Space

AndroidControl 上的完整动作空间及回归头的适用性:

| Action Type | 频率 | 需要坐标 | 回归头适用 | Decode Token 节省 |
|-------------|------|---------|-----------|-----------------|
| CLICK | ~52% | ✓ 必须 | ✓ 主要目标 | ~8 token/step |
| TYPE | ~18% | ✗ | ✗ | N/A |
| SCROLL | ~12% | (可选) | △ 低价值 | ~4 token (仅方向) |
| NAV | ~8% | ✗ | ✗ | N/A |
| WAIT | ~5% | ✗ | ✗ | N/A |
| TERMINATE | ~3% | ✗ | ✗ | N/A |
| LONG_PRESS | ~2% | ✓ 必须 | ✓ 次要目标 | ~8 token/step |

**加权预期节省**: 0.52×8 + 0.02×8 = **4.32 token/step** (仅 CLICK + LONG_PRESS)。
以 baseline ~150 token/step 计: **~2.9% output token 减少** (坐标 token 占比不大)。
但 **延迟节省更显著**: 8 decode step 的时间 >> 1 次 MLP 前向。

---

## 实现状态

### 已就位

- [x] Qwen3-VL-8B-Instruct 模型 + vLLM 服务基础设施
- [x] AndroidControl 数据加载 (`guiaccel/data/android_control.py`)
- [x] 坐标解析适配器 (`guiaccel/model/maiui_action_adapter.py`)
- [x] 评估评分逻辑 (`guiaccel/evaluation/android_eval.py`)
- [x] 实验入口脚本骨架 (`experiments/regression_head.py`)
- [x] vLLM 后端 (`guiaccel/model/service_backend.py`)
- [x] Local transformers 后端 (`guiaccel/model/qwen_backend.py`)

### 待实现

- [ ] **Phase 1: Hidden state 提取**
  - Hook Qwen3-VL forward pass → 获取 `hidden_states`
  - Action type token 定位逻辑
  - (hidden_state, gt_coord) pair 保存/加载
- [ ] **Phase 2: MLP 训练**
  - `CoordRegressionHead` 模型定义
  - 训练循环 (AdamW, Smooth L1, early stopping)
  - 消融实验框架 (hidden_dim, 层选择, 损失函数)
- [ ] **Phase 3: 集成评估**
  - 在 Qwen3-VL decode 管线中插入回归头
  - 端到端延迟对比
  - AndroidControl 标准评估 + 坐标误差分析
- [ ] **可选: 与 Action-Type 早退组合**
  - 在 prefill 阶段预测 action_type
  - CLICK/LONG_PRESS → 回归头, 其他 → 对应策略

---

## 代码索引

### 核心文件

| 路径 | 描述 |
|------|------|
| `experiments/regression_head.py` | 三阶段实验入口 (extract/train/eval) |
| `guiaccel/model/service_backend.py` | vLLM OpenAI-compat 推理后端 |
| `guiaccel/model/qwen_backend.py` | Local transformers 推理后端 |
| `guiaccel/model/maiui_action_adapter.py` | Qwen3-VL output → CanonicalAction 解析 |
| `guiaccel/data/android_control.py` | AndroidControl 数据集加载 |
| `guiaccel/evaluation/android_eval.py` | AndroidControl 评分 (bbox 匹配) |
| `guiaccel/types.py` | 核心数据类: CanonicalAction, BBox, etc. |
| `guiaccel/routing/common.py` | TokenUsage, StepContext, estimate_visual_tokens |

### 待创建文件 (Phase 1 实现时)

| 路径 | 描述 |
|------|------|
| `guiaccel/model/coord_head.py` | CoordRegressionHead 模型定义 |
| `guiaccel/model/hidden_state_extractor.py` | Hidden state 提取 + action_type 定位 |
| `guiaccel/model/coord_decode.py` | 集成: 标准 decode → 回归头替代坐标 |

---

## 消融实验计划

| 实验 | 变量 | 对比 | 预期结论 |
|------|------|------|---------|
| A1 | Hidden state 提取位置 | action_type token vs thinking 最后 token vs 两者拼接 | action_type token 信息最充分 |
| A2 | 使用哪一层 hidden state | last layer vs 倒数第 2/4 层 vs 多层加权 | last layer 足够 |
| A3 | MLP 深度 | 1层线性 vs 2层MLP vs 3层MLP | 2层 MLP 是甜点 |
| A4 | Hidden dim | 128 vs 256 vs 512 | 256 平衡精度与速度 |
| A5 | 损失函数 | MSE vs Smooth L1 vs Wing Loss | Smooth L1 最稳 |
| A6 | 训练数据量 | 1k vs 5k vs 全量 train | 5k 已经饱和 |
| A7 | 坐标表示 | [0,1] vs [0,999] vs 绝对像素 | [0,1] 泛化最好 |
| A8 | 是否加入图像特征 | 仅 h_t vs h_t + global_visual_feature | 仅 h_t 足够 |

---

## 已知风险

1. **Hidden state 信息不足**: action_type token 位置的 hidden state 可能尚未"决定"具体坐标值。transformer 可能计划在后续 decode 步中逐步精化坐标。
   - **缓解**: 消融实验 A1 测试不同提取位置；若 action_type 位置不行，用 thinking 段末尾。

2. **回归头泛化到 OOD app**: 回归头在训练 app 上学到了坐标分布先验，换 app 可能失效。
   - **缓解**: 使用 [0,1] 归一化解耦分辨率；AndroidControl 本身覆盖多 app。

3. **Token saving 绝对值小**: 8 token/step 相比 150 token/step 只有 ~5% 减少。
   - **缓解**: 延迟节省 > token 节省 (decode 是串行的，每省 1 step 就快一次)；与 Action-Type 早退组合可放大效果。

4. **Integration 复杂度**: 在 vLLM serving 中插入回归头需要自定义 decoding 逻辑。
   - **缓解**: Phase 1-2 在 local transformers 路径实现和验证，Phase 3 再考虑 vLLM 集成。

---

## 使用说明

```bash
# 激活环境
source /dkucc/home/rw335/SkillReuse/.conda/maiui-vllm/bin/activate

# Phase 1: 提取 hidden states
cd /dkucc/home/rw335/GUIAccel
python experiments/regression_head.py --mode extract \
    --output-dir outputs/regression_head/extracted \
    --split train --task-limit 5000

# Phase 2: 训练回归头
python experiments/regression_head.py --mode train \
    --output-dir outputs/regression_head/trained \
    --hidden-dim 256

# Phase 3: 评估
python experiments/regression_head.py --mode eval \
    --output-dir outputs/regression_head/eval \
    --split test
```
