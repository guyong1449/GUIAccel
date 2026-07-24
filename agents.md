# agents.md — GUIAccel: Coordinate Regression Head

Technical reference for the GUIAccel coordinate regression head on Qwen3-VL-8B-Instruct.
主攻目标: 将 GUI Agent 的坐标输出从自回归 token 生成替换为单步 MLP 回归。

---

## Model & Serving

| Item | Detail |
|------|--------|
| Backbone | Qwen3-VL-8B-Instruct |
| Architecture | `Qwen3VLForConditionalGeneration` (HuggingFace transformers) |
| Hidden dim | 4096 (`text_config.hidden_size`, 36 layers, 32 heads) |
| vLLM support | Native (OpenAI-compat API, `/v1/chat/completions`) |
| Serving | 8 replicas, 1 GPU each (H20-141G), proxy on `:8000` |
| Coordinate scale | 0-999 归一化 (Qwen3-VL native mobile_use format) |
| Dataset | AndroidControl (train/validation/test) |
| Conda env | `skillreuse-fa2` (`/dkucc/home/rw335/.conda/envs/skillreuse-fa2`, Python 3.10, torch 2.7.0+cu126, flash_attn 2.8.3) |

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
        │     └── Extract h_t ∈ R^4096  (action_type token 的 hidden state)
        │           │
        │           ▼
        │     ┌─────────────────────────────┐
        │     │  CoordHead (trainable)      │
        │     │                             │
        │     │  h_t ─→ Linear(4096, 256)   │
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
Linear(4096, 256):  4096 × 256 + 256 = 917,760 参数
Linear(256, 2):     256 × 2 + 2      = 514 参数
总计:               918,274 参数 (~3.5 MB fp32, ~1.8 MB fp16)
```

相比 Qwen3-VL-8B 的 8.29B 参数，回归头仅增加 **0.011%** 的参数量。

---

## 三阶段流水线

### Phase 1: Hidden State 提取 — GT-Forcing Prefill (`extract`)

```
experiments/regression_head.py --mode extract \
    --output-dir outputs/regression_head/extracted \
    --split train \
    --num-gpus 4
```

**方法名称: GT-Forcing Prefill**

利用 Transformer causal attention mask 的性质：位置 t 的 hidden state
仅依赖 x_0..x_t，与 token 是自回归生成还是一次性输入无关。
因此可将 GT output（到 action_type token 为止）直接拼入输入，
仅做 **单次 forward pass** 而非逐 token generate。

```
AndroidControl train split
    │
    ▼
对每个 episode 的每个 step:
    if step.action_type ∈ {CLICK, LONG_PRESS}:
        │
        ├── 构造 GT 前缀文本:
        │   gt_prefix = '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click"'
        │   ↑ 到 action_type 关键字为止，不包含坐标
        │
        ├── 拼接输入:
        │   input_ids = tokenize([system_prompt + image + user_prompt + gt_prefix])
        │   ↑ GT 前缀被当作"已生成"的内容直接作为 prefill 输入
        │
        ├── 单次 forward pass (非 generate):
        │   outputs = model.forward(input_ids)
        │   h_t = outputs.last_hidden_state[0, -1, :]  # 最后位置
        │   h_t ∈ R^4096
        │   ↑ 无需 output_hidden_states=True
        │     无需 model.generate()
        │     无需保存中间层 hidden states
        │
        └── 记录 ground truth 坐标
            gt_coord = [x/999, y/999]  (归一化到 [0,1])
            │
            ▼
    保存: {
        "hidden_state": h_t,          # torch.Tensor [4096]
        "gt_coord": gt_coord,         # torch.Tensor [2]
        "episode_id": str,
        "step_index": int,
        "screenshot_size": (W, H),
    }
```

**GT-Forcing Prefill vs 自回归 generate 对比**:

| 指标 | 自回归 generate (旧) | GT-Forcing Prefill (新) |
|------|---------------------|------------------------|
| 计算 | ~200 次 decode step | 1 次 forward pass |
| 内存 | KV cache + 全步 hidden states | 单次 forward，无额外存储 |
| 时间/样本 | ~5-10s | ~0.5-1s |
| 可 batch | 否 | **是** |
| 数学等价 | 基准 | 等价 (causal mask) |
| 需 GT 文本 | 否 (模型自生成) | 是 (构造到 action_type 为止) |

**关键实现细节**:

1. **GT 前缀构造**: 使用 AndroidControl 的 GT action_type 构造 JSON 前缀到 `"click"` 关键字，不包含坐标值。

2. **Attention**: 使用 Flash Attention 2 (`attn_implementation="flash_attention_2"`)，ViT + LLM 均走 FA2。

3. **Hidden state 层选择**: 默认使用最后一层 (`last_hidden_state`)。消融实验可尝试倒数第 2-4 层。

4. **多 GPU 并行**: 4 GPU 各加载一份模型，按 episode round-robin 分配，最后合并。

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
    def __init__(self, input_dim=4096, hidden_dim=256):
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
- [x] **Phase 1: GT-Forcing Prefill Hidden state 提取** ← 已完成 2025-07-23
  - GT prefix 构造 (action_type 关键字前缀, 18-19 tokens)
  - 单次 `model.forward(output_hidden_states=True)` 替代 `model.generate()`
  - `mm_token_type_ids` 扩展 (M-RoPE 兼容)
  - 4-GPU 多进程并行提取 (`torch.multiprocessing`)
  - Smoke test 通过: 530ms/sample, hidden_dim=4096, 无 OOM
- [x] `CoordRegressionHead` 模型定义 (`guiaccel/model/coord_head.py`)

### 已完成（相对旧“进行中/待实现”条目；以 outputs 为准）

- [x] **Phase 1 全量提取** — job **31572** COMPLETED; `outputs/regression_head/20260723_050425/` N=46581×4096（merged `metadata=[]` bug known）
- [x] **Phase 2 MLP 训练** — `train_20260723_121025`, best val MAE@999 ≈ **42.61** @ epoch 46
- [x] Decode-eval 管线 + smoke（58）; full-test job **31620** CANCELLED（勿 resume；~800 partial 仅诊断）
- [x] **T1 / E1-A extract API** — `build_gt_prefix` + `--thinking-mode template` (default) + `--extract-point action`; multi-GPU merge `meta`→`metadata` fixed; smoke job **31623** → `outputs/regression_head/e1a_smoke_20260723_201137/`
- [x] **T2 E1-A full re-extract** — job **31624** COMPLETED; `outputs/regression_head/20260723_201700/` N=46581×4096, `thinking_mode=template`, `extract_point=action`, metadata_len=N
- [ ] **Thinking-aware campaign E1–E4** — plan: `docs/plan_e1_e4_thinking_aware.md`；T3 retrain done (see above); fair decode eval not started
- [x] **T3 E1-A CoordRegressionHead retrain** — job **31626** COMPLETED (00:01:06); `outputs/regression_head/train_20260723_223842/`; EXTRACT_SRC=`20260723_201700/extracted`; best val MAE@999 ≈ **44.31** @ epoch 44 (vs Phase-2 baseline ≈42.61 → **worse**); ckpt `trained/coord_head_best.pth`; decode eval not started (parent review→eval)
- [ ] Fair full-test decode eval（新 job，非 31620）

### 待实现 / 后续

- [ ] T4: fair \(\rho\) decode eval on T3 ckpt (`train_20260723_223842/trained/coord_head_best.pth`; val MAE@999≈44.31 worse than baseline 42.61) — not submitted yet
- [ ] E2–E4 + optional E3 parallel after T1 API
- [ ] **E5 SparkUI-style visual⊕\(h_t\) mix** — **post-gate fallback only**（E1–E4 失败门之后；禁止提前实现）
- [ ] 可选: 与 Action-Type 早退组合
- [ ] 可选: vLLM 集成（非本 campaign）

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

### 已创建文件

| 路径 | 描述 |
|------|------|
| `guiaccel/model/coord_head.py` | CoordRegressionHead MLP 模型定义 |
| `guiaccel/model/hidden_state_extractor.py` | GT-Forcing Prefill hidden state 提取 |
| `run_scripts/regression_head/extract_4gpu.sh` | 4-GPU SLURM 提取脚本 |

### 待创建文件

| 路径 | 描述 |
|------|------|
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

---

## 工程规范 (Engineering Standards)

### 推理加速复现规范

- 复现推理加速，涉及推理框架、token 压缩等算法时，重点关注这三个指标：
  1. **时间复杂度**
  2. **新算法和原有算法的 latency 比值**
  3. **token 输入输出的量**

- 除非实在无法避免，**禁止**在适配 backbone 的时候，采用额外造成耗时上升的方法，也就是**禁止额外增加**这三个指标的数值。

- 一定要在完全贴近论文实现方法的基础上，针对新的 backbone 适配，详细分析每个算法步骤的原理，在新情景下带来的：
  - **时间复杂度**
  - **新算法和原有算法的 latency 比值**
  - **token 输入输出的量**

- 引入严谨数学计算证明
- 画出过程中的时序图、模块图，说明插入算法影响到的架构区域，最后整理成报告返回

### 代码质量

- 在不降低数据质量的情况下进行代码处理，**禁止引入降级、简化的实现方法，或者绕过部分实施方法**
- **禁止引入过度防御性代码**，有问题的输入应该报错，而不是通过兜底措施假装能正常运行
- 对于某个方向的多个新增文件和测试，**新建文件夹然后写入**，防止工作区杂乱

### Heart Monitor

- 针对长任务的 subagent，**2min 一次进行任务查询和 log 查看**，确保任务正常进行，程序正常输出

### 模型训练 / 推理 / Eval 操作

- 首先考虑**脚本内实现多核并行**
- 所有 `.sh` 脚本默认 `--time 7-00:00:00`，目的：保证测试和正式任务不因为时长限制导致问题
- 非 smoke 任务的脚本，提交 A40 **4 卡**
- 针对长任务的脚本提交后，启动一个 Heart Monitor，**2min 一次进行任务查询和 log 查看**，确保任务正常进行，程序正常输出
- 提交脚本前要保证 **dry run 能跑通，参数无误**，然后进行 smoke
- 针对数据处理、读取方面的脚本，请慎重思考多核并行，保存 checkpoint 时候对应的内存大小，合理分配参数，一定要使用 `common*` CPU 节点提交任务，严禁使用 GPU 节点
- 训练、评测、推理时确保支持 **resume 参数**，保存 checkpoint 能够续训，编写拉起续训的脚本，默认拉起次数上限为 40000
- 提交任务，产物输出放在新文件夹，相同类型的任务，禁止覆盖
- 转移文件使用 `mv`，禁止使用复制后删除
