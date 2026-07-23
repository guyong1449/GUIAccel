# GUIAccel 研究计划

> GUI Agent 推理加速：四条研究路线
>
> 项目根目录: `/dkucc/home/rw335/GUIAccel`
> 模型: Qwen3-VL-8B-Instruct (vLLM 8×H20-141G)
> 数据集: AndroidControl (train / validation / test)
> 前序工作: Memoir (SkillReuse) — 已在 AAAI-27 投稿

---

## 0. 背景与动机

GUI Agent 推理延迟的瓶颈在 **decode 阶段**。
现有 GUI Agent (Qwen3-VL, MAI-UI) 输出约 150 token，其中 thinking 占 82%+，action 占 18%。
- 思维链压缩 (CoT compression) 的收益有限：输出本就不长，且思维链删除会降准确率。
- 输入侧裁剪 (VisionZip, DivPrune, AppAgentX) 仅减少 prefill 延迟，对 decode 无帮助。
- Memoir (SkillReuse) 通过 skill 路由 + reduced prompt 大幅减少 token，但属于 **routing + input reduction** 范式。

**本项目聚焦 decode 端的系统级加速**，提出 4 条互补研究路线，最终目标是在 AndroidControl 上实现 **2-3× 端到端加速，准确率无损**。

---

## 1. 坐标输出从"文本生成"改为"回归头"（Coordinate Regression Head）

### 1.1 研究假设

GUI Agent 输出中，bounding box 坐标 `[x1, y1, x2, y2]` 或 `[x, y]` 是连续数值，但被 tokenizer 编码成 4-8 个离散 token，经过 4-8 次自回归 decode step 逐位生成。这是一种信息冗余：

```
标准自回归:  ... "coordinate": [  4  5  3  ,    1  2  8  ] → 8 decode step
回归头:      ... "coordinate":  ← MLP(hidden_state) → [0.453, 0.128]  → 1 forward pass
```

在 AndroidControl 上，CLICK/LONG_PRESS 类动作约占 60%+。每次坐标解码消耗 ~8 decode token (Qwen3-VL 0-999 三位整数×2=6 位 + 分隔 + 括号)，替换为单步回归可节省 **40-50% 的 decode token**。

### 1.2 技术方案

```
Qwen3-VL Forward Pass
    ├── Prefill: 处理 (image + prompt) → KV cache
    └── Decode:
        ├── Step 1..N: 生成 <thinking>..., "action": "click"
        ├── ★ 在 action_type token 位置取 hidden state h_t
        │     ↓
        │   Regression MLP:  h_t ∈ R^3584 → Linear(256) → ReLU → Linear(2) → σ
        │     ↓
        │   [x̂, ŷ] ∈ [0, 1]² → 反归一化到 [0, 999]
        │     ↓
        │   拼回 JSON: {"action": "click", "coordinate": [x̂_999, ŷ_999]}
        └── 跳过坐标 token 的自回归生成
```

**三阶段流水线**:

| 阶段 | 输入 | 输出 | 工具 |
|------|------|------|------|
| **extract** | AndroidControl train + Qwen3-VL | (hidden_state, ground_truth_coord) pairs .pt | `experiments/regression_head.py --mode extract` |
| **train** | .pt 文件 | regression_head.pth | `experiments/regression_head.py --mode train` |
| **eval** | AndroidControl test + regression_head.pth | 延迟对比 + 坐标精度 | `experiments/regression_head.py --mode eval` |

### 1.3 关键设计选择

| 决策 | 选项 | 初始选择 | 理由 |
|------|------|---------|------|
| 何时触发回归头 | action_type 输出后 / 整个 action JSON 结构确认后 | action_type 输出后 | 最大化跳过 token 数 |
| 坐标表示 | 绝对像素 / 0-999 归一化 / [0,1] 归一化 | [0,1] 归一化 | 与分辨率解耦 |
| MLP 架构 | 单层线性 / 2层MLP / 3层MLP | 2层 MLP (3584→256→2) | 平衡容量与速度 |
| 训练数据 | 仅 CLICK / 所有含坐标的动作 | 仅 CLICK + LONG_PRESS | 覆盖最高频坐标动作 |
| 损失函数 | MSE / Smooth L1 / Wing Loss | Smooth L1 | 对离群点鲁棒 |
| 与 thinking 的关系 | 保留 thinking + 回归坐标 / 同时压缩 thinking | 保留 thinking | 解耦实验变量 |

### 1.4 评估指标

| 指标 | 定义 | 成功标准 |
|------|------|---------|
| Coordinate MAE | \|pred - gt\| 在 0-999 尺度上的均值 | < 20 (对应 ~2% 屏幕宽度) |
| Action Match Rate | AndroidControl 标准匹配率 (bbox 中心在 GT 元素内) | ≥ baseline |
| Token Saving | CLICK/LONG_PRESS 步骤省下的 decode token 数 | ≥ 6 token/step |
| Latency Reduction | 端到端单步延迟减少百分比 | ≥ 15% (CLICK 步骤) |
| Overall Accuracy | 全部动作类型加权准确率 | ≥ baseline - 0.5% |

### 1.5 风险与备选

- **风险**: hidden state 在 action_type 位置可能还不包含足够的坐标信息（因为模型"打算"在后续 token 中才确定坐标）。
  - **备选**: 用 thinking 最后一个 token 的 hidden state，或将 action_type + 前 N 个 thinking token 拼接后输入回归头。
- **风险**: 回归头在 OOD 屏幕上的泛化性差。
  - **备选**: 增加 coordinate normalization 的 screen-specific 校准；或在训练时加入 app-specific augmentation。

---

## 2. 跨步 KV Cache 增量更新（Cross-Step KV Cache Reuse）

### 2.1 研究假设

GUI Agent 执行 multi-step 任务时，相邻两步的截图通常只有局部变化（如按钮状态变更、滚动位移）。当前每步从零 prefill，浪费大量计算。若能复用前一步的 KV cache 并仅增量更新变化区域的 visual token，可显著减少 prefill 延迟。

### 2.2 技术方案

```
Step t:   Full prefill → KV_cache_t   → decode → action_t
Step t+1: Δ = diff(screenshot_t, screenshot_t+1)
          仅重新编码 Δ 对应的 visual patch token
          KV_cache_{t+1} = merge(KV_cache_t, Δ_kv)  → decode → action_{t+1}
```

- **Screenshot diff**: 将两帧图像按 28×28 patch 对齐，计算 patch-wise SSIM 或 L2 distance，标记变化 patch。
- **Selective re-encode**: 仅对变化的 patch 做 ViT forward，其余 patch 的 KV 直接复用。
- **Position embedding 对齐**: 确保复用的 KV 与新的位置编码 (RoPE) 兼容。

### 2.3 评估指标

| 指标 | 定义 | 成功标准 |
|------|------|---------|
| Prefill Speedup | 多步任务的总 prefill 时间减少 | ≥ 30% |
| Cache Hit Rate | 可复用 patch 占总 patch 数的比例 | ≥ 50% (典型 GUI 场景) |
| Accuracy Delta | 相比逐步完整 prefill 的准确率变化 | ≤ -0.5% |
| Memory Overhead | 需要额外保存的 KV cache 大小 | ≤ 2× 原始 per-step |

### 2.4 依赖与优先级

- 需要修改 vLLM 的 KV cache 管理或使用 local transformers 路径。
- 优先级: **中** — 属于 prefill 侧优化，与 decode 侧加速正交互补。
- 建议在回归头实验跑通后再展开。

---

## 3. Action-Type 条件化早退（Action-Type Conditioned Early Exit）

### 3.1 研究假设

不同 GUI 动作类型所需的 decode 计算量差异巨大：

| 动作类型 | 频率 (AndroidControl) | 所需 decode token | 可优化策略 |
|---------|----------------------|-------------------|-----------|
| CLICK | ~52% | ~15 (thinking + coord) | 回归头替代坐标 |
| TYPE | ~18% | ~20 (thinking + text) | 规则抽取 text |
| SCROLL | ~12% | ~8 (thinking + direction) | 仅 1 token (direction) |
| NAV | ~8% | ~10 (thinking + arg) | 仅 1-2 token |
| WAIT | ~5% | ~5 (thinking) | 0 decode token |
| TERMINATE | ~3% | ~5 (thinking) | 0 decode token |
| LONG_PRESS | ~2% | ~15 (thinking + coord) | 回归头替代坐标 |

若能在 prefill 阶段就预测出 action_type，就可以为不同类型分配不同的 decode 策略：
- WAIT/TERMINATE: 直接输出，跳过 decode。
- SCROLL/NAV: 生成 1-2 个关键 token 后截断。
- TYPE: 从 instruction 中规则抽取文本，跳过 LM 生成。
- CLICK/LONG_PRESS: 回归头 (与研究方向 1 组合)。

### 3.2 技术方案

```
Prefill output → h_last (最后一个 token 的 hidden state)
    ↓
Linear Probe: h_last ∈ R^3584 → Softmax(7)
    ↓
predicted_action_type
    ↓
    ├── WAIT/TERMINATE → 直接返回 {"action": "wait"}
    ├── SCROLL → 1 token decode: direction ∈ {up, down, left, right}
    ├── NAV → 2 token decode: arg ∈ {back, home, enter}
    ├── TYPE → rule_extract(instruction, a11y_tree) → text
    └── CLICK/LONG_PRESS → 回归头 (方向 1) 或 standard decode
```

### 3.3 评估指标

| 指标 | 定义 | 成功标准 |
|------|------|---------|
| Probe Accuracy | 7-class action type 分类准确率 | ≥ 90% |
| Token Saving | 加权平均每步节省 token 数 | ≥ 30% of baseline |
| End-to-End Latency | 完整评估的平均每步延迟 | ≤ 70% of baseline |
| Accuracy Delta | 相比无早退的全量准确率变化 | ≤ -1% |

### 3.4 依赖与优先级

- 需要回归头 (方向 1) 先跑通，才能处理 CLICK/LONG_PRESS 分支。
- TYPE 的规则抽取可复用 Memoir 的 slot_policy 逻辑。
- 优先级: **高** — 与回归头互补组合潜力最大。

---

## 4. GUI-Specific Speculative Decoding (Draft = Memoir 路由器)

### 4.1 研究假设

标准 speculative decoding 用通用小模型 (如 Qwen3-VL-2B) 做 draft。
但 GUI Agent 有特殊性：Memoir 的 skill 路由器可以在 **零 LM 调用** 的情况下为 ~80% 的简单步骤预测完整 action JSON。

**想法**: 把 Memoir 路由器作为 draft model，替代通用小模型：
- Memoir 预测的 action JSON 作为 draft token 序列。
- Qwen3-VL 做 verification (1 forward pass 验证全部 draft token)。
- 接受则直接输出，拒绝则回退标准 decode。

与标准 speculative decoding 的区别：

| 维度 | 标准 Speculative Decoding | Memoir-as-Draft |
|------|--------------------------|-----------------|
| Draft 来源 | 小 LLM (2B params) | 规则路由 (0 param, ~0 ms) |
| Draft 速度 | 比 target 快 5× | 比 target 快 100×+ |
| Draft 质量 | 通用，acceptance ~60-70% | GUI-specific，Memoir 命中时 >90% |
| 覆盖率 | 100% (所有 token) | ~80% 步骤 (Memoir 覆盖率) |
| 实现难度 | vLLM 内置支持 | 需要自定义 draft + verify 管线 |

### 4.2 技术方案

```
Step input → Memoir 路由器
    ├── 路由器接受 (score > threshold):
    │     draft = skill_reduced_action → tokenize → [t1, t2, ..., tk]
    │     verification = Qwen3-VL.forward([t1..tk], prefix=prompt)
    │     if verify_pass: return draft → 1 forward pass (不是 k decode steps)
    │     else: standard decode
    └── 路由器拒绝:
          standard decode
```

### 4.3 评估指标

| 指标 | 定义 | 成功标准 |
|------|------|---------|
| Acceptance Rate | Memoir draft 被 verification 接受的比例 | ≥ 75% (on Memoir-covered steps) |
| Token Throughput | tokens/second (含 overhead) | ≥ 1.5× baseline |
| Accuracy | 最终输出与 standard decode 的匹配率 | ≥ 99% |
| Latency | 端到端平均延迟 (包含路由 + verify overhead) | ≤ 60% baseline |

### 4.4 依赖与优先级

- 需要 Memoir 的 CalibratedSkill 库 (从 SkillReuse 导入)。
- 需要在 vLLM 中实现自定义 draft token 注入 + verification。
- 优先级: **低** — 实现复杂度最高，但潜在加速最大。

---

## 5. 实验时间线

```
Phase 0 (Week 1):   Baseline evaluation 跑通
                     ├── vLLM 启动、AndroidControl test 评估
                     └── 记录 baseline latency / accuracy / token breakdown

Phase 1 (Week 2-3): 坐标回归头 ← 主攻方向
                     ├── extract: 收集 (hidden_state, gt_coord) pairs
                     ├── train: MLP 训练 + 消融实验
                     └── eval: 延迟与准确率对比

Phase 2 (Week 3-4): Action-Type 早退
                     ├── 训练 linear probe (prefill → action_type)
                     ├── 与回归头组合: CLICK → regression, SCROLL → direction only
                     └── 端到端评估组合策略

Phase 3 (Week 5):   KV Cache 增量更新
                     ├── Screenshot diff 分析
                     ├── 可行性验证 (local transformers)
                     └── 多步任务 latency profiling

Phase 4 (Week 6+):  Speculative Decoding + Memoir
                     ├── Memoir draft → verify pipeline
                     ├── vLLM 集成
                     └── 与 Phase 1-3 组合

论文初稿:            Week 7-8
```

---

## 6. 与 Memoir (SkillReuse) 的关系

| 维度 | Memoir | GUIAccel |
|------|--------|----------|
| **核心思路** | routing + input reduction | decode 侧系统级加速 |
| **减少了什么** | input token (60×压缩) + output token (routing 跳过) | output token (回归头) + decode step (早退) |
| **修改模型** | 否 (纯 routing) | 是 (回归头 = 新 MLP head) |
| **可组合** | 是 — Memoir 路由成功的步骤可以跳过所有 decode | 是 — Memoir 失败 fallback 的步骤使用 GUIAccel 加速 decode |
| **正交性** | Memoir 优化的是 "哪些步骤不需要完整推理" | GUIAccel 优化的是 "需要完整推理的步骤如何更快" |

**潜在组合**: Memoir + GUIAccel = 双层加速
- 第一层: Memoir routing → 80% 步骤走 skill 快路 (极低延迟)
- 第二层: 剩余 20% 步骤 → GUIAccel (回归头 + 早退 → 减少 decode 延迟)
- 整体加速 = 0.8 × Memoir 加速 + 0.2 × GUIAccel 加速

---

## 7. 文件索引

| 文件 | 用途 |
|------|------|
| `experiments/baseline_eval.py` | 基线评估 |
| `experiments/regression_head.py` | 坐标回归头实验 |
| `experiments/action_type_early_exit.py` | Action-Type 早退实验 |
| `experiments/speculative_decode.py` | 投机解码实验 |
| `experiments/constrained_decode.py` | 语法约束解码实验 |
| `guiaccel/model/service_backend.py` | vLLM 推理后端 |
| `guiaccel/model/qwen_backend.py` | 本地 transformers 后端 |
| `guiaccel/data/android_control.py` | AndroidControl 数据加载 |
| `guiaccel/evaluation/android_eval.py` | 评估评分逻辑 |
| `scripts/core/start_vllm_replicas.py` | vLLM 服务启动脚本 |
