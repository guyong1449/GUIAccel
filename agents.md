# agents.md — GUIAccel: Coordinate Regression Head

Agent workflow reference for GUIAccel. 方法细节见 `docs/`：
- [GT-Forcing Prefill](./docs/gt_forcing_prefill.md)
- [Coordinate Regression Head 方法与架构](./docs/coord_regression_head.md)

主攻目标: 将 GUI Agent 的坐标输出从自回归 token 生成替换为单步 MLP 回归。

---

## 默认配置 (Defaults)

### Flash Attention 2 (FA2)

- **VLM / LLM 训练和推理默认使用 `flash_attention_2`**，禁止在非诊断场景下回退到 SDPA 或 eager attention。
- Qwen2-VL / Qwen3-VL 配置方式：

```python
# HuggingFace from_pretrained
model = AutoModel.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",  # ViT + LLM 同时启用
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

```json
// JSON config（MSD / SkillReuse 后端）
{
  "model": {
    "attn_implementation": "flash_attention_2"
  }
}
```

- 相关 conda env 需预装 `flash-attn`（`pip install flash-attn --no-build-isolation`），验证：
  ```bash
  python -c "from transformers.utils import is_flash_attn_2_available; print(is_flash_attn_2_available())"
  ```
- 默认 conda envs：`skillreuse-fa2`（Qwen3-VL-8B）、`msd-androidcontrol`（Qwen2-VL-7B MSD）均应包含 FA2。
- **例外**：纯断点调试 / profiling（对比 SDPA vs FA2 latency）时可临时关闭；提交正式任务前必须恢复。

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
- [x] `CoordRegressionHead` 模型定义 (`guiaccel/model/coord_head.py`)

### 已完成（以 outputs 为准）

- [x] **Phase 1 全量提取** — job **31572** COMPLETED; `outputs/regression_head/20260723_050425/` N=46581×4096
- [x] **Phase 2 MLP 训练** — `train_20260723_121025`, best val MAE@999 ≈ **42.61** @ epoch 46
- [x] Decode-eval 管线 + smoke（58）; full-test job **31620** CANCELLED（勿 resume）
- [x] **T1 / E1-A extract API** — `build_gt_prefix` + `--thinking-mode template` (default) + `--extract-point action`; smoke job **31623** → `outputs/regression_head/e1a_smoke_20260723_201137/`
- [x] **T2 E1-A full re-extract** — job **31624** COMPLETED; `outputs/regression_head/20260723_201700/` N=46581×4096
- [ ] **Thinking-aware campaign E1–E4** — plan: `docs/plan_e1_e4_thinking_aware.md`
- [x] **T3 E1-A CoordRegressionHead retrain** — job **31626** COMPLETED; best val MAE@999 ≈ **44.31** @ epoch 44
- [x] **T4 full-test decode eval** — job **31627** COMPLETED; AR bbox hit=**0.911**, Reg=**0.520**; **does not pass** thinking-aware accuracy gate (≥0.80); proceed E2

### 待实现 / 后续

- [x] T4 结果: Reg hit 0.520 vs AR 0.911 — proceed E2
- [ ] E2–E4 + optional E3 parallel after T1 API
- [ ] **E5 SparkUI-style visual⊕\(h_t\) mix** — **post-gate fallback only**（E1–E4 失败门之后；禁止提前实现）
- [ ] 可选: 与 Action-Type 早退组合
- [ ] 可选: vLLM 集成（非本 campaign）

---

## E2: 3-Point CPU Training (thinking_end / action / coord_bracket) — COMPLETED

| Item | Detail |
|------|--------|
| Extract | `outputs/regression_head/20260724_021359` (multi_point=True, N=46581) |
| Job ID | **31657** |
| Script | `run_scripts/regression_head/train_e2_3points_cpu.sh` |
| Status | **COMPLETED** (2026-07-24 19:21 UTC+8, Elapsed=00:05:10) |
| Batch dir | `outputs/regression_head/e2_train3_20260724_071646/` |

### Results per extract point

| Extract Point | Best Epoch | val_MAE@999 | Output Dir |
|--------------|-----------|-------------|-----------|
| **thinking_end** | 49 | **67.19** | `train_e2_thinking_end_20260724_071712/` |
| **action** | 32 | **41.86** | `train_e2_action_20260724_071900/` |
| **coord_bracket** | 45 | **37.16** | `train_e2_coord_bracket_20260724_072020/` |

**Winner by MAE: coord_bracket** (37.16 vs action 41.86 vs thinking_end 67.19)

### E2 Decode Eval — Submitted

| Item | Detail |
|------|--------|
| Eval script | `run_scripts/regression_head/eval_decode_4gpu.sh` |
| Split | test (full, ~5k samples) |
| GPUs | 4×A40 per job |
| **thinking_end** | job **31658** |
| **action** | job **31659** |
| **coord_bracket** | job **31660** |
| Submitted | 2026-07-24 19:23 UTC+8 |

---

## 代码索引

### 核心文件

| 路径 | 描述 |
|------|------|
| `experiments/regression_head.py` | 三阶段实验入口 (extract/train/eval) |
| `guiaccel/model/service_backend.py` | vLLM OpenAI-compat 推理后端 |
| `guiaccel/model/qwen_backend.py` | Local transformers 推理后端 |
| `guiaccel/model/maiui_action_adapter.py` | Qwen3-VL output → CanonicalAction 解析 |
| `guiaccel/model/coord_head.py` | CoordRegressionHead MLP 模型定义 |
| `guiaccel/model/hidden_state_extractor.py` | GT-Forcing Prefill hidden state 提取 |
| `guiaccel/data/android_control.py` | AndroidControl 数据集加载 |
| `guiaccel/evaluation/android_eval.py` | AndroidControl 评分 (bbox 匹配) |
| `guiaccel/types.py` | 核心数据类: CanonicalAction, BBox, etc. |
| `guiaccel/routing/common.py` | TokenUsage, StepContext, estimate_visual_tokens |

### 脚本

| 路径 | 描述 |
|------|------|
| `run_scripts/regression_head/extract_4gpu.sh` | 4-GPU SLURM 提取脚本 |
| `run_scripts/regression_head/train_e2_3points_cpu.sh` | E2 3-point CPU 训练脚本 |
| `run_scripts/regression_head/eval_decode_4gpu.sh` | 4-GPU decode eval 脚本 |

### 待创建

| 路径 | 描述 |
|------|------|
| `guiaccel/model/coord_decode.py` | 集成: 标准 decode → 回归头替代坐标 |

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

### 数据保护

- **禁止私自 `rm` 或删除 `outputs/` 目录下的任何内容**。所有实验产物（extract、train、eval 结果）均存储在 `outputs/` 下，一旦删除不可恢复。如需清理磁盘空间，必须先确认产出已备份或不再需要，并在团队内确认后再操作。

---

## DKUCC pip 环境配置

### pip + HTTP 代理已知限制

由于 pip 内置的 `urllib3` (1.26.x) 在通过 HTTP CONNECT 代理隧道 (`127.0.0.1:17890`) 访问 PyPI 时存在 TLS 握手兼容性问题（`SSLEOFError`），**不推荐**在 DKUCC 上通过代理直连 PyPI。

curl、Python 标准库 `urllib.request`、conda 均不受影响。

### 解决方案：使用国内镜像站

```bash
# 设置 pip 镜像（推荐）
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/

# 单次安装
pip install <package> -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

conda 使用独立的镜像通道配置，不受此问题影响。

### 参考文档

完整文档（根因分析、对比矩阵、conda 镜像配置）：
`dkucc-clash-forwarding` skill → `references/pip-mirror-config.md`
(`~/.cc-switch/skills/dkucc-clash-forwarding/references/pip-mirror-config.md`)
