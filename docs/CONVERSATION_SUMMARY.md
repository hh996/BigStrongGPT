# 项目讨论记录与决策摘要

> 本文档整理自 2026-03-16 与 AI 助手的对话，便于在其他设备继续 BigStrongGPT 重构与训练。
> 相关详细文档：`TRAINING_ROADMAP.md`、`DATA_SOURCES.md`

---

## 1. 项目现状（讨论起点）

**BigStrongGPT** 是一个从零搭建的中文小语言模型项目：

- **架构**：Decoder-Only Transformer（RoPE、RMSNorm、SwiGLU、GQA、Flash Attention、KV Cache）
- **现有规模**：约 25M 参数（512 dim × 8 layers，vocab 6400）
- **已有流程**：Pretrain → Full SFT → LoRA → Web Demo + RAG
- **主要问题**：三个训练脚本重复、硬编码路径、缺 DPO、SFT lr 偏低（5e-7）、无统一 eval

代码结构与 DeepSeek LLM（2024.01）架构高度同构，**重构重点在训练配方和工程，不在改架构**。

---

## 2. DeepSeek 三篇论文速览

| 论文 | 核心 | 预训练算力（约） | 对本项目的参考价值 |
|------|------|-----------------|-------------------|
| **DeepSeek LLM** (2024.01) | Dense 7B/67B，Scaling Laws，SFT + **DPO** | 67B×2T ≈ 60万 H800 GPU·h（~$120万） | ⭐⭐⭐ **首选**：训练配方、数据 pipeline |
| **DeepSeek-V2** (2024.05) | **MLA** + **DeepSeekMoE**，SFT + GRPO | 8.1T ≈ 140万 H800 GPU·h（~$280万） | ⭐⭐ 阶段 2 MoE 架构 |
| **DeepSeek-V3** (2024.12) | 671B MoE，FP8，DualPipe，MTP | 278.8万 H800 GPU·h（~$557万） | ⭐ 工程优化理念，小模型不必实现 |

**7B 从零预训练**：约 50~150 万元人民币，个人不可行。7B 仅作 Teacher 蒸馏，不训练。

---

## 3. 硬件与分工

| 设备 | 显存 | 用途 |
|------|------|------|
| **4070 Ti Super（Windows 本地）** | 16GB | 代码调试、7B 蒸馏数据生成、模型部署推理 |
| **AutoDL 4090/5090** | 24~32GB | 正式 Pretrain / SFT / DPO / MoE 训练 |

**原则**：训练上云，部署本地。

### 4070 部署能力

| 模型 | 能否部署 | 说明 |
|------|---------|------|
| 自研 150~250M | ✅ 非常轻松 | bf16，~0.5~0.7GB 显存 |
| 开源 7B | ✅ 可以 | 推荐 **int4 量化**（~5~7GB）；fp16 约 14GB 很紧 |

---

## 4. 最终确定的训练方案（方案 B）

### 4.1 模型规模

| 阶段 | 规模 | 配置概要 | 预训练数据 |
|------|------|---------|-----------|
| **阶段 1：Dense** | **150~200M** | 768 dim × 12 layers, vocab 16K, GQA 8/2 | 1.5~2B tokens |
| **阶段 2：MoE** | **250M 总参 / ~60M 激活** | + 2 shared + 24 routed experts, top-2, MLA 简化版 | 1~1.5B tokens |

### 4.2 训练流程（已确认）

```
Pretrain（250M 全参，AutoDL）
    ↓
4070：7B 批量生成 QA 蒸馏数据（可与 Pretrain 并行准备）
    ↓
SFT（250M 全参，数据 = 开源 SFT + 7B 生成数据）
    ↓
DPO（250M 全参）
    ↓
[可选] LoRA 微调 250M（领域适配，如医疗）
```

**三条确认**：

1. **250M 全参** Pretrain → SFT → DPO（主线）
2. **4070 上 7B 生成蒸馏数据**，合并进 SFT，不训练 7B
3. **可选 LoRA 微调 250M**（不是 7B）

### 4.3 阶段 2 MoE 参考来源

- **V2 必做**：DeepSeekMoE、MLA 简化版、Expert balance loss、Multi-step LR
- **V3 选做**：aux-loss-free 负载均衡（对比实验）
- **V3 跳过**：FP8、DualPipe、MTP、671B 规模

### 4.4 预算

| 项目 | 费用 |
|------|------|
| 阶段 1 Dense（Pretrain + SFT + DPO） | 60~100 元 |
| 阶段 2 MoE | 80~140 元 |
| 预留调参重跑 | 50~100 元 |
| **合计** | **约 200~350 元** |

---

## 5. 概念澄清（对话中讨论过）

| 术语 | 本质 | 是否 SFT | 本项目 |
|------|------|---------|--------|
| **SFT** | 监督学习，学标准答案 | — | ✅ 必做 |
| **LoRA SFT** | 只训少量 adapter 参数的 SFT | SFT 的一种实现 | ⚠️ 250M 可选领域微调；7B 可选对照 |
| **蒸馏** | 大模型当老师，小模型学 | ❌ 不是 | ✅ 7B 生成数据，不训 7B |
| **DPO** | 用好/坏回答对比优化 | ❌ 不是，是 SFT **之后**的对齐 | ✅ 建议做 |
| **RL / RLHF** | 奖励模型 + 强化学习 | ❌ 不是 | ❌ 暂不做 |
| **GRPO** | 无 Critic 的组内相对 RL（DeepSeek V2/V3） | ❌ 不是 | ❌ 暂不做 |

**正确理解**：SFT 是对话基础；DPO / RL / GRPO 是 SFT 之上的不同对齐方法。

---

## 6. 训练数据来源

详见 `docs/DATA_SOURCES.md`，一键下载：`python scripts/download_datasets.py --all`

### 推荐组合（方案 B）

| 阶段 | 数据集 | 抽样 |
|------|--------|------|
| Pretrain | SkyPile-150B（5~10 shard） | ~1.5B tokens |
| SFT | Belle 0.5M + 7B 蒸馏 | 各 ~2 万，合计 ~4 万 |
| DPO | shibing624/DPO-En-Zh-20k（中文） | 5000 |
| LoRA | huatuo_medical ShareGPT | 1 万 |
| 7B 蒸馏 | 4070 + Qwen2.5-7B 生成 | 1~5 万（脚本待写） |

### 输出文件

```
dataset/
├── pretrain_merged.jsonl   # Pretrain
├── sft_merged.jsonl          # SFT
├── dpo_merged.jsonl          # DPO（训练脚本待实现）
└── lora_medical.jsonl        # LoRA 可选
```

国内下载建议：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 7. 训练配方（对齐 DeepSeek LLM）

### Pretrain

```yaml
optimizer: AdamW(betas=[0.9, 0.95], weight_decay=0.1)
learning_rate: 3e-4 ~ 5e-4
lr_scheduler: multi-step  # warmup → 80% tokens ×0.316 → 90% tokens ×0.1
grad_clip: 1.0
precision: bf16 + fp32 grad accumulation
max_seq_len: 512
```

### SFT

```yaml
learning_rate: 1e-5 ~ 5e-6   # 现有 5e-7 太低，需调整
epochs: 2 ~ 4
```

### DPO

```yaml
learning_rate: 5e-6
epochs: 1
# 格式: {"prompt", "chosen", "rejected"}
```

---

## 8. 待办 Checklist（跨设备继续）

### Month 1 — 工程 + Dense

- [ ] 重构：统一 Trainer + YAML config + `pyproject.toml`
- [ ] 修复 `train/demo.py` chat_template bug
- [ ] 本机/AutoDL：`python scripts/download_datasets.py --all`
- [ ] 4070 本地：25M 跑通 Pretrain → SFT → DPO
- [ ] 4070：部署量化 Qwen2.5-7B，编写蒸馏数据生成脚本
- [ ] AutoDL：150~200M Pretrain（1.5~2B tokens）
- [ ] SFT（含 7B 蒸馏数据）→ DPO
- [ ] 4070 部署自研模型 + 评估

### Month 2 — MoE

- [ ] 实现 DeepSeekMoE + MLA 简化版
- [ ] AutoDL：250M MoE Pretrain → SFT → DPO
- [ ] Dense vs MoE 对比
- [ ] [可选] 250M LoRA 医疗领域

### 代码/文档已创建（本次对话）

| 文件 | 说明 |
|------|------|
| `docs/TRAINING_ROADMAP.md` | 完整训练路线图 |
| `docs/DATA_SOURCES.md` | 数据来源清单 |
| `docs/CONVERSATION_SUMMARY.md` | 本文档 |
| `scripts/download_datasets.py` | 数据集下载与格式转换 |

### 尚未实现

- [ ] 统一 Trainer + YAML config
- [ ] DPO 训练脚本
- [ ] `scripts/generate_distill_data.py`（7B 蒸馏数据生成）
- [ ] MoE / MLA 模型代码
- [ ] Eval 脚本

---

## 9. 关键决策记录

| 决策 | 结论 | 原因 |
|------|------|------|
| 是否从零训 7B | ❌ 否 | 成本 50~150 万，超出个人范围 |
| 7B 的角色 | Teacher 生成蒸馏数据 | 4070 免费，不训练 7B |
| LoRA 调谁 | 优先 250M 领域微调；7B LoRA 仅可选对照 | 250M 全参 SFT 足够 |
| 对齐方法 | SFT + DPO | DeepSeek LLM 路线，比 GRPO 简单 |
| 架构重构 | 保留现有 LLaMA 式架构 | 已与 DeepSeek LLM 同构 |
| 正式训练规模 | 150~200M Dense → 250M MoE | 4070 可部署，AutoDL ~300 元 |

---

## 10. 在其他设备继续的步骤

```bash
# 1. 拉代码（commit 后）
git clone <your-repo> && cd BigStrongGPT

# 2. 环境
pip install -r requirements.txt
pip install huggingface_hub datasets tqdm

# 3. 下载数据（4070 或 AutoDL）
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
python scripts/download_datasets.py --all

# 4. 阅读路线图
# docs/TRAINING_ROADMAP.md
# docs/DATA_SOURCES.md
# docs/CONVERSATION_SUMMARY.md  （本文）

# 5. 从工程重构或 25M 本地调试开始
```

---

*生成时间：2026-03-16*
