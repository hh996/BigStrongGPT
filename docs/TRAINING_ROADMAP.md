# BigStrongGPT 训练路线图

> 目标：在个人算力下（4070 Ti Super + AutoDL），走一遍 DeepSeek 风格的完整 LLM 训练流程，并借助 7B 做蒸馏增强。

**代码布局**：25M 可运行实现在 `legacy/v0_25m/`；本路线图对应的新工程在 `src/`。`dataset/`、`docs/` 在仓库根目录共用。

---

## 1. 硬件与分工

| 硬件 | 显存 | 用途 |
|------|------|------|
| **4070 Ti Super（本地 Windows）** | 16GB | 代码调试、7B 蒸馏数据生成、本地部署推理 |
| **AutoDL 4090/5090** | 24~32GB | 正式预训练 / SFT / DPO / MoE 训练 |

**原则**：训练上云，部署本地。

---

## 2. 模型规模（方案 B，推荐）

| 阶段 | 模型 | 配置概要 | 预训练数据 |
|------|------|---------|-----------|
| **阶段 1：Dense LLM** | **150~200M** | 768 dim × 12 layers, vocab 16K, GQA 8/2 | 1.5~2B tokens |
| **阶段 2：小 MoE** | **250M 总参 / ~60M 激活** | 同上 + 2 shared + 24 routed experts, top-2, MLA 简化版 | 1~1.5B tokens |

**不追求 7B**：7B 从零预训练需 50~150 万元，超出个人学习范围。7B 仅作为 **Teacher（老师）** 参与蒸馏，不参与主线训练。

**4070 部署**：150~250M 自训模型可全精度部署；7B 开源模型用 int4 量化部署（~5~7GB 显存）。

---

## 3. 完整训练流程

```
┌─────────────────────────────────────────────────────────────┐
│  主线：自研小模型（AutoDL）                                    │
├─────────────────────────────────────────────────────────────┤
│  Pretrain → SFT → DPO → [可选] 蒸馏数据再 SFT 一轮           │
│                                                              │
│  阶段 2 追加：MoE Pretrain → SFT → DPO                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  副线：7B Teacher（4070 本地，免费）                           │
├─────────────────────────────────────────────────────────────┤
│  下载量化 Qwen2.5-7B → 批量生成高质量 QA → 供主线 SFT/蒸馏    │
│  [可选] 7B LoRA 微调 → 仅作效果对照，非主线                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 各阶段说明

| 阶段 | 参考 DeepSeek | 做什么 | 数据 |
|------|--------------|--------|------|
| **Pretrain** | LLM §2 | 语言建模，学「说话」 | jsonl 纯文本 / 1.5~2B tokens |
| **SFT** | LLM §4 | 监督微调，学「对话」 | 指令 Q&A，含 7B 生成的数据 |
| **DPO** | LLM §4 | 偏好对齐，减少重复/废话 | chosen vs rejected 对，500~5000 条 |
| **蒸馏增强** | V3 理念（简化） | 7B 生成数据，250M 再 SFT 一轮 | 7B 在 4070 上批量生成 |
| **MoE 阶段** | V2 架构 + V3 稳定性 | MLA + DeepSeekMoE block | 同上流程 |

### 3.2 阶段 2 MoE 参考来源

| 技术 | 来源 | 是否实现 |
|------|------|---------|
| DeepSeekMoE（shared + routed） | V2 | ✅ 必做 |
| MLA（KV 压缩） | V2 | ✅ 简化版 |
| Expert balance loss | V2 | ✅ 必做 |
| Multi-step LR + batch scheduling | V1/V2 | ✅ 必做 |
| Aux-loss-free 负载均衡 | V3 | ⚠️ 有余力再对比 |
| FP8 / DualPipe / MTP | V3 | ❌ 跳过 |

### 3.3 明确不做（现阶段）

| 方法 | 原因 |
|------|------|
| GRPO / RL / RLHF | 实现复杂，250M 上性价比低；DeepSeek V2/V3 才用 |
| 7B 从零预训练 | 成本 50~150 万元 |
| R1 推理蒸馏 | 250M 容量太小，工程量大 |

---

## 4. 蒸馏 vs LoRA：怎么做？

### 4.1 蒸馏（推荐，主线的一部分）

**7B 当老师，250M 当学生。**

```
4070 本地运行量化 7B（Qwen2.5-7B-Instruct int4）
        ↓
对 prompt 列表批量生成高质量回答
        ↓
存为 jsonl（conversations 格式）
        ↓
合并进 250M 的 SFT 数据（或 Pretrain 后再 SFT 时使用）
        ↓
250M 全参数 SFT 学习这些回答
```

- **不需要训练 7B**，只需推理生成数据
- **4070 免费**，耗时主要是生成速度（可过夜跑）
- 建议生成 **1~5 万条** QA，与现有 sft 数据混合

### 4.2 LoRA 微调：调 7B 还是调小模型？

| 对象 | 建议 | 理由 |
|------|------|------|
| **自研 250M** | **主线用全参数 SFT**，LoRA 仅作可选 | 250M 很小，4090 全参 SFT 无压力；全参效果通常更好 |
| **自研 250M LoRA** | ⚠️ 可选：领域适配（如医疗 lora_medical） | 保留现有 `train_lora.py` 思路，用于快速领域实验 |
| **7B LoRA** | ⚠️ 可选对照实验，**非主线** | 4070 能跑；与 250M 同数据微调，对比「大模型 LoRA vs 小模型全参」 |

**结论**：

1. **主线**：250M **全参数** Pretrain → SFT → DPO（+ 7B 蒸馏数据）
2. **7B**：只做 **Teacher 生成数据**，不必 LoRA，除非你想单独拥有一个可定制的 7B 助手
3. **LoRA 7B**：可选 side project，用来对比效果，**不影响 250M 主线**
4. **LoRA 250M**：领域微调时用（如医疗），不是主流程必需

```
推荐优先级：

  必做：250M 全参 SFT + DPO
  推荐：7B 蒸馏数据（4070 生成，不训 7B）
  可选：250M LoRA 领域微调
  可选：7B LoRA 对照实验
```

---

## 5. 训练配方（对齐 DeepSeek LLM）

### Pretrain

```yaml
optimizer: AdamW
betas: [0.9, 0.95]
weight_decay: 0.1
learning_rate: 3e-4 ~ 5e-4
lr_scheduler: multi-step  # warmup 200~500 → 80% tokens ×0.316 → 90% tokens ×0.1
grad_clip: 1.0
precision: bf16 + fp32 grad accumulation
max_seq_len: 512
```

### SFT

```yaml
learning_rate: 1e-5 ~ 5e-6
epochs: 2 ~ 4
# 数据：现有 sft_*.jsonl + 7B 生成数据
```

### DPO

```yaml
learning_rate: 5e-6
epochs: 1
# 数据：500~5000 条 chosen/rejected 对
# 可自动生成：7B 回答 = chosen，250M 差回答 = rejected
```

---

## 6. 预算与时间

| 项目 | 时间 | 费用 |
|------|------|------|
| 工程重构 + 25M 本地调试 | 1 周 | 0 元 |
| 阶段 1：150~200M Dense（Pretrain + SFT + DPO） | 2~3 天 AutoDL | 60~100 元 |
| 7B 蒸馏数据生成 | 1~2 天 4070 本地 | 0 元 |
| 阶段 2：250M MoE | 2~3 天 AutoDL | 80~140 元 |
| 失败重跑 / 调参预留 | — | 50~100 元 |
| **合计** | **约 3~4 周** | **约 200~350 元** |

---

## 7. 执行顺序（Checklist）

### Month 1 — 工程 + Dense

- [ ] 重构代码：统一 Trainer + YAML config
- [ ] 本地 4070：25M 跑通 Pretrain → SFT → DPO
- [ ] 4070：部署量化 7B，编写蒸馏数据生成脚本
- [ ] AutoDL：150~200M 正式 Pretrain（1.5~2B tokens）
- [ ] 合并 7B 生成数据，做 SFT
- [ ] DPO 对齐
- [ ] 4070 本地部署 + 评估

### Month 2 — MoE

- [ ] 实现 DeepSeekMoE block + MLA 简化版
- [ ] 本地验证 forward / backward / load balance
- [ ] AutoDL：250M MoE Pretrain → SFT → DPO
- [ ] 对比 Dense vs MoE（同算力 loss、推理速度、对话质量）
- [ ] [可选] 7B LoRA 对照实验
- [ ] [可选] 250M LoRA 领域微调（医疗等）

---

## 8. 概念速查

| 术语 | 一句话 | 本项目 |
|------|--------|--------|
| **SFT** | 用标准 Q&A 教对话 | ✅ 必做（250M 全参） |
| **DPO** | 用好/坏回答对比优化 | ✅ 建议做 |
| **蒸馏** | 大模型当老师，小模型学 | ✅ 7B 生成数据，不训 7B |
| **LoRA** | 只训少量参数，省显存 | ⚠️ 250M 领域实验 / 7B 可选对照 |
| **RL / GRPO** | 强化学习对齐 | ❌ 暂不做 |

---

## 9. 部署方案

| 模型 | 平台 | 精度 | 显存 |
|------|------|------|------|
| 自研 150~250M | 4070 本地 | bf16 | ~0.5~0.7 GB |
| 自研 250M MoE | 4070 本地 | bf16 | ~0.7 GB |
| 开源 7B（对照/备用） | 4070 本地 | int4 量化 | ~5~7 GB |
| Web Demo + RAG | 4070 本地 | Streamlit | 与上共存 |

---

## 10. 参考论文

| 论文 | 链接 | 本项目参考内容 |
|------|------|---------------|
| DeepSeek LLM | [arxiv:2401.02954](https://arxiv.org/pdf/2401.02954) | 架构、Pretrain 配方、SFT、DPO、Scaling Laws |
| DeepSeek-V2 | [arxiv:2405.04434](https://arxiv.org/pdf/2405.04434) | MLA、DeepSeekMoE、负载均衡 |
| DeepSeek-V3 | [arxiv:2412.19437](https://arxiv.org/pdf/2412.19437) | 训练稳定性、aux-loss-free（选做） |

---

## 11. 训练数据

- 讨论记录与决策摘要：**[CONVERSATION_SUMMARY.md](./CONVERSATION_SUMMARY.md)**
- 数据来源详见 **[DATA_SOURCES.md](./DATA_SOURCES.md)**，一键下载：

```bash
pip install huggingface_hub datasets tqdm
python scripts/download_datasets.py --all
```

---

*最后更新：2026-03-16*
