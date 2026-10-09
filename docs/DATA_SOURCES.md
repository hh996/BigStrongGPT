# 训练数据来源清单

> 针对 250M 模型 Pretrain → SFT → DPO → LoRA 的推荐数据集，均可从 HuggingFace 获取。

---

## 快速下载（推荐）

```bash
cd /data1/huanghao3/BigStrongGPT

# 安装依赖
pip install huggingface_hub datasets tqdm

# 下载全部推荐数据（抽样版，适合先跑通流程）
python scripts/download_datasets.py --all

# 或分阶段下载
python scripts/download_datasets.py --pretrain --max-pretrain-samples 200000
python scripts/download_datasets.py --sft --max-sft-samples 30000
python scripts/download_datasets.py --dpo --max-dpo-samples 5000
python scripts/download_datasets.py --lora --max-lora-samples 10000
```

下载后文件位于 `dataset/raw/`（原始）和 `dataset/`（转换后，可直接训练）。

---

## 1. Pretrain（预训练）

**目标**：1.5~2B tokens  
**格式**：`{"text": "..."}`

| 数据集 | HuggingFace | 体量 | 说明 | 推荐度 |
|--------|-------------|------|------|--------|
| **SkyPile-150B** | [Skywork/SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) | 150B tokens（抽样） | 中文网页语料，已是 jsonl，字段 `text` | ⭐⭐⭐ 正式训练首选 |
| **CLUECorpus2020** | [CLUEbenchmark/CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020) | ~35B 字 | 中文 Common Crawl，需自行转 jsonl | ⭐⭐ |
| **Wikipedia 中文** | `wikipedia` config `20231101.zh` | 较小 | 适合本地快速调试 | ⭐ 调试 |
| **shibing624/medical** | [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) | 含 pretrain 子集 | 医疗领域 pretrain，LoRA 时用 | ⭐ 领域 |

### SkyPile 手动下载（AutoDL 正式训练）

SkyPile 全量约 600GB，**只下几个 shard 即可**（每个 shard 约 1~2GB）：

```bash
# 需先 huggingface-cli login（若数据集需授权，同意 Skywork Community License）
huggingface-cli download Skywork/SkyPile-150B \
  --repo-type dataset \
  --include "data/2021-43_zh_head_0000.jsonl" \
  --include "data/2021-43_zh_head_0001.jsonl" \
  --local-dir dataset/raw/skypile
```

每个 head shard 约数千万 tokens，下 5~10 个 shard 可达 1~2B tokens。

### 抽样建议

| 场景 | 样本数 / 体量 |
|------|--------------|
| 本地 4070 调试 | 5~10 万条（~100M tokens） |
| AutoDL 正式训练 | 100~200 万条（~1.5~2B tokens） |

---

## 2. SFT（监督微调）

**目标**：2~5 万条（250M 不必用百万级）  
**格式**：`{"conversations": [{"content": "..."}, {"content": "..."}]}`

| 数据集 | HuggingFace | 规模 | 说明 | 推荐度 |
|--------|-------------|------|------|--------|
| **Belle 0.5M** | [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | 50 万 | 中文指令，instruction/output 格式，脚本自动转换 | ⭐⭐⭐ |
| **Alpaca 中文** | [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) | 2 万 | 小巧，适合调试 | ⭐⭐⭐ 调试 |
| **COIG** | [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) | 多文件 | 中文多任务，需下载 jsonl 后转换 | ⭐⭐ |
| **MOSS SFT** | [fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data) | 中等 | 中文多轮 | ⭐⭐ |
| **7B 蒸馏生成** | 本地生成 | 1~5 万 | 4070 + Qwen2.5-7B，见 `scripts/generate_distill.py` | ⭐⭐⭐ 推荐 |

### 手动下载 Belle

```bash
huggingface-cli download BelleGroup/train_0.5M_CN \
  --repo-type dataset \
  --local-dir dataset/raw/belle
```

---

## 3. DPO（偏好对齐）

**目标**：500~5000 对  
**格式**：`{"prompt": "...", "chosen": "...", "rejected": "..."}`

| 数据集 | HuggingFace | 规模 | 说明 | 推荐度 |
|--------|-------------|------|------|--------|
| **DPO 中英 20k** | [shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference) | 中文 1 万 | 字段 question/response_chosen/response_rejected，脚本自动转换 | ⭐⭐⭐ |
| **COIG-P** | [m-a-p/COIG-P](https://huggingface.co/datasets/m-a-p/COIG-P) | 100 万 | 大规模中文偏好，可抽样 5000 | ⭐⭐ |
| **btfChinese-DPO-small** | [Karsh-CAI/btfChinese-DPO-small](https://huggingface.co/datasets/Karsh-CAI/btfChinese-DPO-small) | 5000 | 小巧，快速验证 DPO | ⭐⭐ 调试 |
| **自动生成** | 7B vs 250M | 自定义 | chosen=7B 回答，rejected=250M 差回答 | ⭐⭐⭐ |

---

## 4. LoRA 领域微调（可选）

**目标**：5000~2 万条  
**格式**：同 SFT

| 数据集 | HuggingFace | 规模 | 说明 | 推荐度 |
|--------|-------------|------|------|--------|
| **华佗医疗 ShareGPT** | [shibing624/huatuo_medical_qa_sharegpt](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt) | 22 万 | 中文医疗对话 | ⭐⭐⭐ |
| **HuatuoGPT2 SFT** | [FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K) | 14 万 | GPT-4 生成医疗指令 | ⭐⭐ |
| **shibing624/medical** | [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) | 240 万 | 含 pretrain/SFT/reward 多种 | ⭐⭐ |

---

## 5. 7B 蒸馏数据（4070 本地生成）

不下载，用 7B 自己造。Prompt 来源：

| Prompt 来源 | 说明 |
|------------|------|
| `dataset/sft_merged.jsonl` 中的 user 问题 | 让 7B 重新生成更好的回答 |
| `shibing624/alpaca-zh` 的 instruction | 小巧，适合先试 |
| 自建 prompt 列表 | 写作、数学、代码、日常等 |

生成脚本（待实现）：`scripts/generate_distill_data.py`

---

## 6. 推荐组合（直接照着用）

### 方案 A：最小调试（4070 本地，几 GB）

| 阶段 | 数据集 | 抽样 |
|------|--------|------|
| Pretrain | Wikipedia 中文 | 5 万条 |
| SFT | alpaca-zh | 全量 2 万 |
| DPO | btfChinese-DPO-small | 5000 |
| LoRA | huatuo 抽样 | 5000 |

### 方案 B：正式训练（AutoDL，推荐）

| 阶段 | 数据集 | 抽样 |
|------|--------|------|
| Pretrain | SkyPile-150B | 5~10 个 shard（~1.5B tokens） |
| SFT | Belle 0.5M + 7B 蒸馏 | 各 2 万，合计 ~4 万 |
| DPO | DPO-En-Zh 中文部分 | 5000 |
| LoRA | huatuo_medical | 1 万 |

---

## 7. 输出文件对照

| 训练阶段 | 转换后路径 | 训练脚本参数 |
|---------|-----------|-------------|
| Pretrain | `dataset/pretrain_merged.jsonl` | `--data_path ../dataset/pretrain_merged.jsonl` |
| SFT | `dataset/sft_merged.jsonl` | `--data_path ../dataset/sft_merged.jsonl` |
| DPO | `dataset/dpo_merged.jsonl` | （DPO 脚本待实现） |
| LoRA | `dataset/lora_medical.jsonl` | `--data_path ../dataset/lora_medical.jsonl` |

---

## 8. 注意事项

1. **License**：SkyPile 需遵守 [Skywork Community License](https://huggingface.co/datasets/Skywork/SkyPile-150B)；医疗数据仅供研究学习。
2. **gitignore**：`*.jsonl` 已被忽略，数据不会进 git，需在 AutoDL 上重新下载或 scp 传输。
3. **去重**：Pretrain 正式训练前建议 dedup；SFT/DPO 合并后 `sort -u` 或脚本去重。
4. **中文为主**：250M 小模型建议中文占比 >80%。

---

*最后更新：2026-03-16*
