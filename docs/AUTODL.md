# AutoDL 正式预训练

本地冒烟通过后再租卡。学生模型是 `src/bigstrong` 的 **768 × 12 层、GQA 8/2、词表 16384**，不是 `legacy/v0_25m` 的 25M。

## 1. 环境

```bash
git clone <本仓库>
cd BigStrongGPT
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers tokenizers pyyaml datasets tqdm huggingface_hub
```

分词器随仓库：`tokenizer/v1/`。不要在云上重训词表。

## 2. 预训练语料（约 1.5～2B tokens）

SkyPile 每个 head shard 约数千万 tokens。在数据盘执行，下 8～12 个 shard 后合并成一个 jsonl（字段 `text`）：

```bash
huggingface-cli download Skywork/SkyPile-150B \
  --repo-type dataset \
  --local-dir dataset/raw/skypile \
  --include "data/2021-43_zh_head_000*.jsonl"
```

把 shard 拼成 `dataset/pretrain_skypile_full.jsonl`，并把 `configs/v1/pretrain.yaml` 里的 `data.path` 指过去。Wikipedia 抽样只用于本地冒烟。

SFT / DPO 仍可用仓库里已转换的 `sft_merged.jsonl`、`dpo_merged.jsonl`，并在 SFT 前并入 `dataset/sft_distill_7b.jsonl`（本地 4070 用 `scripts/generate_distill.py` 生成）。

## 3. 训练

```bash
export PYTHONPATH=src:.
python -m bigstrong.train --config configs/v1/pretrain.yaml
# 中断后续训
python -m bigstrong.train --config configs/v1/pretrain.yaml --resume output/v1/pretrain/last.pt
python -m bigstrong.train --config configs/v1/sft.yaml
python -m bigstrong.train --config configs/v1/dpo.yaml
```

权重在 `output/v1/*/last.pt`。拉回本地后：

```bash
python scripts/eval_v1.py --config configs/v1/pretrain.yaml --ckpt output/v1/sft/last.pt
```

## 4. 蒸馏（在本地 4070，不在 AutoDL 训 7B）

```bash
python scripts/generate_distill.py --max-samples 20000
```

默认 4bit 加载 `Qwen/Qwen2.5-7B-Instruct`，从 `sft_merged.jsonl` 抽问题，写出 `dataset/sft_distill_7b.jsonl`。把该文件并进 SFT 的 `data.path`（或先合并 jsonl）再跑 `configs/v1/sft.yaml`。
