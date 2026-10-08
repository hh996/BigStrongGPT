# BigStrongGPT v0.25M（Legacy）

约 **25M** 参数（512 dim × 8 layers）的原始实现：Pretrain → Full SFT → LoRA → Web Demo + RAG。

- **数据**：使用仓库根目录 `dataset/`（与新版共用）
- **权重 / 分词器**：`model/`、`output/` 在本目录下
- **新版开发**：见仓库根目录 `src/` 与 `docs/TRAINING_ROADMAP.md`

## 快速开始

```powershell
conda activate BigStrongGPT
cd legacy\v0_25m\scripts
python train_tokenizer.py   # 需先在脚本中启用 train_tokenizer()，数据指向 ../../dataset/

cd ..\train
python train_pretrain.py --data_path ../../dataset/pretrain_wikipedia.jsonl
python train_full_sft.py --data_path ../../dataset/sft_merged.jsonl
python train_lora.py --data_path ../../dataset/lora_medical.jsonl
```

Web Demo（在 `scripts/` 下）：

```powershell
$env:PYTHONPATH = (Resolve-Path ..\..).Path + ";" + (Resolve-Path ..).Path
streamlit run web_demo.py
```

训练脚本已自动把 **仓库根** 与 **legacy/v0_25m** 加入 `PYTHONPATH`，一般无需手动设置。
