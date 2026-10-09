# BigStrongGPT v1

768 维 × 12 层、GQA 8/2、词表 16384。Legacy 25M 在 `legacy/v0_25m/`。

```powershell
conda activate BigStrongGPT
# 词表（已生成则可跳过）
python scripts/train_tokenizer_v1.py

$env:PYTHONPATH = "src;."
python -m bigstrong.train --config configs/v1/smoke_pretrain.yaml
python -m bigstrong.train --config configs/v1/smoke_pretrain.yaml --resume output/v1/smoke_pretrain/last.pt --max-steps 100
python scripts/eval_v1.py
```

正式训练见 `docs/AUTODL.md`。配置：`configs/v1/pretrain.yaml`、`sft.yaml`、`dpo.yaml`。
