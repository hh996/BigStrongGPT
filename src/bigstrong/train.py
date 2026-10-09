"""python -m bigstrong.train --config configs/v1/smoke_pretrain.yaml"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
for p in (str(SRC), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bigstrong.trainer import Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="BigStrongGPT v1 训练")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="last.pt 路径，接着训")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = args.max_steps

    tok_dir = REPO / cfg["tokenizer_dir"]
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    trainer = Trainer(cfg, REPO)
    if args.resume:
        trainer.load_resume(args.resume)
    trainer.fit(tokenizer)


if __name__ == "__main__":
    main()
