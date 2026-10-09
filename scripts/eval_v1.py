"""固定 prompt 看 v1 checkpoint 能否生成。质量在冒烟权重上没有意义。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from bigstrong.modeling import BigStrongConfig, BigStrongForCausalLLM  # noqa: E402

PROMPTS = [
    "你好，请做一下自我介绍。",
    "1+1 等于几？",
    "用一句话解释什么是预训练。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1/smoke_pretrain.yaml")
    parser.add_argument("--ckpt", default="output/v1/smoke_pretrain/last.pt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    with (REPO / args.config).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(REPO / cfg["tokenizer_dir"])
    mcfg = cfg["model"]
    config = BigStrongConfig(
        hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        vocab_size=mcfg["vocab_size"],
    )
    model = BigStrongForCausalLLM(config)
    ckpt = torch.load(REPO / args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        print("---")
        print(prompt)
        print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
