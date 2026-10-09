"""训练 v1 BPE 分词器，词表 16384，供 768×12 模型使用。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "tokenizer" / "v1"
DATA_FILES = [
    REPO / "dataset" / "pretrain_wikipedia.jsonl",
    REPO / "dataset" / "pretrain_skypile.jsonl",
]
VOCAB_SIZE = 16384
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def iter_texts(limit_per_file: int | None):
    for path in DATA_FILES:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit_per_file is not None and i >= limit_per_file:
                    break
                if not line.strip():
                    continue
                yield json.loads(line)["text"]


def train(limit_per_file: int | None = None):
    print(f"语料: {[str(p) for p in DATA_FILES]} vocab={VOCAB_SIZE}")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_texts(limit_per_file), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    tokenizer.model.save(str(OUT_DIR))
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "1": {"content": "<|im_start|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "2": {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}",
    }
    (OUT_DIR / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 {OUT_DIR}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-file", type=int, default=None)
    args = parser.parse_args()
    train(args.limit_per_file)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    print("词表长度", len(tok))


if __name__ == "__main__":
    main()
