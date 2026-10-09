"""用量化 7B 老师批量生成 SFT 数据。不训练 7B。

示例（4070，约 1~5 万条可过夜）:
  python scripts/generate_distill.py --max-samples 20000

默认从 dataset/sft_merged.jsonl 取用户问题，写出 dataset/sft_distill_7b.jsonl。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            row = json.loads(line)
            conv = row.get("conversations") or []
            if not conv:
                continue
            text = (conv[0].get("content") or "").strip()
            if text:
                prompts.append(text)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompts", default=str(REPO / "dataset" / "sft_merged.jsonl"))
    parser.add_argument("--output", default=str(REPO / "dataset" / "sft_distill_7b.jsonl"))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true", help="不用 bitsandbytes 4bit")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(Path(args.prompts), args.max_samples)
    print(f"prompts={len(prompts)} model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs = {
                "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                "device_map": "auto",
            }
            print("使用 4bit 量化加载")
        except Exception as exc:
            print(f"4bit 不可用（{exc}），改为 bf16")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            answer = tokenizer.decode(ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            row = {"conversations": [{"content": prompt}, {"content": answer}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(prompts)}] {prompt[:40]} -> {len(answer)} chars")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
