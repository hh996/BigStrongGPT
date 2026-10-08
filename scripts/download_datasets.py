#!/usr/bin/env python3
"""
下载并转换训练数据为 BigStrongGPT 项目格式。

用法:
  python scripts/download_datasets.py --all
  python scripts/download_datasets.py --pretrain --sft --dpo --lora
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
OUT_DIR = ROOT / "dataset"


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  写入 {path} ({len(rows)} 条)")


def load_hf_dataset(name: str, config: str | None = None, split: str = "train"):
    from datasets import load_dataset

    if config:
        return load_dataset(name, config, split=split)
    return load_dataset(name, split=split)


def sample_rows(rows: list[dict], max_samples: int | None, seed: int = 42) -> list[dict]:
    if max_samples is None or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_samples)


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------

def download_pretrain_wikipedia(max_samples: int | None = 100_000) -> Path:
    """Wikipedia 中文，适合本地调试。"""
    print("\n[Pretrain] Wikipedia 中文...")
    try:
        ds = load_hf_dataset("wikimedia/wikipedia", "20231101.zh")
    except Exception:
        # 兼容旧版 datasets 的加载方式
        ds = load_hf_dataset("wikipedia", "20231101.zh")
    rows = []
    for item in tqdm(ds, desc="  转换"):
        text = (item.get("text") or "").strip()
        if len(text) < 100:
            continue
        rows.append({"text": text})
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "pretrain_wikipedia.jsonl"
    write_jsonl(out, rows)
    return out


def download_pretrain_skypile(max_files: int = 2, max_samples: int | None = None) -> Path | None:
    """下载 SkyPile 前几个 shard（需网络 + 可能需 HF 授权）。"""
    print("\n[Pretrain] SkyPile-150B（抽样 shard）...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  跳过：需要 pip install huggingface_hub")
        return None

    shard_names = [
        "data/2021-43_zh_head_0000.jsonl",
        "data/2021-43_zh_head_0001.jsonl",
        "data/2021-43_zh_head_0002.jsonl",
        "data/2021-43_zh_head_0003.jsonl",
        "data/2021-43_zh_head_0004.jsonl",
    ]
    rows = []
    for shard in shard_names[:max_files]:
        print(f"  下载 {shard} ...")
        try:
            local = hf_hub_download(
                repo_id="Skywork/SkyPile-150B",
                repo_type="dataset",
                filename=shard,
                local_dir=RAW_DIR / "skypile",
            )
            with open(local, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = (data.get("text") or "").strip()
                    if len(text) >= 50:
                        rows.append({"text": text})
        except Exception as e:
            print(f"  下载失败 {shard}: {e}")
            break

    if not rows:
        print("  SkyPile 下载失败，请手动下载或改用 --pretrain-wiki")
        return None

    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "pretrain_skypile.jsonl"
    write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def belle_to_conversations(instruction: str, inp: str, output: str) -> dict:
    user = instruction.strip()
    if inp and inp.strip():
        user = f"{user}\n{inp.strip()}"
    return {
        "conversations": [
            {"content": user},
            {"content": output.strip()},
        ]
    }


def download_sft_alpaca_zh(max_samples: int | None = None) -> Path:
    print("\n[SFT] shibing624/alpaca-zh ...")
    ds = load_hf_dataset("shibing624/alpaca-zh")
    rows = []
    for item in tqdm(ds, desc="  转换"):
        inst = item.get("instruction") or ""
        inp = item.get("input") or ""
        out = item.get("output") or ""
        if not out.strip():
            continue
        rows.append(belle_to_conversations(inst, inp, out))
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "sft_alpaca_zh.jsonl"
    write_jsonl(out, rows)
    return out


def download_sft_belle(max_samples: int | None = 30_000) -> Path | None:
    print("\n[SFT] BelleGroup/train_0.5M_CN ...")
    try:
        ds = load_hf_dataset("BelleGroup/train_0.5M_CN")
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

    rows = []
    for item in tqdm(ds, desc="  转换"):
        rows.append(
            belle_to_conversations(
                item.get("instruction") or "",
                item.get("input") or "",
                item.get("output") or "",
            )
        )
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "sft_belle.jsonl"
    write_jsonl(out, rows)
    return out


def merge_sft_files(out_name: str = "sft_merged.jsonl") -> Path:
    files = sorted(OUT_DIR.glob("sft_*.jsonl"))
    files = [f for f in files if f.name != out_name]
    if not files:
        print("  无 SFT 文件可合并")
        return OUT_DIR / out_name

    seen = set()
    merged = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(json.loads(line))

    out = OUT_DIR / out_name
    write_jsonl(out, merged)
    return out


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def download_dpo_shibing(max_samples: int | None = 5_000) -> Path:
    print("\n[DPO] shibing624/DPO-En-Zh-20k-Preference (中文部分) ...")
    ds = load_hf_dataset("shibing624/DPO-En-Zh-20k-Preference", "zh")
    rows = []
    for item in tqdm(ds, desc="  转换"):
        # 优先中文字段；该数据集含 system/history/question
        q = item.get("question") or item.get("prompt") or ""
        chosen = item.get("response_chosen") or item.get("chosen") or ""
        rejected = item.get("response_rejected") or item.get("rejected") or ""
        system = item.get("system") or ""
        if system:
            q = f"{system}\n{q}".strip()
        if not q or not chosen or not rejected:
            continue
        # 简单过滤：中文占比（含 CJK 字符）
        if sum(1 for c in q + chosen if "\u4e00" <= c <= "\u9fff") < 5:
            continue
        rows.append({"prompt": q.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()})

    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "dpo_zh.jsonl"
    write_jsonl(out, rows)
    return out


def download_dpo_small(max_samples: int | None = 5_000) -> Path | None:
    print("\n[DPO] Karsh-CAI/btfChinese-DPO-small ...")
    try:
        ds = load_hf_dataset("Karsh-CAI/btfChinese-DPO-small")
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

    rows = []
    for item in tqdm(ds, desc="  转换"):
        prompt = item.get("prompt") or item.get("question") or ""
        chosen = item.get("chosen") or item.get("response_chosen") or ""
        rejected = item.get("rejected") or item.get("response_rejected") or ""
        if prompt and chosen and rejected:
            rows.append({"prompt": prompt.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()})
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "dpo_btf_small.jsonl"
    write_jsonl(out, rows)
    return out


def merge_dpo_files(out_name: str = "dpo_merged.jsonl") -> Path:
    files = [OUT_DIR / "dpo_zh.jsonl", OUT_DIR / "dpo_btf_small.jsonl"]
    files = [f for f in files if f.exists()]
    merged = []
    seen = set()
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() in seen:
                    continue
                seen.add(line.strip())
                merged.append(json.loads(line))
    out = OUT_DIR / out_name
    write_jsonl(out, merged)
    return out


# ---------------------------------------------------------------------------
# LoRA 医疗
# ---------------------------------------------------------------------------

def sharegpt_to_conversations(conversations: list) -> dict | None:
    """ShareGPT: from human/gpt 或 role/content."""
    msgs = []
    for turn in conversations:
        if isinstance(turn, dict):
            role = turn.get("from") or turn.get("role") or ""
            content = turn.get("value") or turn.get("content") or ""
            role = role.lower()
            if role in ("human", "user"):
                msgs.append({"content": content.strip()})
            elif role in ("gpt", "assistant", "bot"):
                msgs.append({"content": content.strip()})
    if len(msgs) >= 2 and len(msgs) % 2 == 0:
        return {"conversations": msgs}
    return None


def download_lora_huatuo(max_samples: int | None = 10_000) -> Path | None:
    print("\n[LoRA] shibing624/huatuo_medical_qa_sharegpt ...")
    try:
        ds = load_hf_dataset("shibing624/huatuo_medical_qa_sharegpt")
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

    rows = []
    for item in tqdm(ds, desc="  转换"):
        conv = item.get("conversations") or item.get("conversation") or []
        parsed = sharegpt_to_conversations(conv)
        if parsed:
            rows.append(parsed)
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "lora_medical.jsonl"
    write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="下载并转换 BigStrongGPT 训练数据")
    parser.add_argument("--all", action="store_true", help="下载全部（抽样版）")
    parser.add_argument("--pretrain", action="store_true", help="预训练数据")
    parser.add_argument("--pretrain-wiki", action="store_true", help="仅 Wikipedia 中文（调试）")
    parser.add_argument("--pretrain-skypile", action="store_true", help="SkyPile shard（正式）")
    parser.add_argument("--skypile-files", type=int, default=2, help="SkyPile shard 数量")
    parser.add_argument("--sft", action="store_true", help="SFT 数据")
    parser.add_argument("--dpo", action="store_true", help="DPO 数据")
    parser.add_argument("--lora", action="store_true", help="LoRA 医疗数据")
    parser.add_argument("--max-pretrain-samples", type=int, default=100_000)
    parser.add_argument("--max-sft-samples", type=int, default=30_000)
    parser.add_argument("--max-dpo-samples", type=int, default=5_000)
    parser.add_argument("--max-lora-samples", type=int, default=10_000)
    args = parser.parse_args()

    if args.all:
        args.pretrain = args.sft = args.dpo = args.lora = True
        args.pretrain_wiki = True

    if not any([args.pretrain, args.pretrain_wiki, args.pretrain_skypile, args.sft, args.dpo, args.lora]):
        parser.print_help()
        return

    ensure_dirs()
    print(f"输出目录: {OUT_DIR}")

    if args.pretrain or args.pretrain_wiki:
        download_pretrain_wikipedia(max_samples=args.max_pretrain_samples)

    if args.pretrain or args.pretrain_skypile:
        download_pretrain_skypile(max_files=args.skypile_files, max_samples=args.max_pretrain_samples)

    if args.pretrain and not args.pretrain_wiki and not args.pretrain_skypile:
        download_pretrain_wikipedia(max_samples=args.max_pretrain_samples)

    if args.sft:
        download_sft_alpaca_zh(max_samples=min(args.max_sft_samples, 20_000))
        download_sft_belle(max_samples=args.max_sft_samples)
        merge_sft_files()

    if args.dpo:
        download_dpo_shibing(max_samples=args.max_dpo_samples)
        download_dpo_small(max_samples=args.max_dpo_samples)
        merge_dpo_files()

    if args.lora:
        download_lora_huatuo(max_samples=args.max_lora_samples)

    print("\n完成！请查看 dataset/ 目录。")
    print("训练示例:")
    print("  cd legacy/v0_25m/train")
    print("  python train_pretrain.py")
    print("  python train_full_sft.py")
    print("  python train_lora.py")


if __name__ == "__main__":
    main()
