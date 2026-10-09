"""统一训练循环：pretrain / sft / dpo，由 YAML 切换数据和损失。"""
from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.lm_dataset import DPODataset, PretrainDataset, SFTDataset
from bigstrong.modeling import BigStrongConfig, BigStrongForCausalLLM


def cosine_lr(step: int, total_steps: int, lr: float, min_ratio: float = 0.1) -> float:
    if total_steps <= 1:
        return lr
    return min_ratio * lr + 0.5 * lr * (1 - min_ratio) * (1 + math.cos(math.pi * step / total_steps))


def masked_ce(logits, labels, loss_mask):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
    denom = loss_mask.sum().clamp_min(1)
    return (loss * loss_mask).sum() / denom


def sequence_logps(model, input_ids, labels, loss_mask):
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1)


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


class Trainer:
    def __init__(self, cfg: dict, repo_root: Path):
        self.cfg = cfg
        self.repo_root = repo_root
        train = cfg["train"]
        self.device = train.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.stage = cfg["stage"]
        self.output_dir = repo_root / cfg["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = train.get("max_steps")
        self.grad_clip = float(train.get("grad_clip", 1.0))
        self.log_interval = int(train.get("log_interval", 10))
        self.save_interval = int(train.get("save_interval", 200))
        self.accum = int(train.get("accumulation_steps", 1))
        self.lr = float(train["learning_rate"])
        self.beta = float(train.get("beta", 0.1))
        self.epochs = int(train.get("epochs", 1))

        model_cfg = cfg["model"]
        self.lm_config = BigStrongConfig(
            hidden_size=int(model_cfg["hidden_size"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 8)),
            num_key_value_heads=int(model_cfg.get("num_key_value_heads", 2)),
            vocab_size=int(model_cfg["vocab_size"]),
            max_position_embeddings=int(model_cfg.get("max_position_embeddings", 32768)),
        )
        self.model = BigStrongForCausalLLM(self.lm_config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[{self.stage}] 参数量 {n_params:.2f}M  device={self.device}")

        init_ckpt = cfg.get("init_ckpt")
        if init_ckpt:
            path = repo_root / init_ckpt
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
            print(f"加载初始权重: {path}")

        self.ref_model = None
        if self.stage == "dpo":
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=tuple(train.get("betas", [0.9, 0.95])),
            weight_decay=float(train.get("weight_decay", 0.1)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.startswith("cuda"))
        self.state = TrainState()

    def _loader(self, tokenizer):
        data = self.cfg["data"]
        path = str(self.repo_root / data["path"]) if not os.path.isabs(data["path"]) else data["path"]
        max_len = int(self.cfg["train"]["max_seq_len"])
        max_samples = data.get("max_samples")
        if self.stage == "pretrain":
            ds = PretrainDataset(path, tokenizer, max_length=max_len)
        elif self.stage == "sft":
            ds = SFTDataset(path, tokenizer, max_length=max_len)
        elif self.stage == "dpo":
            ds = DPODataset(path, tokenizer, max_length=max_len, max_samples=max_samples)
        else:
            raise ValueError(f"未知 stage: {self.stage}")
        if max_samples and self.stage != "dpo":
            ds.samples = ds.samples[: int(max_samples)]
        return DataLoader(
            ds,
            batch_size=int(self.cfg["train"]["batch_size"]),
            shuffle=True,
            drop_last=False,
            num_workers=int(self.cfg["train"].get("num_workers", 0)),
            pin_memory=self.device.startswith("cuda"),
        )

    def _loss(self, batch):
        if self.stage == "dpo":
            c_x, c_y, c_m, r_x, r_y, r_m = [t.to(self.device) for t in batch]
            pi_c = sequence_logps(self.model, c_x, c_y, c_m)
            pi_r = sequence_logps(self.model, r_x, r_y, r_m)
            with torch.no_grad():
                ref_c = sequence_logps(self.ref_model, c_x, c_y, c_m)
                ref_r = sequence_logps(self.ref_model, r_x, r_y, r_m)
            logits = self.beta * ((pi_c - pi_r) - (ref_c - ref_r))
            loss = -F.logsigmoid(logits).mean()
            extra = {"reward_acc": (logits > 0).float().mean().item()}
            return loss, extra
        x, y, mask = [t.to(self.device) for t in batch]
        loss = masked_ce(self.model(x).logits, y, mask)
        return loss, {}

    def save(self, tag: str = "last"):
        path = self.output_dir / f"{tag}.pt"
        torch.save(
            {
                "model": {k: v.half() for k, v in self.model.state_dict().items()},
                "optimizer": self.optimizer.state_dict(),
                "step": self.state.step,
                "epoch": self.state.epoch,
                "config": self.lm_config.to_dict(),
                "stage": self.stage,
            },
            path,
        )
        print(f"已保存 {path}")
        return path

    def load_resume(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.state.step = int(ckpt.get("step", 0))
        self.state.epoch = int(ckpt.get("epoch", 0))
        print(f"从 step={self.state.step} 继续: {path}")

    def fit(self, tokenizer):
        loader = self._loader(tokenizer)
        steps_per_epoch = max(len(loader), 1)
        planned = self.epochs * steps_per_epoch
        if self.max_steps:
            planned = min(planned, int(self.max_steps))
        ctx = torch.amp.autocast("cuda") if self.device.startswith("cuda") else torch.autocast("cpu", enabled=False)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        start = time.time()
        stop = False
        for epoch in range(self.state.epoch, self.epochs):
            self.state.epoch = epoch
            for batch in loader:
                if self.max_steps and self.state.step >= int(self.max_steps):
                    stop = True
                    break
                lr = cosine_lr(self.state.step, planned, self.lr)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                with ctx:
                    loss, extra = self._loss(batch)
                    loss = loss / self.accum
                self.scaler.scale(loss).backward()
                if (self.state.step + 1) % self.accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                if self.state.step % self.log_interval == 0:
                    elapsed = time.time() - start
                    msg = (
                        f"epoch {epoch + 1}/{self.epochs} step {self.state.step}/{planned} "
                        f"loss {loss.item() * self.accum:.4f} lr {lr:.6g} elapsed {elapsed:.0f}s"
                    )
                    if extra:
                        msg += " " + " ".join(f"{k}={v:.3f}" for k, v in extra.items())
                    print(msg)
                self.state.step += 1
                if self.save_interval and self.state.step % self.save_interval == 0:
                    self.save("last")
            if stop:
                break
        self.save("last")
        return self.output_dir / "last.pt"
