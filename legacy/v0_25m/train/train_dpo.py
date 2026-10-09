import logging
import os
import sys

__package__ = "trainer"

_V0_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _V0_ROOT not in sys.path:
    sys.path.insert(0, _V0_ROOT)
from path_setup import (  # noqa: E402
    LEGACY_ROOT,
    init_swanlab,
    legacy_output,
    log_swanlab,
    repo_dataset,
    setup_import_paths,
)

setup_import_paths()

import argparse
import copy
import math
import time
import warnings

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset
from model.model_big_strong import BigStrongConfig, BigStrongForCausalLLM

warnings.filterwarnings("ignore")


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def sequence_logps(model, input_ids, labels, loss_mask):
    """对 loss_mask=1 的位置累加 log π(label | prefix)。"""
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1)


def dpo_loss(
    policy_model,
    ref_model,
    chosen_x,
    chosen_y,
    chosen_mask,
    rejected_x,
    rejected_y,
    rejected_mask,
    beta: float,
):
    pi_c = sequence_logps(policy_model, chosen_x, chosen_y, chosen_mask)
    pi_r = sequence_logps(policy_model, rejected_x, rejected_y, rejected_mask)
    with torch.no_grad():
        ref_c = sequence_logps(ref_model, chosen_x, chosen_y, chosen_mask)
        ref_r = sequence_logps(ref_model, rejected_x, rejected_y, rejected_mask)
    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    loss = -F.logsigmoid(logits).mean()
    reward_acc = (logits > 0).float().mean()
    return loss, reward_acc


def train_epoch(epoch, policy_model, ref_model, optimizer, scaler, ctx):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        c_x, c_y, c_m, r_x, r_y, r_m = [t.to(args.device) for t in batch]

        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            loss, reward_acc = dpo_loss(
                policy_model,
                ref_model,
                c_x,
                c_y,
                c_m,
                r_x,
                r_y,
                r_m,
                args.beta,
            )
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_steps = epoch * iter_per_epoch + step
            total_steps = args.epochs * iter_per_epoch
            remaining_steps = total_steps - current_steps
            avg_time_per_step = spend_time / (current_steps + 1)
            remaining_time = remaining_steps * avg_time_per_step
            spend_time_formatted = (
                f"{int(spend_time // 3600):02d}:{int((spend_time % 3600) // 60):02d}:{int(spend_time % 60):02d}"
            )
            remaining_time_formatted = (
                f"{int(remaining_time // 3600):02d}:{int((remaining_time % 3600) // 60):02d}:{int(remaining_time % 60):02d}"
            )
            logger.debug(
                "Epoch:[{}/{}]({}/{}) loss:{:.4f} acc:{:.3f} lr:{:.7f} spent:{} remain:{}".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    reward_acc.item(),
                    optimizer.param_groups[-1]["lr"],
                    spend_time_formatted,
                    remaining_time_formatted,
                )
            )
            log_swanlab(
                {
                    "loss": loss.item() * args.accumulation_steps,
                    "reward_acc": reward_acc.item(),
                    "lr": optimizer.param_groups[-1]["lr"],
                }
            )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(policy_model)

    save_checkpoint(policy_model)
    logger.debug(f"Epoch {epoch + 1} 结束，已保存 checkpoint")


def save_checkpoint(model):
    model.eval()
    ckp = f"{args.save_dir}/dpo_{lm_config.hidden_size}.pth"
    state_dict = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(state_dict, ckp)
    model.train()


def load_sft_weights(model, ckp_path):
    state_dict = torch.load(ckp_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    return model.to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigStrongGPT DPO")
    parser.add_argument(
        "--out_dir", type=str, default=legacy_output("dpo_output") + os.sep
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument(
        "--data_path", type=str, default=repo_dataset("dpo_merged.jsonl")
    )
    parser.add_argument(
        "--sft_ckpt",
        type=str,
        default=legacy_output("sft_output", "full_sft_512.pth"),
    )
    parser.add_argument("--max_samples", type=int, default=None, help="调试用，限制样本数")

    args = parser.parse_args()
    os.makedirs(legacy_output("dpo_output"), exist_ok=True)

    logger = logging.Logger("BigStrongGPT-DPO")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(legacy_output("dpo_output", "dpo.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    lm_config = BigStrongConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    init_swanlab(project="BigStrongGPT", experiment_name="dpo", config=args)

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda")
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    tokenizer = AutoTokenizer.from_pretrained(str(LEGACY_ROOT / "model"))
    policy_model = BigStrongForCausalLLM(lm_config)
    load_sft_weights(policy_model, args.sft_ckpt)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    logger.debug(
        "Policy 可训练参数量：{:.3f}M".format(
            sum(p.numel() for p in policy_model.parameters() if p.requires_grad) / 1e6
        )
    )

    train_ds = DPODataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        max_samples=args.max_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=device_type == "cuda",
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device_type == "cuda")
    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    iter_per_epoch = len(train_loader)

    policy_model.train()
    for epoch in range(args.epochs):
        train_epoch(epoch, policy_model, ref_model, optimizer, scaler, ctx)
