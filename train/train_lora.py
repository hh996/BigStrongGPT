import logging
import os
import sys

__package__ = "trainer"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import swanlab
from dotenv import load_dotenv

from model.model_big_strong import BigStrongForCausalLLM, BigStrongConfig

import argparse
import time
import math
import warnings
import torch
from torch import optim, nn
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora

warnings.filterwarnings("ignore")


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 代码和full_sft「几乎」一致
def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 计算已用时间和预估剩余时间
            current_steps = epoch * iter_per_epoch + step
            total_steps = args.epochs * iter_per_epoch
            remaining_steps = total_steps - current_steps
            avg_time_per_step = spend_time / (current_steps + 1)
            remaining_time = remaining_steps * avg_time_per_step
            # 格式化时间显示
            spend_time_formatted = f"{int(spend_time // 3600):02d}:{int((spend_time % 3600) // 60):02d}:{int(spend_time % 60):02d}"
            remaining_time_formatted = f"{int(remaining_time // 3600):02d}:{int((remaining_time % 3600) // 60):02d}:{int(remaining_time % 60):02d}"

            # 打印训练进度信息
            logger.debug(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} time spent:{}; time remaining:{};".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]["lr"],
                    spend_time_formatted,
                    remaining_time_formatted,
                )
            )

            # 启用SwanLab，记录训练指标
            swanlab.log(
                {
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]["lr"],
                }
            )

        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f"{args.save_dir}/lora_{lm_config.hidden_size}.pth"

            os.makedirs(os.path.dirname(ckp), exist_ok=True)
            # 【区别1】只保存lora权重即可
            save_lora(model, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("../model/")
    model = BigStrongForCausalLLM(lm_config)
    ckp = f"../output/sft_output/full_sft_512.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    logger.debug(
        f"LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    return model.to(args.device), tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigStrongGPT SFT with LoRA")
    parser.add_argument("--out_dir", type=str, default="../output/lora_output/")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument(
        "--data_path", type=str, default="../dataset/lora_medical.jsonl"
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default="lora_medical",
        help="根据任务保存成lora_(英文/医学/心理...)",
    )
    args = parser.parse_args()

    # 日志模块
    logger = logging.Logger("BigStrongGPT")
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 控制台输出INFO及以上级别

    file_handler = logging.FileHandler("../output/lora_output/lora.log")
    file_handler.setLevel(logging.DEBUG)  # 文件保存DEBUG及以上级别

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 给日志器添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    lm_config = BigStrongConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # ==================== 实验跟踪初始化 ====================
    # 加载swanlab key
    load_dotenv()
    swanlab.login(api_key=os.getenv("SWANLAB_API_KEY"))
    run = swanlab.init(
        project="BigStrongGPT",  # 项目名称
        experiment_name="lora",  # 实验名称
        config=args,  # 保存所有超参数
    )

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    model, tokenizer = init_model(lm_config)
    apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    lora_params_count = sum(
        p.numel() for name, p in model.named_parameters() if "lora" in name
    )  # LoRA 参数数量

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_params.append(param)

    # 只对 LoRA 参数进行优化
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    iter_per_epoch = len(train_loader)

    for epoch in range(args.epochs):
        train_epoch(epoch)
