import logging
import os
import sys

__package__ = "trainer"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import swanlab
import os
from dotenv import load_dotenv

import argparse
import time
import math
import warnings
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_big_strong import BigStrongConfig, BigStrongForCausalLLM
from dataset.lm_dataset import PretrainDataset


warnings.filterwarnings("ignore")


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


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
        
        # 放大损失值以避免低精度数值下溢
        scaler.scale(loss).backward()

        # 每accumulation_steps步执行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 放大的梯度恢复到原始比例
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 每log_interval步记录一次日志
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

        # 每save_interval步保存一次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}.pth"
            state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("../model/")
    model = BigStrongForCausalLLM(lm_config).to(args.device)
    logger.debug(
        f"LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigStrongGPT Pretrain")
    parser.add_argument("--out_dir", type=str, default="../output/pretrain_output/")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    # 日志模块
    logger = logging.Logger("BigStrongGPT")
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 控制台输出INFO及以上级别

    file_handler = logging.FileHandler("../output/pretrain_output/pretrain.log")
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

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda")

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # ==================== 实验跟踪初始化 ====================
    # 加载swanlab key
    load_dotenv()
    swanlab.login(api_key=os.getenv("SWANLAB_API_KEY"))
    run = swanlab.init(
        project="BigStrongGPT",  # 项目名称
        experiment_name="Pretrain",  # 实验名称
        config=args,  # 保存所有超参数
    )

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
