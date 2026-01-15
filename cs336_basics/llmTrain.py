import argparse
import os
import torch
import numpy as np
import wandb

from cs336_basics.cross_entropy import CEloss
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.lr_cosine_schedule import lr_cosine_schedule
from cs336_basics.transformer_lm import transformerLM
from cs336_basics.AdamW import AdamWoptimizer
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

def train(args):
    print(f"开始训练！ 使用设备：{args.device}")
    model = transformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        device=args.device,
        dtype=torch.float32
    )
    optimizer = AdamWoptimizer(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    start_iter = 0

    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            print(f"从检查点恢复训练：{args.resume_from}")
            start_iter = load_checkpoint(args.resume_from, model, optimizer)
            print(f"恢复到迭代 {start_iter}")
        else:
            print(f"检查点路径不存在：{args.resume_from}，从头开始训练。")

    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode='r')

    wandb.init(project="cs336-assignment1", config=args)

    model.train()

    for i in range(start_iter, args.max_iters):

        lr = lr_cosine_schedule(
            t=i,
            alpha_max=args.lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.max_iters
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr.item()

        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )


        logits = model(x)

        loss = CEloss(y, logits)

        optimizer.zero_grad()
        loss.backward()

        clip_gradients(
            list(model.parameters()),
            max_norm=args.max_grad_norm
        )

        optimizer.step()

        log_dict = {"loss": loss.item(), "lr": lr.item(), "iter": i}

        if i % args.eval_interval == 0 or i == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(args.eval_iters)
                for k in range(args.eval_iters):
                    x_val, y_val = get_batch(
                        val_data,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                        device=args.device
                    )
                    logits_val = model(x_val)
                    val_loss = CEloss(y_val, logits_val)
                    losses[k] = val_loss.item()
            model.train()
            val_loss = losses.mean()
            log_dict["val_loss"] = val_loss
            print(f"Step {i} Val Loss: {val_loss}")

        wandb.log(log_dict)

        if i % args.save_interval == 0 and i > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_iter_{i}.pt")
            save_checkpoint(model, optimizer, i, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    final_ckpt_path = os.path.join(args.checkpoint_dir, f"final_checkpoint.pt")
    save_checkpoint(model, optimizer, i, final_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Assignment 1: Transformer LM Training")

    parser.add_argument("--train_data_path", type=str, required=True, help="训练数据的路径（numpy memmap 格式）")
    parser.add_argument("--val_data_path", type=str, required=True, help="验证数据的路径（numpy memmap 格式）")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="保存检查点的目录")
    parser.add_argument("--resume_from", type=str, default=None, help="从检查点恢复训练的路径")

    parser.add_argument("--vocab_size", type=int, default=50257, help="词汇表大小")
    parser.add_argument("--context_length", type=int, default=1024, help="上下文长度")
    parser.add_argument("--num_layers", type=int, default=12, help="Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=12, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--d_ff", type=int, default=3072, help="前馈网络维度")

    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="学习率预热迭代次数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪的最大范数")

    parser.add_argument("--max_iters", type=int, default=100000, help="最大训练迭代次数")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="学习率预热的迭代次数")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--eval_interval", type=int, default=1000, help="评估间隔（迭代次数）")
    parser.add_argument("--eval_iters", type=int, default=10, help="评估时的迭代次数")
    parser.add_argument("--save_interval", type=int, default=5000, help="保存检查点的间隔（迭代次数）")

    args = parser.parse_args()

    print("Training configurations:")
    for k, v in vars(args).items():
        print(f" {k}: {v}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)
