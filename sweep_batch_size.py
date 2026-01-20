import subprocess
import sys
import os

TRAIN_SCRIPT = "cs336_basics/llmTrain.py"

# 数据路径
TRAIN_DATA = "data/tiny_train.bin"
VAL_DATA = "data/tiny_valid.bin" # 注意：之前的脚本写的是 tiny_val.bin，根据你之前的参数改为 tiny_valid.bin

# 固定的架构参数 (保持不变以控制变量)
ARCH_ARGS = [
    "--vocab_size", "10000",
    "--context_length", "256",
    "--d_model", "512",
    "--d_ff", "1344",
    "--num_layers", "4",
    "--num_heads", "16",
    "--max_grad_norm", "1.0",
]

# 基准配置
BASE_BATCH_SIZE = 512
BASE_LR = 0.005

# 我们要测试的 Batch Sizes
batch_sizes_to_test = [64, 128, 256, 1024] 
# 注意：512 已经跑过了

# 通用设置
MAX_ITERS = 500
WARMUP_ITERS = 50 
EVAL_INTERVAL = 100

for bs in batch_sizes_to_test:
    
    # === 应用线性缩放法则 ===
    scaled_lr = BASE_LR * (bs / BASE_BATCH_SIZE)
    scaled_lr = float(f"{scaled_lr:.6f}")

    run_name = f"batch_sweep_bs{bs}_lr{scaled_lr}"
    checkpoint_dir = f"checkpoints/{run_name}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"Starting Experiment: {run_name}")
    print(f"Batch Size: {bs} | Scaled LR: {scaled_lr} (Linear Scaling Rule)")
    print("="*60 + "\n")

    cmd = [
        "uv", "run", "python", TRAIN_SCRIPT,
        "--train_data_path", TRAIN_DATA,
        "--val_data_path", VAL_DATA,
        "--checkpoint_dir", checkpoint_dir,

        "--lr", str(scaled_lr),
        "--batch_size", str(bs),
        "--max_iters", str(MAX_ITERS),
        "--warmup_iters", str(WARMUP_ITERS),
        "--min_lr", str(scaled_lr / 10),

        *ARCH_ARGS,

        "--weight_decay", "0.01",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--eps", "1e-8",

        "--eval_interval", str(EVAL_INTERVAL),
        "--save_interval", "2000",
        "--device", "cuda"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Experiment {run_name} failed! Moving to next...")
    except KeyboardInterrupt:
        print("\nSweep interrupted by user.")
        break

print("\nBatch size sweep completed!")