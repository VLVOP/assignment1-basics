import subprocess
import sys
import os

TRAIN_SCRIPT = "cs336_basics/llmTrain.py"

# User provided paths
TRAIN_DATA = "data/tiny_train.bin"
VAL_DATA = "data/tiny_val.bin"

# Architecture and fixed hyperparameters
ARCH_ARGS = [
    "--vocab_size", "10000",
    "--context_length", "256",
    "--d_model", "512",
    "--d_ff", "1344",
    "--num_layers", "4",
    "--num_heads", "16",
    "--max_grad_norm", "1.0",
]

# Learning rates to test for divergence
# 0.005 was optimal. We go higher to find the breaking point.
learning_rates = [0.01, 0.02, 0.05, 0.1]

BATCH_SIZE = 512
MAX_ITERS = 500
WARMUP_ITERS = 50
EVAL_INTERVAL = 50  # Frequent evaluation to catch the divergence moment

for lr in learning_rates:

    run_name = f"divergence_test_lr{lr}"
    checkpoint_dir = f"checkpoints/{run_name}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*40}")
    print(f"Starting Experiment: {run_name}")
    print(f"Testing High LR: {lr} to observe divergence behavior")
    print(f"{'='*40}\n")

    cmd = [
        "uv", "run", "python", TRAIN_SCRIPT,
        "--train_data_path", TRAIN_DATA,
        "--val_data_path", VAL_DATA,
        "--checkpoint_dir", checkpoint_dir,

        "--lr", str(lr),
        "--batch_size", str(BATCH_SIZE),
        "--max_iters", str(MAX_ITERS),
        "--warmup_iters", str(WARMUP_ITERS),
        "--min_lr", "0.00001",

        *ARCH_ARGS,

        "--weight_decay", "0.01",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--eps", "1e-8",

        "--eval_interval", str(EVAL_INTERVAL),
        "--save_interval", "2000", # No need to save checkpoints for crash tests
        "--device", "cuda"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Experiment {run_name} failed! (Likely diverged with NaN loss). This is expected.")
    except KeyboardInterrupt:
        print("\nSweep interrupted by user.")
        break

print("\nDivergence search completed!")
