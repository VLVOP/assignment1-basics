import subprocess
import sys

TRAIN_SCRIPT = "cs336_basics/llmTrain.py"

TRAIN_DATA = "data/tiny_train.bin"
VAL_DATA = "data/tiny_val.bin"

ARCH_ARGS = [
    "--vocab_size", "10000",
    "--context_length", "256",
    "--d_model", "512",
    "--d_ff", "1344",
    "--num_layers", "4",
    "--num_heads", "16",
]


learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

BATCH_SIZE = 512
MAX_ITERS = 500
EVAL_INTERVAL = 100

WARMUP_ITERS = 50

for lr in learning_rates:

    run_name = f"sweep_bs{BATCH_SIZE}_lr{lr}"
    checkpoint_dir = f"checkpoints/{run_name}"

    print(f"\n{'='*40}")
    print(f"Starting Experiment: {run_name}")
    print(f"lR: {lr} | Batch: {BATCH_SIZE} | Device: Expecting CUDA (A800)")
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
        print(f"Experiment {run_name} failed! Moving to next ...")
    
    except FileNotFoundError:
        print(f"Error: 'uv' command not found. Make sure uv is installed and in your PATH.")

    except KeyboardInterrupt:
        print("\n Sweep interrupted by user.")
        exit(0)

print("\n Sweep completed!")