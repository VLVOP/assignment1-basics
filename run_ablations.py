import subprocess
import sys
import math

BASE_ARGS = {
    "train_data_path": "data/tiny_train.bin",
    "val_data_path": "data/tiny_valid.bin",
    "vocab_size": "10000",
    "context_length": "256",
    "d_model": "512",
    "num_layers": "4",
    "num_heads": "16",
    "d_ff": "1344",
    "max_iters": "5000",
    "eval_interval": "500",
    "save_interval": "5000",
    "batch_size": "512",
    "lr": "5e-3",
    "warmup_iters": "100",
    "rope_theta": "10000.0",
}


experiments = {
    {
        "name": "baseline_swiglu_prenorm",
        "description": "Baseline: Pre-Norm, RoPE, SwiGLU",
        "args": {}
    },
    {
        "name": "ablation_no_rmsnorm",
        "description": "Ablation: Remove RMSNorm ",
        "args": {"no_rmsnorm": True}
    },
    {
        "name": "ablation_post_norm",
        "description": "Ablation: Remove RMSNorm",
        "args": {"norm_type": "post"}
    },
    {
        "name": "ablation_no_rope",
        "description": "Ablation: Remove RoPE",
        "args": {"no_rope": True}
    },
    {
        "name": "ablation_silu_ffn",
        "description": "Ablation: SiLU FFN (Parameter matched)",
        "args": {
            "ffn_type": "silu",
            "d_ff": "2016"
        }
    }
}

TRAIN_SCRIPT = "cs336_basics/llmTrain.py"

def run_experiment(exp_config):
    run_name = exp_config["name"]
    print(f"\n{'='*60}")
    print(f"Running Experiment: {run_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*60}\n")

    cmd = [
        "uv", "run", "python", TRAIN_SCRIPT,
        "--checkpoint_dir", f"checkpoints/{run_name}",
    ]

    for k, v in BASE_ARGS.items():
        if k not in exp_config["args"]:
            cmd.extend([f"--{k}", str(v)])

    for k, v in exp_config["args"].items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        
        else:
            cmd.extend([f"--{k}", str(v)])

    print("Executing command:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] Experiment {run_name} completed.")
    except subprocess.CalledProcessError:
        print(f"\n[FAILURE] Experiment {run_name} failed.")
    except KeyboardInterrupt:
        print(f"\n[STOP] Ablation experiments interrupted by user.")
        sys.exit(0)

def main():
    print ("Starting Ablation Experiments...")

    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: 'uv' command not found or not working. Please install and configure 'uv' (e.g., via Univention).")
        sys.exit(1)

    for exp in experiments:
        run_experiment(exp)

    print ("\nAll Ablation Experiments Completed.")

if __name__ == "__main__":
    main()