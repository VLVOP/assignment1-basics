import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_logs(log_path, output_dir):
    iters = []
    times = []
    losses = []
    val_iters = []
    val_losses = []

    print(f"Reading logs from {log_path}...")
    try:
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Check required keys
                    if 'iter' in entry and 'loss' in entry and 'time' in entry:
                        iters.append(entry['iter'])
                        times.append(entry['time'])
                        losses.append(entry['loss'])
                    
                    if 'val_loss' in entry:
                        val_iters.append(entry['iter'])
                        val_losses.append(entry['val_loss'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return

    if not iters:
        print("No valid log entries found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss vs Steps
    plt.figure(figsize=(10, 6))
    plt.plot(iters, losses, label='Train Loss', alpha=0.3)
    # Moving Average (Simple)
    window = 50
    if len(losses) > window:
        moving_avg = []
        for i in range(len(losses) - window + 1):
            chunk = losses[i:i+window]
            moving_avg.append(sum(chunk) / window)
        plt.plot(iters[window-1:], moving_avg, label=f'Train Loss (SMA {window})', color='blue')
    
    if val_iters:
        plt.plot(val_iters, val_losses, label='Val Loss', color='red', marker='o', linestyle='--')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Steps')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    out_path_steps = os.path.join(output_dir, 'loss_vs_steps.png')
    plt.savefig(out_path_steps)
    print(f"Saved plot to {out_path_steps}")
    plt.close()

    # Plot 2: Loss vs Wallclock Time
    plt.figure(figsize=(10, 6))
    plt.plot(times, losses, label='Train Loss', alpha=0.3)
    
    if val_iters:
        # Match val_iters to times based on step count
        val_times = []
        for vi in val_iters:
            if vi in iters:
                idx = iters.index(vi)
                val_times.append(times[idx])
            else:
                # Fallback: interpolate roughly if missing (unlikely if logged together)
                pass
        
        # Only plot if we found matching times
        if len(val_times) == len(val_losses):
            plt.plot(val_times, val_losses, label='Val Loss', color='red', marker='o', linestyle='--')

    plt.xlabel('Wallclock Time (seconds)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Time')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    out_path_time = os.path.join(output_dir, 'loss_vs_time.png')
    plt.savefig(out_path_time)
    print(f"Saved plot to {out_path_time}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log.jsonl file")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_logs(args.log_path, args.output_dir)
