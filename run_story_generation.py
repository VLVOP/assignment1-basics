import torch
import os
import argparse
from cs336_basics.transformer_lm import transformerLM
from cs336_basics.tokenizer import tokenizer
from cs336_basics.generate import generate
from cs336_basics.checkpoint import load_checkpoint

# 默认路径
DEFAULT_VOCAB = "data/TinyStoriesV2-GPT4-vocab.json"
DEFAULT_MERGES = "data/TinyStoriesV2-GPT4-merges.txt"
DEFAULT_CKPT = "checkpoints/tiny_best_baseline/final_checkpoint.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to checkpoint")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB)
    parser.add_argument("--merges", type=str, default=DEFAULT_MERGES)
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily.")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.vocab}...")
    if not os.path.exists(args.vocab):
        # Fallback for common filenames
        if os.path.exists("tinystories_vocab.json"):
            args.vocab = "tinystories_vocab.json"
            args.merges = "tinystories_merges.txt"
            print(f"File not found, falling back to {args.vocab}")
        else:
            print("Error: Vocab file not found.")
            return

    enc = tokenizer.from_file(args.vocab, args.merges, special_tokens=["<|endoftext|>"])

    # 2. Setup Model (Params must match training!)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1344,
        device=device,
        dtype=torch.float32
    )

    # 3. Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    try:
        load_checkpoint(args.ckpt, model, optimizer=None)
    except Exception as e:
        print(f"Warning: load_checkpoint failed ({e}). Trying raw state_dict load...")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    model.to(device)
    model.eval()

    # 4. Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens...")
    
    input_ids = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    output_ids = generate(
        model, 
        input_ids, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temp
    )
    
    output_text = enc.decode(output_ids[0].tolist())
    
    print("\n" + "="*40)
    print("GENERATED STORY:")
    print("="*40)
    print(output_text)
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
