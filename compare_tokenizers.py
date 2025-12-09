import sys
import os
import argparse
import json

# Ensure the package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.tokenizer import tokenizer

def load_text(filepath):
    """Reads text from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

def calculate_compression_ratio(text_bytes, tokens):
    """Calculates bytes per token."""
    if not tokens:
        return 0.0
    return len(text_bytes) / len(tokens)

def main():
    parser = argparse.ArgumentParser(description="Compare Tokenizers on a text sample.")
    
    # Arguments for TinyStories Tokenizer
    parser.add_argument("--ts_vocab", type=str, required=True, help="Path to TinyStories vocab.json")
    parser.add_argument("--ts_merges", type=str, required=True, help="Path to TinyStories merges.txt")
    
    # Arguments for OpenWebText Tokenizer (Optional for part b, but good for comparison)
    parser.add_argument("--owt_vocab", type=str, default=None, help="Path to OpenWebText vocab.json")
    parser.add_argument("--owt_merges", type=str, default=None, help="Path to OpenWebText merges.txt")
    
    # Argument for the text file to tokenize
    parser.add_argument("--text_file", type=str, required=True, help="Path to the text file to tokenize (e.g., OpenWebText sample)")

    args = parser.parse_args()

    # --- 1. Load Text ---
    print(f"Loading text from: {args.text_file}")
    text_content = load_text(args.text_file)
    text_bytes = text_content.encode("utf-8")
    print(f"Original Text Length: {len(text_bytes)} bytes")

    # --- 2. Load TinyStories Tokenizer ---
    print("\nLoading TinyStories Tokenizer...")
    if not os.path.exists(args.ts_vocab) or not os.path.exists(args.ts_merges):
        print(f"Error: TinyStories tokenizer files not found.")
        sys.exit(1)
        
    ts_tokenizer = tokenizer.from_file(args.ts_vocab, args.ts_merges, special_tokens=["<|endoftext|>"])

    # --- 3. Tokenize with TinyStories Tokenizer ---
    print(f"Tokenizing with TinyStories Tokenizer...")
    ts_ids = ts_tokenizer.encode(text_content)
    ts_ratio = calculate_compression_ratio(text_bytes, ts_ids)
    
    print(f"Token Count (TS): {len(ts_ids)}")
    print(f"Compression Ratio (TS): {ts_ratio:.4f} bytes/token")

    # --- 4. Qualitative Analysis (Preview) ---
    print("\n--- Qualitative Analysis Sample ---")
    preview_len = min(20, len(ts_ids))
    decoded_tokens = [ts_tokenizer.decode([tid]) for tid in ts_ids[:preview_len]]
    print(f"First {preview_len} tokens (TS): {decoded_tokens}")

    # --- 5. Compare with OWT Tokenizer (if provided) ---
    if args.owt_vocab and args.owt_merges:
        print("\nLoading OpenWebText Tokenizer...")
        if os.path.exists(args.owt_vocab) and os.path.exists(args.owt_merges):
            owt_tokenizer = tokenizer.from_file(args.owt_vocab, args.owt_merges, special_tokens=["<|endoftext|>"])
            
            print(f"Tokenizing with OpenWebText Tokenizer...")
            owt_ids = owt_tokenizer.encode(text_content)
            owt_ratio = calculate_compression_ratio(text_bytes, owt_ids)
            
            print(f"Token Count (OWT): {len(owt_ids)}")
            print(f"Compression Ratio (OWT): {owt_ratio:.4f} bytes/token")
            
            print(f"\nComparison:")
            print(f"TS Ratio: {ts_ratio:.4f} vs OWT Ratio: {owt_ratio:.4f}")
            if ts_ratio < owt_ratio:
                print("TinyStories tokenizer has a WORSE (lower) compression ratio.")
            else:
                print("TinyStories tokenizer has a BETTER (higher) compression ratio.")
        else:
            print(f"Warning: OpenWebText tokenizer files provided but not found.")

if __name__ == "__main__":
    main()