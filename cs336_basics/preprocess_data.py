import numpy as np
import os
import argparse
from cs336_basics.tokenizer import tokenizer

def process_and_save(input_path: str, output_path: str, tok: tokenizer, chunk_size: int = 1000000):
    """
    Read text line-by-line, tokenize, and save to a binary file as uint16.
    Uses buffering to minimize memory usage.
    """
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}")
        return

    # Check if we should delete existing file to start fresh
    if os.path.exists(output_path):
        print(f"Removing existing output file: {output_path}")
        os.remove(output_path)

    buffer = []
    total_tokens = 0
    
    print(f"Processing {input_path} -> {output_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "wb") as f_out:
        
        for line in f_in:
            # Tokenize the line
            ids = tok.encode(line)
            buffer.extend(ids)
            
            # If buffer is full, flush to disk
            if len(buffer) >= chunk_size:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(f_out)
                total_tokens += len(buffer)
                buffer = [] # Reset buffer
        
        # Flush remaining tokens
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += len(buffer)
            
    print(f"Done. Saved {total_tokens} tokens to {output_path}")

def main():
    # Determine the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    
    # Define paths
    tiny_vocab = os.path.join(data_dir, "TinyStoriesV2-GPT4-vocab.json")
    tiny_merges = os.path.join(data_dir, "TinyStoriesV2-GPT4-merges.txt")
    
    owt_vocab = os.path.join(data_dir, "owt_vocab.json")
    owt_merges = os.path.join(data_dir, "owt_merges.txt")
    
    # 1. Process TinyStories (if vocab exists)
    if os.path.exists(tiny_vocab) and os.path.exists(tiny_merges):
        print("\n--- Processing TinyStories ---")
        try:
            tok_tiny = tokenizer.from_file(tiny_vocab, tiny_merges, special_tokens=["<|endoftext|>"])
            
            process_and_save(
                os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt"),
                os.path.join(data_dir, "tiny_train.bin"),
                tok_tiny
            )
            process_and_save(
                os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt"),
                os.path.join(data_dir, "tiny_valid.bin"),
                tok_tiny
            )
        except Exception as e:
            print(f"Error processing TinyStories: {e}")
    else:
        print(f"\nSkipping TinyStories: Vocab/Merges not found at {tiny_vocab}")

    # 2. Process OpenWebText (if vocab exists)
    if os.path.exists(owt_vocab) and os.path.exists(owt_merges):
        print("\n--- Processing OpenWebText ---")
        try:
            tok_owt = tokenizer.from_file(owt_vocab, owt_merges, special_tokens=["<|endoftext|>"])
            
            process_and_save(
                os.path.join(data_dir, "owt_train.txt"),
                os.path.join(data_dir, "owt_train.bin"),
                tok_owt
            )
            process_and_save(
                os.path.join(data_dir, "owt_valid.txt"),
                os.path.join(data_dir, "owt_valid.bin"),
                tok_owt
            )
        except Exception as e:
            print(f"Error processing OWT: {e}")
    else:
        print(f"\nSkipping OpenWebText: Vocab/Merges not found at {owt_vocab}")

if __name__ == "__main__":
    main()
