import collections
import regex
import heapq
import sys
import os
from cs336_basics.train_bpe import train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

def debug_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    print(f"Total merges found: {len(merges)}")
    print(f"Total reference merges: {len(reference_merges)}")

    for i, (mine, ref) in enumerate(zip(merges, reference_merges)):
        if mine != ref:
            print(f"Mismatch at index {i}:")
            print(f"  Mine: {mine}")
            print(f"  Ref:  {ref}")
            # Print previous few for context
            if i > 0:
                print(f"  Prev Mine: {merges[i-1]}")
                print(f"  Prev Ref:  {reference_merges[i-1]}")
            break
        else:
            if i < 10:
               print(f"Match {i}: {mine}")

if __name__ == "__main__":
    debug_train_bpe()
