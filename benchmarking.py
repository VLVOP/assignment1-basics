import time
import os

from cs336_basics.tokenizer import tokenizer

VOCAB_PATH = "tests/fixtures/train-bpe-reference-vocab.json"
MERGES_PATH = "tests/fixtures/train-bpe-reference-merges.txt"

if not os.path.exists(VOCAB_PATH) or not os.path.exists(MERGES_PATH):
    print(f"Error: Vocab or merges file not found.")
    if os.path.exists("tinystories_vocab.json"):
        VOCAB_PATH = "tinystories_vocab.json"
        MERGES_PATH = "tinystories_merges.txt"
        print(f"Using local file: {VOCAB_PATH}")

print("Loading tokenizer...")

enc = tokenizer.from_file(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])



with open("tests/fixtures/tinystories_sample_5M.txt", "r", encoding="utf-8") as f:
    text = f.read()
start = time.time()
enc.encode(text)
end = time.time()

print(f"吞吐量: {len(text.encode('utf-8')) / (end - start) / 1024:.2f} KB/s") 

throughput = len(text.encode('utf-8')) / (end - start) 
total_bytes_pile = 825 * 1024**3

print(f"預計處理 Pile 數據集所需時間: {total_bytes_pile / throughput :.4f} 秒")