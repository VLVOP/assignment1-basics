import sys
import os

# Add current directory to sys.path so we can import cs336_basics
sys.path.append(os.getcwd())

from cs336_basics.tokenizer import tokenizer

# Mock vocab and merges
# Let's assume 0 -> b'\xc3', 1 -> b'\xa9', 2 -> b'a'
vocab = {
    0: b'\xc3',
    1: b'\xa9',
    2: b'a'
}
merges = []

t = tokenizer(vocab, merges)

# Test decoding a single partial byte
try:
    print(f"Decoding [0] (b'\xc3'): {t.decode([0])}")
except UnicodeDecodeError as e:
    print(f"Caught expected error: {e}")

# Test decoding full sequence
print(f"Decoding [0, 1] (b'\xc3\xa9'): {t.decode([0, 1])}")

# Proposed fix verification
print("\nWith errors='replace':")
try:
    byte_seq = b"".join(vocab[i] for i in [0])
    print(f"Decoding [0] with replace: {byte_seq.decode('utf-8', errors='replace')}")
except Exception as e:
    print(f"Error: {e}")

