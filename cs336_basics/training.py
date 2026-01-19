import json
import random
import os
from cs336_basics.train_bpe import train_bpe, read_chunks
from cs336_basics.tokenizer import tokenizer
from cs336_basics.bpe_utils import bytes_to_unicode_str

def save_vocab(vocab: dict[int, bytes], filepath: str):

    # Save as token_str -> id, using standard GPT-2 mapping
    vocab_to_save = {
        bytes_to_unicode_str(token_bytes): idx
        for idx, token_bytes in vocab.items()
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_to_save, f, indent=2, ensure_ascii=False)

def save_merges(merges: list[tuple[bytes, bytes]], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        for b1, b2 in merges:
            s1 = bytes_to_unicode_str(b1)
            s2 = bytes_to_unicode_str(b2)
            f.write(f"{s1} {s2}\n")

def ReservoirSample(stream, k):

    reservoir = []

    iterator = iter(stream)

    for _ in range(k):
        try:
            item = next(iterator)
            reservoir.append(item)
        except StopIteration:
            return reservoir
        
    for i, item in enumerate(iterator):

        m = k + i + 1
        j = random.randint(0, m - 1)

        if j < k:
            reservoir[j] = item
    
    return reservoir

def get_sample_docs(file_path: str, num_docs: int = 10, separator: str = "<|endoftext|>") -> list[str]:
    """
    從指定文件中采樣n個非空文檔。

    Args:
        file_path (str): 文檔文件的路徑。
        num_docs (int): 需要采樣的文檔數量。
        separator (str): 用於分隔文檔的標記。
    
    Returns:
        list[str]: 采樣到的文檔列表。
    """
    docs = []

    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    
    iterator = read_chunks(file_path, separator=separator)

    filtered_iterator = (doc for doc in iterator if doc.strip())

    return ReservoirSample(filtered_iterator, num_docs)

def evaluate_tokenizer(vocab_path, merges_path, sample_docs):

    tok = tokenizer.from_file(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    total_bytes = 0
    total_tokens = 0

    for doc in sample_docs:

        doc_bytes = doc.encode("utf-8")
        total_bytes += len(doc_bytes)

        ids = tok.encode(doc)
        total_tokens += len(ids)

    if total_tokens == 0:
        return 0.0
    
    return total_bytes / total_tokens

if __name__ == "__main__":
    
    print("This is the training module.")

    # Determine the project root directory (one level up from this script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Define absolute paths for data and output files
    data_dir = os.path.join(project_root, "data")
    
    owt_input_path = os.path.join(data_dir, "owt_train.txt")
    owt_vocab_path = os.path.join(data_dir, "owt_vocab.json")
    owt_merges_path = os.path.join(data_dir, "owt_merges.txt")
    
    tiny_input_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    tiny_vocab_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-vocab.json")
    tiny_merges_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-merges.txt")

    # print(f"Training OWT tokenizer from {owt_input_path}...")
    # owt_vocab, owt_merges = train_bpe(
    #     input_path = owt_input_path,
    #     vocab_size = 32000,
    #     special_tokens = ["<|endoftext|>"]
    # )

    # save_vocab(owt_vocab, owt_vocab_path)
    # save_merges(owt_merges, owt_merges_path)

    print(f"Training TinyStories tokenizer from {tiny_input_path}...")
    Tiny_vocab, Tiny_merges = train_bpe(
        input_path = tiny_input_path,
        vocab_size = 10000,
        special_tokens=["<|endoftext|>"]
    )

    save_vocab(Tiny_vocab, tiny_vocab_path)
    save_merges(Tiny_merges, tiny_merges_path)

    print("Sampling documents for evaluation...")
    owt_docs = get_sample_docs(owt_input_path, num_docs=10)
    Tiny_docs = get_sample_docs(tiny_input_path, num_docs=10)

    print("Evaluating OWT tokenizer...")
    if os.path.exists(owt_vocab_path) and os.path.exists(owt_merges_path):
        ratio_owt = evaluate_tokenizer(owt_vocab_path, owt_merges_path, owt_docs)
        print(f"OWT Tokenizer Bytes per Token: {ratio_owt:.4f}")
    else:
        print("OWT Tokenizer files not found, skipping evaluation.")
    
    print("Evaluating TinyStories tokenizer...")
    ratio_Tiny = evaluate_tokenizer(tiny_vocab_path, tiny_merges_path, Tiny_docs)

    print(f"TinyStoriesV2-GPT4 Tokenizer Bytes per Token: {ratio_Tiny:.4f}")


    

