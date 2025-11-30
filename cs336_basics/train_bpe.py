import collections
import regex

GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" 

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_counts = collections.Counter()

    escaped_toekns = [regex.escape(token) for token in special_tokens]

    joined_pattern = "|".join(escaped_toekns)

    final_pattern = "(" + joined_pattern + ")"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:

            if not special_tokens:
                parts = [line]
            else:
                parts = regex.split(final_pattern, line)

            for part in parts:
                if not part:
                    continue

                if special_tokens and part in special_tokens:
                    continue

                sub_parts = regex.findall(GPT2_SPLIT_PATTERN, part)
                for word_str in sub_parts:
                    word_bytes = word_str.encode('utf-8')
                    word_tuple = tuple([bytes([b]) for b in word_bytes])
                    word_counts[word_tuple] += 1
                    
            
    vocab : dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')

    # 创建反向映射
    byte2id = {v : k for k, v in vocab.items()}

    merges = []
    while len(vocab) < vocab_size:
        # 得到初始对列表
        pairs = collections.Counter()

        for ward_tuple, count in word_counts.items():
            for i in range(len(ward_tuple) - 1):
                pair = (ward_tuple[i], ward_tuple[i + 1])
                pairs[pair] += count

        if not pairs:
            break
        
        best_pair = max(pairs, key=lambda p : (pairs[p], p))

        # 创建新符号
        new_symbol = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_symbol

        byte2id[new_symbol] = len(vocab) - 1

        merges.append(best_pair)

        # 更新word_counts
        new_word_counts = collections.Counter()
        for word_tuple, count in word_counts.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_counts[tuple(new_word)] += count
        word_counts = new_word_counts

    return vocab, merges