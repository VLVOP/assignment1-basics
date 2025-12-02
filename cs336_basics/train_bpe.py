import collections
import regex
import heapq

GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" 

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_counts = collections.Counter()

    # escaped_toekns = [regex.escape(token) for token in special_tokens]

    # joined_pattern = "|".join(escaped_toekns)

    # final_pattern = "(" + joined_pattern + ")"

    delimiter = special_tokens[0] if special_tokens else None

    gpt2_pat = regex.compile(GPT2_SPLIT_PATTERN)

    if special_tokens:
        escaped_tokens = [regex.escape(token) for token in special_tokens]
        joined_pattern = "|".join(escaped_tokens)
        final_pattern = "(" + joined_pattern + ")"
        special_pat = regex.compile(final_pattern)
    else:
        special_pat = None


    for text_chunk in read_chunks(input_path, separator=delimiter):
        if special_tokens:
            parts = special_pat.split(text_chunk) if special_tokens else [text_chunk]
        else:
            parts = [text_chunk]
        for part in parts:
            if not part:
                continue
            
            if special_tokens and part in special_tokens:
                continue

            sub_parts = gpt2_pat.findall(part)
            for word_str in sub_parts:
                word_bytes = word_str.encode('utf-8')
                word_tuple = tuple([bytes([b]) for b in word_bytes])
                word_counts[word_tuple] += 1
    
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     if special_tokens:
            
        # for line in f:

        #     if not special_tokens:
        #         parts = [line]
        #     else:
        #         parts = regex.split(final_pattern, line)

        #     for part in parts:
        #         if not part:
        #             continue

        #         if special_tokens and part in special_tokens:
        #             continue

        #         sub_parts = regex.findall(GPT2_SPLIT_PATTERN, part)
        #         for word_str in sub_parts:
        #             word_bytes = word_str.encode('utf-8')
        #             word_tuple = tuple([bytes([b]) for b in word_bytes])
        #             word_counts[word_tuple] += 1
                    
            
    vocab : dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')

    # 创建反向映射
    byte2id = {v : k for k, v in vocab.items()}

    merges = []
    # while len(vocab) < vocab_size:
    #     # 得到初始对列表
    #     pairs = collections.Counter()

    #     for ward_tuple, count in word_counts.items():
    #         for i in range(len(ward_tuple) - 1):
    #             pair = (ward_tuple[i], ward_tuple[i + 1])
    #             pairs[pair] += count

    #     if not pairs:
    #         break
        
    #     best_pair = max(pairs, key=lambda p : (pairs[p], p))

    #     # 创建新符号
    #     new_symbol = best_pair[0] + best_pair[1]

    #     vocab[len(vocab)] = new_symbol

    #     byte2id[new_symbol] = len(vocab) - 1

    #     merges.append(best_pair)

    #     # 更新word_counts
    #     new_word_counts = collections.Counter()
    #     for word_tuple, count in word_counts.items():
    #         new_word = []
    #         i = 0
    #         while i < len(word_tuple):
    #             if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
    #                 new_word.append(new_symbol)
    #                 i += 2
    #             else:
    #                 new_word.append(word_tuple[i])
    #                 i += 1
    #         new_word_counts[tuple(new_word)] += count
    #     word_counts = new_word_counts

    return vocab, merges

def read_chunks(file_path, chunk_size=1024*1024, separator="<|endoftext|>"):
    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            while True:
                try:
                    if separator is None:
                        yield buffer
                        buffer = ""
                        break

                    part, buffer = buffer.split(separator, 1)

                    yield part
                except ValueError:
                    break

        if buffer:
            yield buffer