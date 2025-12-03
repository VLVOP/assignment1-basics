import collections
import heapq
import regex

GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class _RevPair:
    """Wrapper so heapq breaks ties by *larger* pairs, matching reference merges."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "._RevPair") -> bool:  # type: ignore[name-defined]
        return self.pair > other.pair


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_counts = collections.Counter()
    delimiter = special_tokens[0] if special_tokens else None
    gpt2_pat = regex.compile(GPT2_SPLIT_PATTERN)

    if special_tokens:
        escaped_tokens = [regex.escape(token) for token in special_tokens]
        special_pat = regex.compile("(" + "|".join(escaped_tokens) + ")")
    else:
        special_pat = None

    for text_chunk in read_chunks(input_path, separator=delimiter):
        parts = special_pat.split(text_chunk) if special_pat else [text_chunk]
        for part in parts:
            if not part or (special_tokens and part in special_tokens):
                continue
            for word_str in gpt2_pat.findall(part):
                word_bytes = word_str.encode("utf-8")
                word_tuple = tuple(bytes([b]) for b in word_bytes)
                word_counts[word_tuple] += 1

    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    words_list: list[list[bytes]] = []
    counts_list: list[int] = []
    for word_tuple, count in word_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(count)

    stats: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    indices: dict[tuple[bytes, bytes], set[int]] = collections.defaultdict(set)

    for idx, word in enumerate(words_list):
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            stats[pair] += counts_list[idx]
            indices[pair].add(idx)

    heap: list[tuple[int, _RevPair]] = [(-freq, _RevPair(pair)) for pair, freq in stats.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size:
        if not heap:
            break
        freq, pair_obj = heapq.heappop(heap)
        pair = pair_obj.pair
        count = -freq

        # Lazy deletion check
        if stats.get(pair, 0) != count:
            continue
        if count < 1:
            break

        new_symbol = pair[0] + pair[1]
        vocab[len(vocab)] = new_symbol
        merges.append(pair)

        changes: set[tuple[bytes, bytes]] = set()
        affected_indices = list(indices[pair])

        for word_idx in affected_indices:
            word = words_list[word_idx]
            w_count = counts_list[word_idx]

            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    # update left neighbor stats
                    if new_word:
                        prev_token = new_word[-1]
                        stats[(prev_token, pair[0])] -= w_count
                        stats[(prev_token, new_symbol)] += w_count
                        changes.add((prev_token, pair[0]))
                        changes.add((prev_token, new_symbol))
                        indices[(prev_token, new_symbol)].add(word_idx)

                    # update right neighbor stats
                    if i + 2 < len(word):
                        next_token = word[i + 2]
                        stats[(pair[1], next_token)] -= w_count
                        stats[(new_symbol, next_token)] += w_count
                        changes.add((pair[1], next_token))
                        changes.add((new_symbol, next_token))
                        indices[(new_symbol, next_token)].add(word_idx)

                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            words_list[word_idx] = new_word

        for p in changes:
            if stats[p] > 0:
                heapq.heappush(heap, (-stats[p], _RevPair(p)))

        stats.pop(pair, None)
        indices.pop(pair, None)

    return vocab, merges


def read_chunks(file_path, chunk_size=1024 * 1024, separator="<|endoftext|>"):
    with open(file_path, "r", encoding="utf-8") as f:
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
