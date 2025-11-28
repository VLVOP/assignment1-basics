import heapq
import json
import regex

from typing import Iterable, Iterator

GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.byte2id = {v : k for k, v in vocab.items()}
        self.merge_rank = {pair : i for i, pair in enumerate(merges)}
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

    @classmethod
    def from_file(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = cls._load_vocab(vocab_filepath)

        merges = cls._load_merges(merges_filepath)

        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def _load_vocab(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        vocab = {}

        for idx_str, byte_list in vocab_dict.items():
            vocab[int(idx_str)] = bytes(byte_list)
        
        return vocab
    
    @staticmethod
    def _load_merges(filepath):
        merges = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line :
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    byte1 = parts[0].encode("utf-8")
                    byte2 = parts[1].encode("utf-8")
                    merges.append((byte1, byte2))

        return merges

    def encode(self, text: str) -> list[int]:

        if not text:
            return []
        
        def split_specials(s : str):
            specials = self.special_tokens_sorted
            normal = []
            i = 0
            while i < len(s):
                for tok in specials:
                    if s.startswith(tok, i):
                        if normal:
                            yield False, "".join(normal)
                            normal = []
                        yield True, tok
                        i += len(tok)
                        break
                else:
                    normal.append(s[i])
                    i += 1
            if normal:
                yield False, "".join(normal)
        
        def bpe_merge(chunk: bytes) -> list[bytes]:

            if not chunk:
                return []
            
            tokens = [bytes([blist]) for blist in chunk]
            n = len(tokens)
            prev = [i - 1 for i in range(n)]
            nxt = [i + 1 for i in range(n)]
            nxt[-1] = -1
            alive = [True] * n

            heap: list[tuple[int, int]] = []
            for i in range(n - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None:
                    heapq.heappush(heap, (rank, i))

            while heap:
                rank, i = heapq.heappop(heap)
                # 判断pair是否失效
                if not alive[i]:
                    continue
                j = nxt[i]
                if j == -1 or not alive[j]:
                    continue
                
                # 确认这一对组合当前的rank仍然和堆顶的rank匹配(懒惰删除过期项)
                pair = (tokens[i], tokens[j])
                current_rank = self.merge_rank.get(pair)
                if current_rank != rank:
                    continue

                tokens[i] = tokens[i] + tokens[j]
                alive[j] = False

                # 维护链表
                nj = nxt[j]
                nxt[i] = nj
                if nj != -1:
                    prev[nj] = i
                
                # 左邻对入堆
                li = prev[i]
                if li != -1 and alive [li]:
                    pair_left = (tokens[li], tokens[i])
                    r_left = self.merge_rank.get((tokens[li], tokens[i]))
                    if r_left is not None:
                        heapq.heappush(heap, (r_left, li))

                # 右邻对入堆
                if nj != -1 and alive[nj]:
                    pair_right = (tokens[i], tokens[nj])
                    r_right = self.merge_rank.get((tokens[i], tokens[nj]))
                    if r_right is not None:
                        heapq.heappush(heap, (r_right, i))

            res = []
            idx = 0
            while idx != -1 and idx < n:
                if alive[idx]:
                    res.append(tokens[idx])
                idx = nxt[idx]

            return res 

        token_ids : list[int] = []
        for is_special, segment in split_specials(text):
            if is_special:
                seg_bytes = segment.encode("utf-8")
                tok_id = self.byte2id.get(seg_bytes)
                if tok_id is None:
                    raise KeyError(f"Special token {segment!r} not in vocab")
                token_ids.append(tok_id)
            else:
                for sub_text in regex.findall(GPT2_SPLIT_PATTERN, segment):
                    for tok in bpe_merge(sub_text.encode("utf-8")):
                        tok_id = self.byte2id.get(tok)
                        if tok_id is None:
                            raise KeyError(f"Token bytes {tok} not in vocab")
                        token_ids.append(tok_id)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for tok_id in self.encode(chunk):
                yield tok_id


    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
