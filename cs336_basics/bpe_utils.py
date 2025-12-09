import functools

@functools.lru_cache()
def get_byte_encoder() -> dict[int, str]:
    """
    Returns a mapping from individual bytes (integers 0-255) to their
    Unicode string representation, consistent with GPT-2's BPE.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

@functools.lru_cache()
def get_byte_decoder() -> dict[str, int]:
    """
    Returns the inverse mapping of get_byte_encoder.
    """
    encoder = get_byte_encoder()
    return {v: k for k, v in encoder.items()}

def bytes_to_unicode_str(b: bytes) -> str:
    """
    Encodes a byte sequence into a Unicode string using the GPT-2 byte encoding.
    """
    encoder = get_byte_encoder()
    return "".join(encoder[x] for x in b)

def unicode_str_to_bytes(s: str) -> bytes:
    """
    Decodes a Unicode string (from GPT-2 byte encoding) back into a byte sequence.
    """
    decoder = get_byte_decoder()
    return bytes(decoder[c] for c in s)
