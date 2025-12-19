import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum, repeat, rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_k = d_k
        self.device = device
        
        indices = torch.arange(0, d_k, 2, device=device)
        inv_freq = 1.0 / (theta ** (indices.float() / d_k))

        i = torch.arange(max_seq_len, device=device)

        freqs = einsum(i, inv_freq, "i, j -> i j") 
        
        emb = repeat(freqs, "i j -> i (j repeat)", repeat=2)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        if x.ndim > cos.ndim:
            diff = x.ndim - cos.ndim

            pattern = " ".join(["1"] * diff)
            cos = rearrange(cos, f"... s d -> ... {pattern} s d")
            sin = rearrange(sin, f"... s d -> ... {pattern} s d")
        
        x_pairs = rearrange(x, "... (d two) -> ... d two", d=self.d_k // 2)

        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        x_rot_pairs = torch.stack((-x2, x1), dim=-1)

        x_rotated = rearrange(x_rot_pairs, "... d two -> ... (d two)")

        return x * cos + x_rotated * sin

