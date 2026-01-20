from einops import rearrange
import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor

from cs336_basics.linear import llmLinearModel
from cs336_basics.scaled_dot_product_attention import scaledDotProductAttention
from cs336_basics.RoPE import RotaryPositionalEmbedding


class causalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, use_rope: bool = True):
        super(causalMultiheadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.w_qkv = llmLinearModel(d_model, 3 * d_model, device=device, dtype=dtype)
        self.use_rope = use_rope    

        self.linear = llmLinearModel(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "batch_size ... seq_len d_model"], rope=None, token_position=None) -> Float[Tensor, "... seq_len d_out"]:
        Q, K, V = rearrange(
            self.w_qkv(x),
            "batch_size ... seq_len (three h d) -> three batch_size ... h seq_len d",
            three = 3,
            h = self.num_heads
        )

        if token_position is not None and self.use_rope:
            Q = rope(Q, token_position)
            K = rope(K, token_position)

        mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2]), device=x.device)).bool()

        attn_out = scaledDotProductAttention(Q, K, V, mask)

        out = rearrange(attn_out, "... heads seq_len d_head -> ... seq_len (heads d_head)")

        return self.linear(out)


