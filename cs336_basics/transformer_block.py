from jaxtyping import Float
from torch import Tensor
from cs336_basics.RMSNorm import llmRMSNorm
from cs336_basics.SwiGLU import llmSWiGLU
from cs336_basics.multihead_self_attention import causalMultiheadSelfAttention
from cs336_basics.RoPE import RotaryPositionalEmbedding
import torch.nn as nn
import torch

class preNormTransBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super(preNormTransBlock, self).__init__()

        self.RMSNorm1 = llmRMSNorm(d_model, device=device, dtype=dtype)
        self.RMSNorm2 = llmRMSNorm(d_model, device=device, dtype=dtype)
        self.MHA = causalMultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.FFN = llmSWiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "batch ... seq_len d_model"], rope=None, token_position=None) -> Float[Tensor, "batch ... seq_len d_model"]:
        
        out1 = x + self.MHA(self.RMSNorm1(x), rope, token_position)
        out2 = out1 + self.FFN(self.RMSNorm2(out1))

        return out2
