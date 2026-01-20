from jaxtyping import Float
from torch import Tensor
from cs336_basics.SiLU import SiLU
from cs336_basics.RMSNorm import llmRMSNorm
from cs336_basics.SwiGLU import llmSWiGLU
from cs336_basics.linear import llmLinearModel
from cs336_basics.multihead_self_attention import causalMultiheadSelfAttention
from cs336_basics.RoPE import RotaryPositionalEmbedding
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, use_rope: bool = True, ffn_type: str = "swiglu", norm_type: str = "pre", use_rmsnorm: bool = True):
        super(TransformerBlock, self).__init__()
        self.use_rmsnorm = use_rmsnorm
        self.norm_type = norm_type

        if use_rmsnorm:
            self.RMSNorm1 = llmRMSNorm(d_model, device=device, dtype=dtype)
            self.RMSNorm2 = llmRMSNorm(d_model, device=device, dtype=dtype)
        
        self.MHA = causalMultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype, use_rope=use_rope)
        
        if ffn_type == "swiglu":
            self.FFN = llmSWiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.FFN = nn.Sequential(
                llmLinearModel(d_model, d_ff, device=device, dtype=dtype),
                SiLU(),
                llmLinearModel(d_ff, d_model, device=device, dtype=dtype)
            )

    def forward(self, x: Float[Tensor, "batch ... seq_len d_model"], rope=None, token_position=None) -> Float[Tensor, "batch ... seq_len d_model"]:
        
        residual = x
        if self.use_rmsnorm and self.norm_type == "pre":
            x = self.RMSNorm1(x)
        
        x = self.MHA(x, rope, token_position)
        
        if self.use_rmsnorm and self.norm_type == "post":
            x = residual + x
            x = self.RMSNorm1(x)
        else:
            x = residual + x

        residual = x
        if self.use_rmsnorm and self.norm_type == "pre":
            x = self.RMSNorm2(x)
        
        x = self.FFN(x)

        if self.use_rmsnorm and self.norm_type == "post":
            x = residual + x
            x = self.RMSNorm2(x)
        else:
            x = residual + x

        return x
