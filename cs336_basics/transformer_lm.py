from einops import repeat
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from cs336_basics.embedding import llmEmbeddingModel
from cs336_basics.transformer_block import preNormTransBlock
from cs336_basics.RMSNorm import llmRMSNorm
from cs336_basics.linear import llmLinearModel
from cs336_basics.softmax import softmax
from cs336_basics.RoPE import RotaryPositionalEmbedding

class transformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float = 10000.0, device=None, dtype=None):
        super(transformerLM, self).__init__()

        self.embedding = llmEmbeddingModel(vocab_size, d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            preNormTransBlock(
                d_model,
                num_heads,
                d_ff,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        self.RMSNorm = llmRMSNorm(d_model, device=device, dtype=dtype)
        self.linear = llmLinearModel(d_model, vocab_size, device=device, dtype=dtype)
        
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length, device=device)

    def forward(self, x: Float[Tensor, "batch_size seq_len"], token_position=None) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        
        emb = self.embedding(x)

        if token_position is None:
            batch, seq, _ = emb.shape
            pos = torch.arange(seq, device=x.device)
            token_position = repeat(pos, "s -> b s", b = batch)

        for layer in self.layers:
            emb = layer(emb, rope=self.rope, token_position=token_position)
        
        trans_emb_norm = self.RMSNorm(emb)

        linear_out = self.linear(trans_emb_norm)

        return linear_out


