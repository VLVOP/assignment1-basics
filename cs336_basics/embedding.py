import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor

class llmEmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(llmEmbeddingModel, self).__init__()
        self.matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1.0
        torch.nn.init.trunc_normal_(
            self.matrix,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, token_ids: Float[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len embedding_dim"]:

        return self.matrix[token_ids]