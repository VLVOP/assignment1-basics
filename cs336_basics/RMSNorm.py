from einops import einsum
import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor
import math

class llmRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(llmRMSNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS calculation
        RMS_a = math.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = einsum("b s d, d -> b s d", x, self.g) / RMS_a

        return x_normed.to(in_dtype)
