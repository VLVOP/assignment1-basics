import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor
import math

class llmLinearModel(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(llmLinearModel, self).__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )


    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return x @ self.W.T
