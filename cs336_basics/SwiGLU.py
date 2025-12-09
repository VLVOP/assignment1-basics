import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor
from cs336_basics.linear import llmLinearModel

class llmSWiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(llmSWiGLU, self).__init__()
        self.linear1 = llmLinearModel(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = llmLinearModel(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = llmLinearModel(d_model, d_ff, device=device, dtype=dtype)
        self.silu = nn.SiLU()
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x1 = self.linear1(x)
        x2 = self.linear3(x)
        return self.linear2(self.silu(x1) * x2)

