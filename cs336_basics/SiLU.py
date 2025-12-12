import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:

        return x*self.sigmoid(x)