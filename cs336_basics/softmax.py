from jaxtyping import Float
from torch import Tensor
import torch

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x_max = x.max(dim=dim, keepdim=True).values

    x_safe = x - x_max

    exp_x = torch.exp(x_safe)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)
