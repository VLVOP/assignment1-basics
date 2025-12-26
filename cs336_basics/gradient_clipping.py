import torch
from jaxtyping import Float
from torch import Tensor

def clip_gradients(parameters: list[Tensor], max_norm: float) -> None:
    params_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None))

    if params_norm > max_norm:
        clip_coef = max_norm / (params_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)