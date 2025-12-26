import torch 
from jaxtyping import Float
from torch import Tensor

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> Float[Tensor, ""]:
    if t < T_w:
        return alpha_max * t / T_w
    elif t >= T_w and t <= T_c:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + torch.cos(torch.pi * (t - T_w) / (T_c - T_w)))
    else:
        return alpha_min