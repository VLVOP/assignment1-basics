import torch
from math import sqrt 
from torch import Tensor
from jaxtyping import Float
from typing import Optional
from collections.abc import Callable, Iterable

class AdamWoptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: Iterable[float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                m = state["m"]
                v = state["v"]
                state['step'] += 1
                t = state["step"]
                
                grad = p.grad
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                lr_t = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= lr * weight_decay * p.data
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                state["m"] = m
                state["v"] = v

        return loss
