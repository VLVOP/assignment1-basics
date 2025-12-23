import torch
from math import sqrt
from torch import Tensor
from jaxtyping import Float
from typing import Optional
from collections.abc import Callable, Iterable

class SGDoptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0 :
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) :
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data = p.data - lr / sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss
    
if __name__ == "__main__":
    
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGDoptimizer([weights], lr=1e-3)

    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()