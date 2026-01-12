import torch
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor

def get_batch(
    x: Int[np.ndarray, "N"],
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    
    len = len(x)

    ix = torch.randint(len - context_length, (batch_size,))

    x_batch = [x[i : i + context_length].astype(np.int64) for i in ix]
    y_batch = [x[i + 1 : i + context_length + 1].astype(np.int64) for i in ix]

    inputs = torch.tensor(np.stack(x_batch), dtype=torch.long, device=device)
    targets = torch.tensor(np.stack(y_batch), dtype=torch.long, device=device)

    return inputs, targets