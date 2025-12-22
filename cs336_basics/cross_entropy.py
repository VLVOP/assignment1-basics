import torch
from torch import Tensor
from jaxtyping import Float, Int
from einx import get_at, max, sum, mean, reduce

def CEloss(targets: Int[Tensor, "batch_size ..."], logits: Float[Tensor, "batch_size ... vocab_size"]) -> Float[Tensor, ""]:
    
    logits_max_value = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max_value
    logits_max_label = get_at("... [v], ... -> ...", logits_stable, targets)

    log_sum_exp = torch.log(sum("... [v] -> ...", torch.exp(logits_stable)))

    ce_loss_tensor = log_sum_exp - logits_max_label

    ce_loss = mean("... ->", ce_loss_tensor)

    return ce_loss
    
