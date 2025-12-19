from jaxtyping import Float, Bool
from torch import Tensor
import torch
from cs336_basics.softmax import softmax

def scaledDotProductAttention(
    Q: Float[Tensor, "batch_size ... seq_len d_k"],
    K: Float[Tensor, "batch_size ... seq_len d_k"],
    V: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"]=None
) -> Float[Tensor, "batch_size ... d_v"]:
    
    pre_softmax_scores = torch.einsum("... i k, ... j k -> ... i j", Q, K) / (Q.shape[-1] ** 0.5)

    if mask is not None:
        pre_softmax_scores = pre_softmax_scores.masked_fill(mask == False, float('-inf'))

    attn_weights = softmax(pre_softmax_scores, dim=-1)

    attn_score = torch.einsum("... i j, ... j v -> ... i v", attn_weights, V)

    return attn_score