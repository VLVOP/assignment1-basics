import torch
from jaxtyping import Float, Int
from torch import Tensor


def generate(model: torch.nn.Module, input_ids: Int[Tensor, "batch_size seq_len"], max_new_tokens: int, eos_token_id: int = None, temperature: float = 1.0, top_p: float = 0.9) -> Int[Tensor, "batch_size total_seq"]:

    model.eval()

    idx = input_ids

    for _ in range(max_new_tokens):

        if hasattr(model, "rope") and hasattr(model.rope, "cos_cached"):
            context_length = model.rope.cos_cached.shape[0]
            idx_cond = idx[:, -context_length:]
        else:
            idx_cond = idx

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if temperature > 0.0:
            logits = logits / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    1, sorted_indices, sorted_indices_to_remove
                )

                logits[indices_to_remove] = float("-inf")
            
            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_token_id is not None:
            if (idx_next == eos_token_id).all():
                break
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx



    