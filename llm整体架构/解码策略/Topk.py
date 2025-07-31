import torch
import torch.nn.functional as F

#@torch.inference_mode()
@torch.no_grad()
def generate(self, idx, 
                eos, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        # forward the model to get the logits for the index in the sequence
        logits = self(idx_cond)##===================================
        logits = logits[:, -1, :] # crop to just the final time step
        if temperature == 0.0:
            # "sample" the single most likely index
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next==eos:
            break
    return idx