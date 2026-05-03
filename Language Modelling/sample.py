"""
sample.py — Name generation from a trained character-level model.
"""

import torch
import torch.nn.functional as F


def generate_names(
    model,
    itos: dict[int, str],
    stoi: dict[str, int],
    *,
    ctx_len: int = 4,
    count: int = 20,
) -> list[str]:
    """Sample `count` names from the model."""
    names: list[str] = []

    for _ in range(count):
        out_str = ""
        ctx = [stoi["."]] * ctx_len

        while True:
            logits = model(ctx)
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1).item()
            char = itos[idx]

            if char == ".":
                break

            out_str += char
            ctx = ctx[1:] + [idx]

        names.append(out_str)

    return names
