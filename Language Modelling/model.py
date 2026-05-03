"""
model.py — Character-level MLP language model.

Architecture: Embedding → Hidden Layer (tanh) → Output Logits
"""

import torch


class CharMLP:
    """A character-level MLP language model.

    Args:
        vocab_size: Number of characters in the vocabulary.
        emb_dim: Dimensionality of character embeddings.
        ctx_len: Number of context characters the model sees.
        hidden_size: Number of neurons in the hidden layer.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 10,
        ctx_len: int = 4,
        hidden_size: int = 200,
    ):
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size

        # Embedding table: each character → emb_dim-dimensional vector
        self.emb = torch.randn(vocab_size, emb_dim, requires_grad=True)

        # Hidden layer
        self.W1 = torch.randn(ctx_len * emb_dim, hidden_size, requires_grad=True)
        self.b1 = torch.zeros(hidden_size, requires_grad=True)

        # Output layer
        self.W2 = torch.randn(hidden_size, vocab_size, requires_grad=True)
        self.b2 = torch.zeros(vocab_size, requires_grad=True)

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.emb, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x) -> torch.Tensor:
        """Forward pass: context indices → logits.

        Args:
            x: Tensor of shape (batch_size, ctx_len) with character indices.

        Returns:
            Logits of shape (batch_size, vocab_size).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        h = self.emb[x]                                      # (batch, ctx_len, emb_dim)
        h = h.view(-1, self.ctx_len * self.emb.shape[1])      # (batch, ctx_len * emb_dim)
        h = torch.tanh(h @ self.W1 + self.b1)                 # (batch, hidden_size)
        logits = h @ self.W2 + self.b2                         # (batch, vocab_size)
        return logits
