"""
train.py — Training script for the character-level language model.

Usage:
    python train.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import load_words, build_vocab, build_dataset, train_dev_test_split
from model import CharMLP
from sample import generate_names

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
CTX_LEN = 4
EMB_DIM = 10
HIDDEN_SIZE = 200
EPOCHS = 100
BATCH_SIZE = 32
LR_HIGH = 0.1
LR_LOW = 0.01
LR_DECAY_EPOCH = 50


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(
    model: CharMLP,
    X_train: list,
    Y_train: list,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr_high: float = LR_HIGH,
    lr_low: float = LR_LOW,
    lr_decay_epoch: int = LR_DECAY_EPOCH,
) -> list[float]:
    """Train the model and return per-step loss history."""
    loss_history: list[float] = []

    for e in range(epochs):
        lr = lr_high if e < lr_decay_epoch else lr_low

        for step in range(len(X_train) // batch_size):
            X_batch = torch.tensor(X_train[step * batch_size : (step + 1) * batch_size])
            Y_batch = torch.tensor(Y_train[step * batch_size : (step + 1) * batch_size])

            # Zero gradients
            for p in model.parameters:
                p.grad = None

            # Forward + loss
            logits = model(X_batch)
            loss = F.cross_entropy(input=logits, target=Y_batch)

            # Backward + update
            loss.backward()
            for p in model.parameters:
                p.data += -lr * p.grad

            loss_history.append(loss.item())

        print(f"Epoch {e + 1:3d}/{epochs}  |  loss: {loss.item():.4f}  |  lr: {lr}")

    return loss_history


def evaluate(model: CharMLP, X: list, Y: list) -> float:
    """Compute cross-entropy loss on a dataset split."""
    logits = model(torch.tensor(X))
    return F.cross_entropy(input=logits, target=torch.tensor(Y)).item()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # Load data
    words = load_words()
    stoi, itos, vocab_size = build_vocab(words)
    X, Y = build_dataset(words, stoi, ctx_len=CTX_LEN)
    (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test) = train_dev_test_split(X, Y)

    print(f"Train: {len(X_train)}  |  Dev: {len(X_dev)}  |  Test: {len(X_test)}")

    # Build model
    model = CharMLP(
        vocab_size=vocab_size,
        emb_dim=EMB_DIM,
        ctx_len=CTX_LEN,
        hidden_size=HIDDEN_SIZE,
    )
    total_params = sum(p.numel() for p in model.parameters)
    print(f"Model parameters: {total_params}")

    # Train
    loss_history = train(model, X_train, Y_train)

    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, linewidth=0.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # Evaluate
    train_loss = evaluate(model, X_train, Y_train)
    dev_loss = evaluate(model, X_dev, Y_dev)
    test_loss = evaluate(model, X_test, Y_test)
    print(f"\nTrain loss: {train_loss:.4f}")
    print(f"Dev loss:   {dev_loss:.4f}")
    print(f"Test loss:  {test_loss:.4f}")

    # Generate sample names
    print("\n--- Generated Names ---")
    names = generate_names(model, itos, stoi, ctx_len=CTX_LEN, count=20)
    for name in names:
        print(name)


if __name__ == "__main__":
    main()
