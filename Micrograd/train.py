"""
train.py — Training script for the circle-classification toy problem.

Generates 500 random 2D points, labels them as inside/outside a circle
(radius √0.5), and trains a small MLP to learn the boundary.

Usage:
    python train.py

"""

import random

from engine import Value
from nn import MLP
from losses import hinge_squared_loss
from viz import plot_data, plot_decision_boundary

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
SEED = 42
DATA_SIZE = 500
LEARNING_RATE = 0.05
EPOCHS = 60
BATCH_SIZE = 10
GRAD_CLIP = 10.0

# Architecture: 2 → 8 → 16 → 16 → 8 → 1
INPUT_DIM = 2
LAYER_DIMS = [8, 16, 16, 8, 1]


# ------------------------------------------------------------------
# Data generation
# ------------------------------------------------------------------

def generate_circle_data(
    n: int = DATA_SIZE, seed: int = SEED
) -> tuple[list[tuple[float, float]], list[int]]:
    """Generate `n` random 2D points labelled +1 (inside circle) or -1 (outside).

    Circle has radius √0.5, centred at the origin, in the [-1, 1]² square.
    """
    random.seed(seed)
    inputs = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
    labels = [1 if x ** 2 + y ** 2 <= 0.5 else -1 for x, y in inputs]
    return inputs, labels


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(
    model: MLP,
    inputs: list[tuple[float, float]],
    labels: list[int],
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    grad_clip: float = GRAD_CLIP,
) -> list[float]:
    """Train `model` on the given dataset and return per-batch loss history."""
    n = len(inputs)
    loss_history: list[float] = []

    for epoch in range(epochs):
        # Shuffle each epoch
        combined = list(zip(inputs, labels))
        random.shuffle(combined)
        shuffled_inputs, shuffled_labels = zip(*combined)

        epoch_loss = 0.0
        n_batches = n // batch_size

        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size

            # Forward pass + loss accumulation
            batch_loss = Value(0.0)
            for inp, label in zip(
                shuffled_inputs[start:end], shuffled_labels[start:end]
            ):
                prediction = model(inp)[0]
                batch_loss = batch_loss + hinge_squared_loss(prediction, label)
            batch_loss = batch_loss * (1.0 / batch_size)  # mean over batch

            # Backward pass
            batch_loss.backward()

            # Gradient-clipped SGD update
            for p in model.parameters:
                clipped = max(-grad_clip, min(grad_clip, p.grad))
                p.data -= lr * clipped
            model.zero_grad()

            loss_history.append(batch_loss.data)
            epoch_loss += batch_loss.data

        avg = epoch_loss / n_batches
        print(f"Epoch {epoch + 1:3d}/{epochs}  |  avg loss: {avg:.4f}")

    return loss_history


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    inputs, labels = generate_circle_data()

    # Visualise the raw data
    plot_data(inputs, labels, title="Circle Dataset (√0.5 radius)")

    # Build and train the network
    model = MLP(input_dim=INPUT_DIM, layer_dims=LAYER_DIMS)
    print(f"Model parameters: {len(model.parameters)}")
    loss_history = train(model, inputs, labels)

    # Visualise the learned decision boundary
    plot_decision_boundary(model, inputs, labels)


if __name__ == "__main__":
    main()
