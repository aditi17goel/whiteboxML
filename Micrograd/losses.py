"""
losses.py — Loss functions for training.

Each function takes (prediction, label) and returns a scalar `Value`
that participates in the computation graph.
"""

from engine import Value


def mse_loss(prediction: Value, label: float) -> Value:
    """Mean squared error: (prediction - label)²"""
    return (prediction - label) ** 2


def hinge_squared_loss(prediction: Value, label: float) -> Value:
    """Squared hinge loss for binary classification (labels ∈ {-1, +1})."""
    return (Value(1) - label * prediction) ** 2
