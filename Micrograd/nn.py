"""
nn.py — Neural network building blocks on top of the autograd engine.

Provides Neuron, Layer, and MLP classes. All parameters are `Value` objects
so gradients flow automatically via `engine.Value.backward()`.

"""

from __future__ import annotations

import math
import random

from engine import Value


class Neuron:
    """A single neuron: computes tanh(w · x + b).

    Uses Xavier-style weight initialization (scale = 1/√input_dim) to
    keep pre-activations in the region where tanh has meaningful gradients.
    """

    def __init__(self, input_dim: int, *, layer_id: int = 0, neuron_id: int = 0):
        scale = 1.0 / math.sqrt(input_dim)
        self.weights = [
            Value(random.uniform(-scale, scale), label=f"w:{layer_id}:{neuron_id}:{i}")
            for i in range(input_dim)
        ]
        self.bias = Value(0.0, label=f"b:{layer_id}:{neuron_id}")
        self.input_dim = input_dim

    def __call__(self, x: list) -> Value:
        assert len(x) == self.input_dim, (
            f"Expected {self.input_dim} inputs, got {len(x)}"
        )
        activation = self.bias
        for xi, wi in zip(x, self.weights):
            activation = activation + xi * wi
        return activation.tanh()

    @property
    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]


class Layer:
    """A fully-connected layer of neurons."""

    def __init__(self, input_dim: int, output_dim: int, *, layer_id: int = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = [
            Neuron(input_dim, layer_id=layer_id, neuron_id=i)
            for i in range(output_dim)
        ]

    def __call__(self, x: list) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    @property
    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters]


class MLP:
    """A multi-layer perceptron (fully-connected feed-forward network).

    Example:
        >>> net = MLP(input_dim=2, layer_dims=[8, 16, 16, 8, 1])
        >>> out = net([0.5, -0.3])   # returns list of Value
    """

    def __init__(self, input_dim: int, layer_dims: list[int]):
        self.input_dim = input_dim
        self.layers: list[Layer] = []
        prev_dim = input_dim
        for i, dim in enumerate(layer_dims):
            self.layers.append(Layer(prev_dim, dim, layer_id=i))
            prev_dim = dim

    def __call__(self, x: list) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters]

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero."""
        for p in self.parameters:
            p.zero_grad()
