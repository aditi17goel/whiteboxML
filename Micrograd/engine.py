"""
engine.py — The autograd engine.

Implements the `Value` class (a scalar with automatic differentiation)
and the topological-sort-based backpropagation routine.

"""

from __future__ import annotations

import math


class Value:
    """A scalar value that tracks its computation graph for automatic differentiation.

    Every arithmetic operation on `Value` objects builds a DAG. Calling
    `backward()` on the final node walks the graph in reverse topological
    order and fills in `.grad` for every upstream node via the chain rule.
    """

    def __init__(self, data: float, children: list["Value"] | None = None, label: str = ""):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._children = children if isinstance(children, list) else []
        self._backward = lambda: None  # no-op by default
        self._op = ""

    # ------------------------------------------------------------------
    # Arithmetic operations (forward pass + local backward definition)
    # ------------------------------------------------------------------

    def __add__(self, other: "Value | int | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other])
        out._op = "+"

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: "Value | int | float") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        out = Value(-self.data, [self])
        out._op = "neg"

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: "Value | int | float") -> "Value":
        return self + (-other)

    def __rsub__(self, other: "Value | int | float") -> "Value":
        return (-self) + other

    def __mul__(self, other: "Value | int | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other])
        out._op = "*"

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: "Value | int | float") -> "Value":
        return self * other

    def __pow__(self, exponent: int | float) -> "Value":
        assert isinstance(exponent, (int, float)), "Only int/float exponents are supported"
        out = Value(self.data ** exponent, [self])
        out._op = f"**{exponent}"

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: "Value | int | float") -> "Value":
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def __rtruediv__(self, other: "Value | int | float") -> "Value":
        return other * (self ** -1) if isinstance(other, Value) else Value(other) * (self ** -1)

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, [self])
        out._op = "tanh"

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Run backpropagation from this node through the entire computation graph."""
        # Build a topological ordering (root → leaves)
        topo_order: list[Value] = []
        visited: set[int] = set()

        def _build_topo(node: Value):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    _build_topo(child)
                topo_order.append(node)

        _build_topo(self)

        # Walk in reverse (from output to inputs)
        self.grad = 1.0
        for node in reversed(topo_order):
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        self.grad = 0.0

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
