"""
test_engine.py — Correctness tests for the autograd engine.

Compares forward and backward pass results against PyTorch to validate
that Value produces identical gradients.

Usage:
    python test_engine.py
"""

import torch
from engine import Value


def test_simple_expression():
    """Test: y = h + q + q*x  where  z = 2x+2+x, q = tanh(z)+z*x, h = tanh(z²)"""
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # Same computation in PyTorch
    x = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    tol = 1e-6
    assert abs(ymg.data - ypt.data.item()) < tol, (
        f"Forward mismatch: {ymg.data} vs {ypt.data.item()}"
    )
    assert abs(xmg.grad - xpt.grad.item()) < tol, (
        f"Gradient mismatch for x: {xmg.grad} vs {xpt.grad.item()}"
    )
    print("✓ test_simple_expression passed")


def test_complex_expression():
    """Test a longer expression with +, *, **, /, tanh, and negation."""
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # Same computation in PyTorch
    a = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    assert abs(gmg.data - gpt.data.item()) < tol, (
        f"Forward mismatch: {gmg.data} vs {gpt.data.item()}"
    )
    assert abs(amg.grad - apt.grad.item()) < tol, (
        f"Gradient mismatch for a: {amg.grad} vs {apt.grad.item()}"
    )
    assert abs(bmg.grad - bpt.grad.item()) < tol, (
        f"Gradient mismatch for b: {bmg.grad} vs {bpt.grad.item()}"
    )
    print("✓ test_complex_expression passed")


if __name__ == "__main__":
    test_simple_expression()
    test_complex_expression()
    print("\nAll tests passed! ✅")
