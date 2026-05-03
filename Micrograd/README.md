# Micrograd — Neural Networks & Backprop from Scratch

A from-scratch implementation of a scalar-valued autograd engine and a small neural network library built on top of it. Inspired by Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

📝 **Blog post:** [Neural Networks from Scratch — Micrograd](https://goeladiti.substack.com/p/neural-networks-from-scratch-micrograd)

---

## Structure

The original rough implementation lives in [`micrograd_rough_implementation.ipynb`](./micrograd_rough_implementation.ipynb). I then split it out into separate files for better structure and readability:

| File | What it does |
|------|-------------|
| `engine.py` | The `Value` class — a scalar with automatic differentiation and built-in `backward()` |
| `nn.py` | `Neuron`, `Layer`, `MLP` — neural network building blocks |
| `losses.py` | Loss functions (`mse_loss`, `hinge_squared_loss`) |
| `viz.py` | Visualisation utilities — computation graphs, data plots, decision boundaries |
| `train.py` | Training script for the circle-classification toy problem |
| `test_engine.py` | Correctness tests — compares gradients against PyTorch |

---

## Quick Start

```bash
python train.py
```

This generates 500 random 2D points, labels them as inside/outside a circle, and trains a small MLP (`2 → 8 → 16 → 16 → 8 → 1`) to learn the boundary.
