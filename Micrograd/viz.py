"""
viz.py — Visualisation utilities.

- `draw_graph`: render a Value computation graph with Graphviz.
- `plot_decision_boundary`: colour-coded 2D decision region for a trained model.
- `plot_data`: scatter plot of labelled 2D data.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import graphviz
except ImportError:
    graphviz = None  # Graphviz is optional (not always available outside Colab)

from engine import Value


# ------------------------------------------------------------------
# Computation graph rendering
# ------------------------------------------------------------------

def draw_graph(root: Value) -> "graphviz.Digraph | None":
    """Render the computation graph rooted at `root` as a Graphviz diagram.

    Returns a `graphviz.Digraph` object (call `.render()` to save, or
    display inline in Jupyter).  Returns `None` if graphviz is not installed.
    """
    if graphviz is None:
        print("graphviz is not installed — skipping graph rendering.")
        return None

    dot = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})
    visited: set[int] = set()

    def _add_node(node: Value):
        nid = str(id(node))
        if id(node) in visited:
            return
        visited.add(id(node))

        # Node label
        parts = []
        if node.label:
            parts.append(node.label)
        parts.append(f"data {node.data:.4f}")
        parts.append(f"grad {node.grad:.4f}")
        dot.node(nid, "{ " + " | ".join(parts) + " }", shape="record")

        # Operation node + edges from children
        if node._children:
            op_id = nid + node._op
            dot.node(op_id, node._op)
            dot.edge(op_id, nid)
            for child in node._children:
                _add_node(child)
                dot.edge(str(id(child)), op_id)

    _add_node(root)
    return dot


# ------------------------------------------------------------------
# Data & decision-boundary plots
# ------------------------------------------------------------------

def plot_data(
    inputs: list[tuple[float, float]],
    labels: list[int],
    *,
    title: str = "Training Data",
    figsize: tuple[int, int] = (6, 6),
) -> None:
    """Scatter plot of 2D data coloured by label."""
    arr = np.array(inputs)
    plt.figure(figsize=figsize)
    plt.scatter(arr[:, 0], arr[:, 1], c=labels, s=20, cmap="jet")
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.show()


def plot_decision_boundary(
    model,
    inputs: list[tuple[float, float]],
    labels: list[int],
    *,
    resolution: float = 0.05,
    figsize: tuple[int, int] = (7, 7),
) -> None:
    """Visualise the decision boundary of a trained 2-input model.

    Creates a mesh over [-1.1, 1.1]², evaluates the model at each grid point,
    and overlays the original training data.
    """
    x_min, x_max = -1.1, 1.1
    y_min, y_max = -1.1, 1.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = [model(list(map(float, pt)))[0].data for pt in grid_points]
    Z = np.array(scores).reshape(xx.shape)

    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    arr = np.array(inputs)
    plt.scatter(arr[:, 0], arr[:, 1], c=labels, s=20, edgecolors="k", cmap=plt.cm.Spectral)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Decision Boundary")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.show()
