# Language Modelling — Character-Level Name Generation

A character-level MLP language model that learns to generate human-like names. Based on Andrej Karpathy's [Lecture 2](https://www.youtube.com/watch?v=PaCmpygFfXo) and [Lecture 3](https://www.youtube.com/watch?v=TCH_1BHY58I) from the Neural Networks: Zero to Hero series.

📝 **Blog post:** [Character-Level Language Models](https://goeladiti.substack.com/p/character-level-language-models)

---

## Structure

The original rough implementation lives in [`makemore_rough_implementation.ipynb`](./makemore_rough_implementation.ipynb). I then split it out into separate files for better structure and readability:

| File | What it does |
|------|-------------|
| `data.py` | Vocabulary building, context-window extraction, train/dev/test splitting |
| `model.py` | `CharMLP` — embedding table → hidden layer (tanh) → output logits |
| `train.py` | Training loop with LR decay, evaluation, and loss plotting |
| `sample.py` | Autoregressive name generation from a trained model |

---

## Quick Start

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install PyTorch and Matplotlib
pip install torch matplotlib

python3 train.py
```

To download the dataset, run:
```bash
wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```
---

## Architecture

```
Input (4 chars) → Embedding (27×10) → Concat (40-d) → Hidden (200, tanh) → Logits (27)
```

Trained with cross-entropy loss, mini-batch SGD (batch size 32), and a simple learning rate schedule (0.1 → 0.01 at epoch 50).
