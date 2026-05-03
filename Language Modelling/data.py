"""
data.py — Dataset utilities for character-level language modelling.

Handles vocabulary building, context-window extraction, and
train/dev/test splitting from a names.txt file.
"""

import random
import urllib.request


def build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str], int]:
    """Build character-to-index and index-to-character mappings.

    Returns (stoi, itos, vocab_size). The special '.' token is index 0.
    """
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos, len(itos)


def build_dataset(
    words: list[str],
    stoi: dict[str, int],
    ctx_len: int = 4,
) -> tuple[list[list[int]], list[int]]:
    """Convert a list of words into (context, target) training pairs.

    Each word is padded with '.' tokens and slid over with a window of
    size `ctx_len` to produce input sequences and their next-character targets.
    """
    X, Y = [], []
    for w in words:
        w = "." * ctx_len + w + "."
        for i in range(len(w) - ctx_len):
            x = [stoi[c] for c in w[i : i + ctx_len]]
            y = stoi[w[i + ctx_len]]
            X.append(x)
            Y.append(y)
    return X, Y


def train_dev_test_split(
    X: list, Y: list, train_frac: float = 0.8, dev_frac: float = 0.1
) -> tuple:
    """Split data into train / dev / test sets."""
    n = len(X)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_dev, Y_dev = X[n_train : n_train + n_dev], Y[n_train : n_train + n_dev]
    X_test, Y_test = X[n_train + n_dev :], Y[n_train + n_dev :]

    return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)


def load_words(
    url: str = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", 
    seed: int = 42
) -> list[str]:
    """Load and shuffle the word list directly from a URL (fallback to local)."""
    try:
        response = urllib.request.urlopen(url)
        words = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"Warning: Could not download dataset ({e}). Falling back to local 'names.txt'.")
        words = open("names.txt", "r").read().splitlines()
        
    random.seed(seed)
    random.shuffle(words)
    return words
