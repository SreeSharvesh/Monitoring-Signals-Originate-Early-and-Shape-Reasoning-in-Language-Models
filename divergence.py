
# divergence.py
# Layerwise divergence utilities, including prefix-window and anchor-aligned comparisons.

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import re

def cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def layerwise_cosine_distance(hidden_A: List[np.ndarray],
                              hidden_B: List[np.ndarray],
                              align_tokens: int) -> List[float]:
    """
    hidden_*: list of arrays per layer: [emb, layer1, layer2, ...], each (1, S, d)
    Returns per-layer cosine distance (1 - cosine) after averaging token dimension over first align_tokens.
    """
    print("In layerwise cosine")
    assert len(hidden_A) == len(hidden_B)
    L = len(hidden_A)
    dists: List[float] = []
    for li in range(1, L):  # skip embeddings [0]
        a = hidden_A[li][0, :align_tokens, :].mean(axis=0)
        b = hidden_B[li][0, :align_tokens, :].mean(axis=0)
        d = 1.0 - cosine(a, b)
        dists.append(d)
    return dists

def pick_prefix_window(seq_lens: Tuple[int, int, int], frac: float = 0.6, min_tokens: int = 24) -> int:
    """Choose a conservative prefix window over which to compare, avoiding late drift."""
    print("In pick prefix window")
    m = int(min(seq_lens) * frac)
    return max(min_tokens, m)

SENT_END = re.compile(r'[.!?]')

def find_first_sentence_end(text: str) -> Optional[int]:
    """Heuristic: token count until first sentence end after 'Reasoning:' label; returns None if not found."""
    body = text.split("Reasoning:", 1)[-1] if "Reasoning:" in text else text
    m = SENT_END.search(body)
    if not m:
        return None
    # Return an approximate char index; token alignment should be handled upstream if needed.
    return m.end()
