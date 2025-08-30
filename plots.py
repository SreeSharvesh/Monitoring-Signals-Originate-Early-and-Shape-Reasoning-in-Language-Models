
# plots.py
# Bootstrap CIs for divergence curves and simple matplotlib output.

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bootstrap_ci(arr: np.ndarray, iters: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    print("In bootstap ci")
    means = []
    n = len(arr)
    for _ in range(iters):
        idx = np.random.randint(0, n, size=n)
        means.append(arr[idx].mean())
    means = np.sort(means)
    lo = np.percentile(means, 100 * (alpha/2))
    hi = np.percentile(means, 100 * (1 - alpha/2))
    return float(np.mean(arr)), float(lo), float(hi)

def plot_divergence(df: pd.DataFrame, out_png: str):
    """
    df columns expected:
      - layer (int)
      - div_mon (float)  # per-item per-layer Monitored–Hidden distances
      - div_pla (float)  # per-item per-layer Placebo–Hidden distances
    """

    print("Plotting divergence")
    layers = sorted(df["layer"].unique())
    mon_means = []
    mon_los = []
    mon_his = []
    pla_means = []
    pla_los = []
    pla_his = []
    for L in layers:
        mvals = df.loc[df.layer == L, "div_mon"].to_numpy()
        pvals = df.loc[df.layer == L, "div_pla"].to_numpy()
        m_mean, m_lo, m_hi = bootstrap_ci(mvals)
        p_mean, p_lo, p_hi = bootstrap_ci(pvals)
        mon_means.append(m_mean); mon_los.append(m_lo); mon_his.append(m_hi)
        pla_means.append(p_mean); pla_los.append(p_lo); pla_his.append(p_hi)

    plt.figure(figsize=(7.5,4.5))
    plt.plot(layers, mon_means, label="Monitored–Hidden")
    plt.fill_between(layers, mon_los, mon_his, alpha=0.2)
    plt.plot(layers, pla_means, label="Placebo–Hidden")
    plt.fill_between(layers, pla_los, pla_his, alpha=0.2)
    plt.xlabel("Layer index (1..N)")
    plt.ylabel("Cosine distance (1 - cos)")
    plt.title("Divergence vs Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
