"""Shared portfolio construction: every score-based model goes through the same rules as the production strategy."""
from __future__ import annotations

import numpy as np

from .data import Market


def scores_to_weights(S: np.ndarray, mkt: Market, rows: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> np.ndarray:
    """S[len(rows), N] predicted excess return -> long-only weights (sum <= 1, remainder is cash).

    Top `top_k` assets with score > threshold, inverse-volatility weighted, each owning at most 1/top_k;
    empty slots go to the best positive-scoring defensive asset, else stay in cash.
    """
    W = np.zeros_like(S, dtype=float)
    vol = mkt.vol[rows]
    for i in range(len(S)):
        s = np.where(np.isfinite(S[i]), S[i], -np.inf)
        order = np.argsort(-s)[:top_k]
        picks = [j for j in order if s[j] > threshold]
        if picks:
            iv = 1.0 / np.maximum(vol[i, picks], 1e-6)
            W[i, picks] = iv / iv.sum() * len(picks) / top_k
        idle = top_k - len(picks)
        if idle > 0:
            d = np.where(mkt.defensive_mask, s, -np.inf)
            j = int(np.argmax(d))
            if d[j] > threshold:
                W[i, j] += idle / top_k
    return W


def smooth(W: np.ndarray, n: int) -> np.ndarray:
    """Average of the last n weekly weight vectors (the production strategy's smoothing)."""
    if n <= 1:
        return W
    out = np.zeros_like(W)
    for i in range(len(W)):
        out[i] = W[max(0, i - n + 1) : i + 1].mean(axis=0)
    return out
