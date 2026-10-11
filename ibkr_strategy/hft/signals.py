"""Online (event-driven) microstructure features and a tiny linear alpha model.

The features replicate `features.build` exactly (imbalance, trailing signed trade flow) so a model fitted on the
vectorised research path can score live ticks. HFT models must be this small: tens of nanoseconds in C++, a few
microseconds in Python. `test_hft.py` asserts online/offline parity."""
from __future__ import annotations

from collections import deque

import numpy as np

from .core import Quote, Trade

SIGNAL_FEATURES = ["imb", "flow_100ms", "flow_1s"]


class OnlineFeatures:
    def __init__(self):
        self.trades: deque = deque()   # (t, signed_volume)
        self.f100 = 0.0
        self.f1s = 0.0
        self._q100: deque = deque()
        self.imb = 0.0

    def on_trade(self, tr: Trade) -> None:
        sv = tr.side * tr.sz
        self.trades.append((tr.t, sv))
        self.f1s += sv
        self._q100.append((tr.t, sv))
        self.f100 += sv

    def on_quote(self, q: Quote) -> None:
        self.imb = (q.bsz - q.asz) / (q.bsz + q.asz)
        t = q.t
        while self.trades and self.trades[0][0] <= t - 1000:
            self.f1s -= self.trades.popleft()[1]
        while self._q100 and self._q100[0][0] <= t - 100:
            self.f100 -= self._q100.popleft()[1]

    def vector(self) -> np.ndarray:
        return np.array([self.imb, self.f100 / 1e3, self.f1s / 1e3])


class LinearAlpha:
    """score = intercept + w . x  -> predicted mid move (ticks) over the next second."""

    def __init__(self, w: np.ndarray, b: float = 0.0):
        self.w = np.asarray(w, float)
        self.b = float(b)

    def score(self, x: np.ndarray) -> float:
        return self.b + float(self.w @ x)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> "LinearAlpha":
        y = np.clip(y, -5, 5)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + l2 * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        w_raw = w / sd
        return cls(w_raw, y.mean() - mu @ w_raw)
