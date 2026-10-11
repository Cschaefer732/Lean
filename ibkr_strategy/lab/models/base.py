from __future__ import annotations

import numpy as np

from ..construct import scores_to_weights
from ..data import Market, momentum_gate, rank_target


class SupervisedStrategy:
    """Wraps a regressor that predicts next-`horizon`-week excess return per (date, asset)."""

    name = "supervised"
    top_k = 5
    use_rank = False   # train on cross-sectional rank of forward excess return instead of the raw value
    gate = False       # restrict picks to assets passing the production absolute-momentum rule

    def make_model(self):
        raise NotImplementedError

    def _xy(self, mkt: Market, rows: np.ndarray):
        X = mkt.X[rows].reshape(-1, mkt.X.shape[-1])
        y = (rank_target(mkt.y[rows]) if self.use_rank else mkt.y[rows]).reshape(-1)
        return X, y

    def fit(self, mkt: Market, train_rows: np.ndarray) -> None:
        X, y = self._xy(mkt, train_rows)
        ok = np.isfinite(y)
        self.model = self.make_model()
        self.model.fit(X[ok], np.clip(y[ok], *np.nanpercentile(y[ok], [1, 99])))
        self.scores_ = None

    def scores(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        X = mkt.X[rows].reshape(-1, mkt.X.shape[-1])
        return self.model.predict(X).reshape(len(rows), mkt.N)

    def weights(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        S = self.scores(mkt, rows)
        self.last_scores = S
        return gated_weights(S, mkt, rows, self.top_k, self.gate, self.use_rank)


def gated_weights(S, mkt: Market, rows: np.ndarray, top_k: int, gate: bool, rank_scores: bool) -> np.ndarray:
    if gate:
        S = np.where(momentum_gate(mkt, rows), S, -np.inf)
    # rank-target scores are centred at 0 but carry no absolute-return meaning, so no zero threshold applies
    return scores_to_weights(S, mkt, rows, top_k=top_k, threshold=-1e9 if rank_scores else 0.0)
