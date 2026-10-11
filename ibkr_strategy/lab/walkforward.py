"""Leak-free walk-forward evaluation: expanding training window, periodic retraining, label purging."""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .data import Market


class Strategy(Protocol):
    name: str

    def fit(self, mkt: Market, train_rows: np.ndarray) -> None: ...
    def weights(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        """Risk-asset weights [len(rows), N] using only information available at each row's date."""


def schedule(mkt: Market, oos_start: str, retrain_weeks: int = 52) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train_rows, test_rows)] blocks. A training row i is usable at block start t0 only if its label window
    (i .. i+horizon, and the next-week return used by GA/RL) has fully elapsed: i <= t0 - horizon - 1."""
    first = int(np.searchsorted(mkt.dates.values, np.datetime64(pd.Timestamp(oos_start))))
    blocks = []
    for t0 in range(first, mkt.T, retrain_weeks):
        t1 = min(t0 + retrain_weeks, mkt.T)
        train = np.arange(0, t0 - mkt.horizon)
        blocks.append((train, np.arange(t0, t1)))
    return blocks


def walk_forward(factory: Callable[[], Strategy], mkt: Market, oos_start: str, retrain_weeks: int = 52,
                 verbose: bool = False, return_scores: bool = False):
    out, scores = [], []
    for train, test in schedule(mkt, oos_start, retrain_weeks):
        strat = factory()
        strat.fit(mkt, train)
        w = strat.weights(mkt, test)
        out.append(pd.DataFrame(w, index=mkt.dates[test], columns=mkt.assets))
        if hasattr(strat, "last_scores"):
            scores.append(pd.DataFrame(strat.last_scores, index=mkt.dates[test], columns=mkt.assets))
        if verbose:
            print(f"    fit on {len(train)} weeks -> {mkt.dates[test[0]].date()} .. {mkt.dates[test[-1]].date()}", flush=True)
    W = pd.concat(out)
    if return_scores:
        return W, (pd.concat(scores) if scores else None)
    return W
