"""Stateless signal generation: weights are a pure function of the price history up to the last row."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def _momentum(prices: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.Series:
    last = prices.iloc[-1]
    parts = [last / prices.iloc[-1 - lb] - 1 for lb in lookbacks]
    return sum(parts) / len(parts)


def single_signal(prices: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> pd.Series:
    """Target weights for one decision date from dividend/split-adjusted daily closes (last row = decision date).

    1. Score each risk asset by the mean of its 1/3/6/12-month returns.
    2. Keep the `top_k` highest scores that are positive (absolute momentum).
    3. Weight picks by inverse 63-day volatility; each pick owns at most 1/top_k of the book.
    4. Slots left empty (not enough positive-momentum assets) go to the best positive-momentum defensive asset,
       else stay in cash. The returned weights sum to <= 1; the remainder is cash.

    Assets without enough history are ignored rather than poisoning the whole book.
    """
    need = max(max(cfg.lookbacks), cfg.vol_window) + 1
    window = prices.iloc[-need:]
    usable = [s for s in cfg.symbols if s in prices.columns and len(window) >= need and window[s].notna().all()]

    weights = pd.Series(0.0, index=list(cfg.symbols))
    risk = [s for s in cfg.risk_universe if s in usable]
    if not risk:
        return weights

    mom = _momentum(prices[usable], cfg.lookbacks)
    vol = prices[risk].pct_change().iloc[-cfg.vol_window :].std() * np.sqrt(252)

    eligible = mom[risk][mom[risk] > 0].sort_values(ascending=False)
    picks = list(eligible.index[: cfg.top_k])
    if picks:
        inv_vol = 1.0 / vol[picks].replace(0, np.nan)
        base = (inv_vol / inv_vol.sum()).fillna(1.0 / len(picks))
        weights[picks] = base * len(picks) / cfg.top_k

    idle = cfg.top_k - len(picks)
    if idle > 0:
        defensive = [s for s in cfg.defensive if s in usable]
        if defensive:
            dm = mom[defensive]
            dm = dm[dm > 0].sort_values(ascending=False)
            if len(dm):
                weights[dm.index[0]] += idle / cfg.top_k
    return weights


def decision_dates(index: pd.DatetimeIndex, n_weeks: int) -> list[pd.Timestamp]:
    """Last bar plus the last trading day of each of the previous `n_weeks - 1` ISO weeks (oldest first)."""
    iso = index.isocalendar()
    key = pd.Series(list(zip(iso.year.values, iso.week.values)), index=index)
    last_per_week = pd.Series(index, index=index).groupby(key.values, sort=False).last()
    prior = [pd.Timestamp(d) for d in last_per_week.values[:-1]][-(n_weeks - 1) :] if n_weeks > 1 else []
    return [*prior, index[-1]]


def target_weights(prices: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> pd.Series:
    """Final portfolio weights (including the cash proxy) for the last row of `prices`.

    The book is the average of the signal on the decision date and on the last trading day of the preceding
    weeks. This removes the dependence on which weekday a monthly rebalance happens to land on (research: Sharpe
    ranged 0.51-0.75 across rebalance days) and cuts turnover versus a raw weekly rotation.
    """
    prices = prices.sort_index()
    if len(prices) < cfg.min_history:
        raise ValueError(f"need >= {cfg.min_history} daily bars, got {len(prices)}")
    dates = decision_dates(prices.index, cfg.smoothing_weeks)
    signals = [single_signal(prices.loc[:d], cfg) for d in dates]
    w = sum(signals) / len(signals)
    w[cfg.cash] = max(0.0, 1.0 - float(w.sum()))
    return w
