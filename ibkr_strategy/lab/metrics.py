"""Performance, signal-quality and multiple-testing statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

from .data import CASH, Market, engine
from .construct import smooth


def backtest_weights(W: pd.DataFrame, mkt: Market, smooth_weeks: int = 4, cost_bps: float = 5.0, lag: int = 1) -> pd.DataFrame:
    """Weekly risk-asset weights -> daily returns through the shared engine (drift, costs, execution lag, SHY cash)."""
    Ws = pd.DataFrame(smooth(W.values, smooth_weeks), index=W.index, columns=W.columns)
    return engine.backtest(Ws, mkt.rets, cost_bps=cost_bps, lag=lag, cash_rets=mkt.rets[CASH])


def summarize(ret: pd.Series, rf: pd.Series, turnover: pd.Series | None = None) -> dict:
    s = engine.stats(ret, rf)
    if turnover is not None and s:
        s["TO/yr"] = turnover.sum() / (len(ret) / 252)
    return s


def information_coefficient(scores: pd.DataFrame, mkt: Market) -> dict:
    """Per-date Spearman rank correlation between predicted and realised excess return."""
    y = pd.DataFrame(mkt.y, index=mkt.dates, columns=mkt.assets).reindex(scores.index)
    ics = []
    for d in scores.index:
        a, b = scores.loc[d], y.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 6 and a[ok].std() > 0:
            ics.append(sst.spearmanr(a[ok], b[ok])[0])
    ics = np.array(ics)
    if not len(ics):
        return {}
    # overlapping 4-week labels -> effective sample is ~len/horizon
    eff = len(ics) / mkt.horizon
    return {"IC": ics.mean(), "IC_t": ics.mean() / (ics.std() / np.sqrt(eff)), "IC_hit": (ics > 0).mean()}


def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series, block: int = 21, n: int = 3000, seed: int = 7) -> dict:
    """Paired block bootstrap of Sharpe(a) - Sharpe(b)."""
    idx = a.index.intersection(b.index)
    ea, eb = (a.loc[idx] - rf.reindex(idx)).values, (b.loc[idx] - rf.reindex(idx)).values
    rng = np.random.default_rng(seed)
    T = len(idx)
    nb = int(np.ceil(T / block))
    d = np.empty(n)
    for k in range(n):
        st = rng.integers(0, T - block, nb)
        ix = np.concatenate([np.arange(s, s + block) for s in st])[:T]
        d[k] = (ea[ix].mean() / ea[ix].std() - eb[ix].mean() / eb[ix].std()) * np.sqrt(252)
    return {"diff": d.mean(), "lo": np.percentile(d, 5), "hi": np.percentile(d, 95), "p_gt0": (d > 0).mean()}


def deflated_sharpe(ret: pd.Series, rf: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Bailey & Lopez de Prado (2014): probability the true Sharpe is > 0 after accounting for having tried
    len(trial_sharpes) strategies (best-of-N selection bias), non-normal returns and sample length.
    `trial_sharpes` are the per-period (daily) Sharpe ratios of all strategies tried."""
    r = (ret - rf.reindex(ret.index).fillna(0)).dropna().values
    T = len(r)
    sr = r.mean() / r.std()
    skew, kurt = sst.skew(r), sst.kurtosis(r, fisher=False)
    n = max(len(trial_sharpes), 2)
    g = 0.5772156649
    sr0 = np.std(trial_sharpes, ddof=1) * ((1 - g) * sst.norm.ppf(1 - 1 / n) + g * sst.norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(sst.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def daily_sharpe(ret: pd.Series, rf: pd.Series) -> float:
    r = (ret - rf.reindex(ret.index).fillna(0)).dropna()
    return float(r.mean() / r.std())
