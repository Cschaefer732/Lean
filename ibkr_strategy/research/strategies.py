"""Research versions of candidate strategies. Each returns a target-weight DataFrame on decision dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import month_ends, week_ends


def momentum(px: pd.DataFrame, lookbacks=(21, 63, 126, 252)) -> pd.DataFrame:
    return sum(px / px.shift(lb) - 1 for lb in lookbacks) / len(lookbacks)


def realized_vol(rets: pd.DataFrame, win: int = 63) -> pd.DataFrame:
    return rets.rolling(win).std() * np.sqrt(252)


def buy_hold(universe: list[str], px: pd.DataFrame, weights: dict[str, float] | None = None, freq="M"):
    dates = month_ends(px.index)
    w = pd.DataFrame(0.0, index=dates, columns=px.columns)
    for s in universe:
        w[s] = 1 / len(universe) if weights is None else weights[s]
    return w


def sma_trend(px, rets, universe, sma_days=210, freq="M", cash=None):
    """Faber-style: equal weight slots; slot idle (cash) when asset below its SMA."""
    dates = month_ends(px.index)
    sma = px[universe].rolling(sma_days).mean()
    on = (px[universe] > sma).reindex(dates)
    w = pd.DataFrame(0.0, index=dates, columns=px.columns)
    w[universe] = on.astype(float) / len(universe)
    return w.iloc[:]


def rotation(
    px: pd.DataFrame,
    rets: pd.DataFrame,
    universe: list[str],
    top_k: int = 4,
    lookbacks=(21, 63, 126, 252),
    abs_filter: bool = True,
    weighting: str = "invvol",
    target_vol: float | None = None,
    max_lev: float = 1.0,
    vol_win: int = 63,
    freq: str = "M",
    trend_sma: int | None = None,
    dates=None,
    buffer: int = 0,
    defensive: list[str] | None = None,
):
    """Cross-sectional + absolute momentum rotation with optional inverse-vol weights and portfolio vol targeting."""
    if dates is None:
        dates = month_ends(px.index) if freq == "M" else week_ends(px.index)
    mom = momentum(px[universe], lookbacks)
    vol = realized_vol(rets[universe], vol_win)
    sma = px[universe].rolling(trend_sma).mean() if trend_sma else None
    defensive_mom = momentum(px[defensive], lookbacks) if defensive else None
    rows = {}
    held: list[str] = []
    for d in dates:
        if d not in mom.index or mom.loc[d].isna().any() or vol.loc[d].isna().any():
            continue
        m = mom.loc[d]
        eligible = m[m > 0] if abs_filter else m
        if sma is not None:
            eligible = eligible[px.loc[d, eligible.index] > sma.loc[d, eligible.index]]
        ranked = eligible.sort_values(ascending=False)
        picks = ranked.head(top_k).index
        if buffer and held:
            keep = [h for h in held if h in ranked.index[: top_k + buffer]]
            fill = [x for x in ranked.index if x not in keep]
            picks = pd.Index((keep + fill)[:top_k])
        held = list(picks)
        w = pd.Series(0.0, index=px.columns)
        if len(picks):
            if weighting == "invvol":
                iv = 1 / vol.loc[d, picks]
                base = iv / iv.sum()
            else:
                base = pd.Series(1 / len(picks), index=picks)
            # slots: unfilled slots stay in cash so a risk-off regime de-grosses automatically
            w[picks] = base * len(picks) / top_k
        if defensive and len(picks) < top_k:
            dm = defensive_mom.loc[d]
            best = dm[dm > 0].sort_values(ascending=False).head(1).index
            if len(best):
                w[best[0]] += (top_k - len(picks)) / top_k
        if target_vol and w.sum() > 0:
            hist = rets[universe].loc[:d].tail(vol_win)
            pv = float(np.sqrt(w[universe].values @ (hist.cov().values * 252) @ w[universe].values))
            if pv > 0:
                w = w * min(target_vol / pv, max_lev / w.sum())
        rows[d] = w
    return pd.DataFrame(rows).T
