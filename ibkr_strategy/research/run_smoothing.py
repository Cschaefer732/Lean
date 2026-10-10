from __future__ import annotations

import pandas as pd

from engine import CASH, CORE, backtest, fmt, load, stats, week_ends
from strategies import rotation

rets, px = load(CORE + [CASH])
cash_r = rets[CASH]
IS = slice("2007-01-01", "2015-12-31")
OOS = slice("2016-01-01", None)
DEF = ["IEF", "TLT", "GLD"]


def smoothed(n_signals: int, step: int = 1, **kw) -> pd.DataFrame:
    """Target = mean of the last n weekly signals (spaced `step` weeks apart)."""
    sig = rotation(px, rets, CORE, freq="W", top_k=5, defensive=DEF, **kw)
    parts = [sig.shift(step * j) for j in range(n_signals)]
    return sum(p.fillna(0) for p in parts).iloc[step * (n_signals - 1) :] / n_signals


def show(name, tgt, **bt_kw):
    bt = backtest(tgt, rets, cash_rets=cash_r, **bt_kw)
    for lab, sl in (("IS ", IS), ("OOS", OOS), ("ALL", slice("2007-01-01", None))):
        r = bt["ret"].loc[sl]
        print(fmt(f"{name} [{lab}]", stats(r, cash_r), bt["turnover"].loc[sl].sum() / (len(r) / 252)))
    return bt


print("=== average of last N weekly signals (weekly rebalance of account) ===")
for n in (1, 2, 4, 6, 8, 13):
    show(f"N={n:2d} weekly signals", smoothed(n))
print()
print("=== N=4 with different parameters ===")
show("N=4, lb 3-6-12", smoothed(4, lookbacks=(63, 126, 252)))
show("N=4, lb 1-3-6-12 vw126", smoothed(4, vol_win=126))
show("N=4, ew weights", smoothed(4, weighting="ew"))
print()
print("=== N=4 lag sensitivity ===")
for lag in (0, 1, 2):
    show(f"N=4 lag{lag}", smoothed(4), lag=lag)
