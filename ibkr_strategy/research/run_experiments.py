from __future__ import annotations

import numpy as np
import pandas as pd

from engine import CASH, CORE, backtest, fmt, load, month_ends, stats
from strategies import rotation

EXTRA = ["VGK", "EWJ", "XLU", "XLP", "XLV"]
rets, px = load(CORE + EXTRA + [CASH])
cash_r = rets[CASH]
IS = slice("2007-01-01", "2015-12-31")
OOS = slice("2016-01-01", None)
BASE = dict(top_k=5, lookbacks=(21, 63, 126, 252), weighting="invvol", vol_win=63)


def run(name, universe=CORE, dates=None, verbose=True, **kw):
    cfg = {**BASE, **kw}
    tgt = rotation(px, rets, universe, dates=dates, **cfg)
    bt = backtest(tgt, rets, cash_rets=cash_r)
    if verbose:
        for lab, sl in (("IS ", IS), ("OOS", OOS)):
            r = bt["ret"].loc[sl]
            print(fmt(f"{name} [{lab}]", stats(r, cash_r), bt["turnover"].loc[sl].sum() / (len(r) / 252)))
    return bt


def offset_dates(offset: int) -> pd.DatetimeIndex:
    """Decision day = `offset` trading days before month end."""
    me = month_ends(px.index)
    pos = px.index.get_indexer(me)
    return pd.DatetimeIndex([px.index[p - offset] for p in pos if p - offset >= 0])


print("=== 1. rebalance-day timing luck (decision N trading days before month-end) ===")
res = []
for off in range(0, 21):
    bt = run("", dates=offset_dates(off), verbose=False)
    full = stats(bt["ret"].loc[IS.start :], cash_r)
    res.append((off, full["Sharpe"], full["CAGR"], full["MaxDD"]))
r = pd.DataFrame(res, columns=["offset", "Sharpe", "CAGR", "MaxDD"]).set_index("offset")
print(r.round(3).T.to_string())
print(f"Sharpe range {r.Sharpe.min():.2f}-{r.Sharpe.max():.2f}, CAGR range {r.CAGR.min()*100:.1f}-{r.CAGR.max()*100:.1f}%")

print("\n=== 2. tranching: average of 4 sleeves rebalanced at offsets 0/5/10/15 ===")
sleeves = [run("", dates=offset_dates(o), verbose=False) for o in (0, 5, 10, 15)]
tr = pd.DataFrame({"ret": sum(s["ret"] for s in sleeves) / 4, "turnover": sum(s["turnover"] for s in sleeves) / 4})
for lab, sl in (("IS ", IS), ("OOS", OOS)):
    rr = tr["ret"].loc[sl]
    print(fmt(f"tranched x4 [{lab}]", stats(rr, cash_r), tr["turnover"].loc[sl].sum() / (len(rr) / 252)))

print("\n=== 3. hysteresis buffer ===")
for b in (0, 1, 2, 3):
    run(f"buffer={b}", buffer=b)

print("\n=== 4. defensive fill of idle slots ===")
run("idle -> cash (SHY)")
run("idle -> best of IEF/TLT/GLD", defensive=["IEF", "TLT", "GLD"])
run("idle -> best of IEF/TIP", defensive=["IEF", "TIP"])

print("\n=== 5. universe size ===")
run("CORE12")
run("CORE12 + VGK EWJ + 3 sectors", universe=CORE + EXTRA)
run("equity+bond+gold only (8)", universe=["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ"])

print("\n=== 6. extra trend filter (price > 200d SMA) ===")
run("no extra filter")
run("+ price>200d SMA", trend_sma=200)
run("+ price>150d SMA", trend_sma=150)
run("no abs-momentum filter", abs_filter=False)

print("\n=== 7. cost sensitivity (base) ===")
tgt = rotation(px, rets, CORE, **BASE)
for c in (0, 5, 10, 25, 50):
    bt = backtest(tgt, rets, cost_bps=c, cash_rets=cash_r)
    print(fmt(f"cost {c:2d}bps/turnover FULL", stats(bt['ret'].loc[IS.start :], cash_r)))
print("\n=== 8. execution lag sensitivity ===")
for lag in (1, 2, 3):
    bt = backtest(tgt, rets, lag=lag, cash_rets=cash_r)
    print(fmt(f"lag {lag}d FULL", stats(bt["ret"].loc[IS.start :], cash_r)))
