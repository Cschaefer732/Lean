from __future__ import annotations

import pandas as pd

from engine import CASH, CORE, backtest, fmt, load, stats
from strategies import buy_hold, rotation, sma_trend

rets, px = load(CORE + [CASH])
cash_r = rets[CASH]
IS = slice("2007-01-01", "2015-12-31")
OOS = slice("2016-01-01", None)
FULL = slice("2007-01-01", None)

print(f"data {px.index[0].date()} -> {px.index[-1].date()}  ({len(px)} days)\n")


def report(name, target, **kw):
    bt = backtest(target, rets, cash_rets=cash_r, **kw)
    out = []
    for lab, sl in (("FULL", FULL), ("IS  ", IS), ("OOS ", OOS)):
        r = bt["ret"].loc[sl]
        to = bt["turnover"].loc[sl].sum() / (len(r) / 252)
        out.append(fmt(f"{name} [{lab}]", stats(r, cash_r), to))
    print("\n".join(out), "\n")
    return bt


report("SPY buy&hold", buy_hold(["SPY"], px))
report("60/40 SPY/IEF", buy_hold(["SPY", "IEF"], px, {"SPY": 0.6, "IEF": 0.4}))
report("EW 12 assets", buy_hold(CORE, px))
report("Faber 10m SMA EW12 (cash=SHY)", sma_trend(px, rets, CORE, 210))
report("Rotation top4 invvol, abs filter", rotation(px, rets, CORE, top_k=4))
report("Rotation top4 EW, abs filter", rotation(px, rets, CORE, top_k=4, weighting="ew"))
report("Rotation top6 invvol, abs filter", rotation(px, rets, CORE, top_k=6))
report("Rotation top4 invvol +10%volT", rotation(px, rets, CORE, top_k=4, target_vol=0.10, max_lev=1.0))
report("Rotation top4 invvol +10%volT lev1.5", rotation(px, rets, CORE, top_k=4, target_vol=0.10, max_lev=1.5))
