from __future__ import annotations

import itertools

import pandas as pd

from engine import CASH, CORE, backtest, load, stats
from strategies import rotation

rets, px = load(CORE + [CASH])
cash_r = rets[CASH]
IS = slice("2007-01-01", "2015-12-31")
OOS = slice("2016-01-01", None)

LOOKBACKS = {
    "1-3-6-12": (21, 63, 126, 252),
    "3-6-12": (63, 126, 252),
    "1-3-6": (21, 63, 126),
    "6-12": (126, 252),
    "12": (252,),
    "3": (63,),
    "1-3-6-9-12": (21, 63, 126, 189, 252),
}
rows = []
for (lbn, lb), k, wt, freq, vw in itertools.product(
    LOOKBACKS.items(), (2, 3, 4, 5, 6), ("invvol", "ew"), ("M", "W"), (63, 126)
):
    if wt == "ew" and vw == 126:
        continue
    tgt = rotation(px, rets, CORE, top_k=k, lookbacks=lb, weighting=wt, vol_win=vw, freq=freq)
    bt = backtest(tgt, rets, cash_rets=cash_r)
    si, so = stats(bt["ret"].loc[IS], cash_r), stats(bt["ret"].loc[OOS], cash_r)
    rows.append(
        dict(lb=lbn, k=k, w=wt, freq=freq, vw=vw,
             IS_sh=si["Sharpe"], OOS_sh=so["Sharpe"], IS_cagr=si["CAGR"], OOS_cagr=so["CAGR"],
             IS_dd=si["MaxDD"], OOS_dd=so["MaxDD"], IS_cal=si["Calmar"], OOS_cal=so["Calmar"],
             TO=bt["turnover"].sum() / (len(bt) / 252))
    )
df = pd.DataFrame(rows)
df.to_csv("grid_results.csv", index=False)
pd.set_option("display.width", 250)
print("n configs", len(df))
print("\nIS-ranked top 12 (what a researcher would pick) and their OOS:")
print(df.sort_values("IS_sh", ascending=False).head(12).round(3).to_string(index=False))
print("\nCorrelation IS vs OOS Sharpe:", round(df.IS_sh.corr(df.OOS_sh), 3))
print("\nMean by factor (IS_sh / OOS_sh / TO):")
for c in ("lb", "k", "w", "freq", "vw"):
    print(df.groupby(c)[["IS_sh", "OOS_sh", "IS_dd", "OOS_dd", "TO"]].mean().round(3), "\n")
