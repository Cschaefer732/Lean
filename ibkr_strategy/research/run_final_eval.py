from __future__ import annotations

import numpy as np
import pandas as pd

from engine import CASH, CORE, backtest, fmt, load, month_ends, stats
from strategies import buy_hold, rotation, sma_trend

rets, px = load(CORE + [CASH])
cash_r = rets[CASH]
START = "2007-01-01"
DEF = ["IEF", "TLT", "GLD"]

sig = rotation(px, rets, CORE, freq="W", top_k=5, defensive=DEF)
rot = sum(sig.shift(j).fillna(0) for j in range(4)).iloc[3:] / 4
faber = sma_trend(px, rets, CORE, 210)

print("=== blend with slow Faber trend sleeve (target-weight blend) ===")
for a in (1.0, 0.8, 0.6, 0.5):
    idx = rot.index.union(faber.index)
    tgt = a * rot.reindex(idx).ffill().fillna(0) + (1 - a) * faber.reindex(idx).ffill().fillna(0)
    bt = backtest(tgt, rets, cash_rets=cash_r)
    print(fmt(f"{int(a*100)}% rotation / {int((1-a)*100)}% faber", stats(bt["ret"].loc[START:], cash_r), bt["turnover"].loc[START:].sum() / (len(bt.loc[START:]) / 252)))

print("\n=== leverage (applied to target weights; borrow = SHY + 1.5%/yr) ===")
for L in (1.0, 1.25, 1.5, 2.0):
    bt = backtest(rot * L, rets, cash_rets=cash_r, borrow_bps=150)
    print(fmt(f"leverage {L:.2f}x", stats(bt["ret"].loc[START:], cash_r)))

bt = backtest(rot, rets, cash_rets=cash_r)
spy = backtest(buy_hold(["SPY"], px), rets, cash_rets=cash_r)
sixty = backtest(buy_hold(["SPY", "IEF"], px, {"SPY": 0.6, "IEF": 0.4}), rets, cash_rets=cash_r)

print("\n=== headline (2007-01 -> 2026-10, 1-day lag, 5bps/turnover) ===")
for n, b in (("STRATEGY", bt), ("SPY", spy), ("60/40", sixty)):
    print(fmt(n, stats(b["ret"].loc[START:], cash_r)))

print("\n=== calendar years (%) ===")
yr = pd.DataFrame({n: b["ret"].loc[START:].groupby(b.loc[START:].index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
                   for n, b in (("STRAT", bt), ("SPY", spy), ("60/40", sixty))}).round(1)
print(yr.T.to_string())
print(f"strategy beat SPY in {int((yr.STRAT > yr.SPY).sum())}/{len(yr)} years; had a down year in {int((yr.STRAT < 0).sum())}/{len(yr)} (SPY {int((yr.SPY < 0).sum())})")

print("\n=== worst drawdown episodes ===")
for n, b in (("STRAT", bt), ("SPY", spy)):
    eq = (1 + b["ret"].loc[START:]).cumprod()
    dd = eq / eq.cummax() - 1
    ep, start = [], None
    for d, v in dd.items():
        if v < 0 and start is None:
            start = d
        if v == 0 and start is not None:
            seg = dd.loc[start:d]
            ep.append((seg.min(), seg.idxmin(), start, d)); start = None
    if start is not None:
        seg = dd.loc[start:]
        ep.append((seg.min(), seg.idxmin(), start, None))
    print(n, [(f"{m*100:.1f}%", str(t.date())) for m, t, _, _ in sorted(ep)[:4]])

print("\n=== block-bootstrap (21d blocks, 5000 resamples) of Sharpe difference vs SPY and 60/40 ===")
rng = np.random.default_rng(7)
a = (bt["ret"].loc[START:] - cash_r.loc[START:]).values
for n, b in (("SPY", spy), ("60/40", sixty)):
    o = (b["ret"].loc[START:] - cash_r.loc[START:]).values
    T, blk = len(a), 21
    nb = int(np.ceil(T / blk))
    diffs, dd_win = [], 0
    for _ in range(5000):
        st = rng.integers(0, T - blk, nb)
        ix = np.concatenate([np.arange(s, s + blk) for s in st])[:T]
        sa, so = a[ix], o[ix]
        diffs.append(sa.mean() / sa.std() * np.sqrt(252) - so.mean() / so.std() * np.sqrt(252))
    diffs = np.array(diffs)
    print(f"Sharpe(strat) - Sharpe({n}): mean {diffs.mean():.2f}, 90% CI [{np.percentile(diffs,5):.2f}, {np.percentile(diffs,95):.2f}], P(>0)={np.mean(diffs>0):.2f}")
