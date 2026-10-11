"""Cross-check the LEAN backtest against the research engine using LEAN's exact timing (decide + fill at Thursday's close = engine lag 1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import CASH, CORE, backtest, fmt, load, stats, week_ends  # noqa: E402
from momentum_rotation import StrategyConfig, target_weights  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "lean" / "results" / "IBKRMomentumRotation.json"
cfg = StrategyConfig()
rets, px = load(list(cfg.symbols))

lean = json.load(open(RESULTS))
eq = lean["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
lean_eq = pd.Series({pd.to_datetime(v[0], unit="s").normalize(): v[-1] if isinstance(v, list) else v["y"] for v in eq})
# LEAN stamps each daily sample at the following midnight; map to the trading day whose close it reflects.
prev_day = px.index.searchsorted(lean_eq.index, side="left") - 1
lean_eq = pd.Series(lean_eq.values, index=px.index[prev_day.clip(0)])
lean_eq = lean_eq[~lean_eq.index.duplicated(keep="last")]
lean_ret = lean_eq.pct_change().dropna()

we = week_ends(px.index)
pos = px.index.get_indexer(we)
thursdays = pd.DatetimeIndex([px.index[p - 1] for p in pos if p >= 1])
thursdays = thursdays[[px.index.get_loc(d) + 1 >= cfg.min_history for d in thursdays]]
tgt = pd.DataFrame({d: target_weights(px.loc[:d], cfg) for d in thursdays}).T * 0.99
res = backtest(tgt, rets, cost_bps=0, lag=1, cash_rets=None)
mine = res["ret"].reindex(lean_ret.index).dropna()

both = pd.concat([mine, lean_ret], axis=1, keys=["research", "lean"], sort=True).dropna()
print(f"overlap days {len(both)}  daily-return correlation {both.corr().iloc[0, 1]:.4f}")
for name, r in (("research engine (LEAN timing, 0 cost, cash=0)", both.research), ("LEAN IBKR model", both.lean)):
    print(fmt(name, stats(r)))
w = both.resample("W").apply(lambda x: (1 + x).prod() - 1)
print(f"weekly-return correlation {w.corr().iloc[0, 1]:.4f}")

print("\ndaily corr at relative shifts of the LEAN series (0 = aligned):")
for sh in (-2, -1, 0, 1, 2):
    c = pd.concat([mine, lean_ret.shift(sh)], axis=1, sort=True).dropna().corr().iloc[0, 1]
    print(f"  shift {sh:+d}: {c:.4f}")

diff = (both.research - both.lean)
print("\nlargest daily gaps (research - lean):")
print(diff.reindex(diff.abs().sort_values(ascending=False).index).head(8).round(4).to_string())
print("share of days |diff| > 0.5%:", round((diff.abs() > 0.005).mean(), 3))
by_dow = diff.abs().groupby(diff.index.dayofweek).mean().round(5)
print("mean |diff| by weekday (0=Mon):", by_dow.to_dict())
