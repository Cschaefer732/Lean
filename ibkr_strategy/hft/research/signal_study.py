"""Do L1 order-book / trade-flow features predict short-horizon mid-price moves on real data, out of sample by day?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hft.data import available_days, load_day  # noqa: E402
from hft.features import FEATURES, HORIZONS_MS, TICK, build, on_grid  # noqa: E402


def ic(a, b):
    return sst.spearmanr(a, b)[0]


def study(symbol: str, horizon: str = "1s", verbose=True):
    days = available_days(symbol)
    raw = {d: build(load_day(symbol, d)) for d in days}
    data = {d: on_grid(raw[d]) for d in days}
    train, test = days[:3], days[3:]
    out = {"symbol": symbol, "n_test": sum(len(data[d]["t"]) for d in test)}
    if verbose:
        print(f"\n=== {symbol.upper()}  train {train[0]}..{train[-1]}  test {test[0]}..{test[-1]}  horizon {horizon} ===")
        print(f"grid samples/day ~{np.mean([len(data[d]['t']) for d in days]):,.0f}; median spread {np.median(np.concatenate([data[d]['X'][:, 1] for d in days])):.0f} tick(s)")

    event_ic = ic(np.concatenate([raw[d]["X"][:, 0] for d in days[3:]]), np.concatenate([raw[d]["y"][horizon] for d in days[3:]]))
    out["event_time_imb_ic"] = event_ic
    if verbose:
        print(f"(imbalance IC if sampled in EVENT time instead of a 100ms grid: {event_ic:+.3f} -> inflated by quote bursts)")
    Xte = np.vstack([data[d]["X"] for d in test])
    yte = np.concatenate([data[d]["y"][horizon] for d in test])
    # univariate IC and the economics of the most extreme signal buckets
    rows = []
    for j, f in enumerate(FEATURES):
        rows.append((f, ic(Xte[:, j], yte)))
    out["univariate"] = rows
    if verbose:
        print("univariate OOS rank IC with forward mid move:")
        for f, v in sorted(rows, key=lambda r: -abs(r[1])):
            print(f"  {f:12s} {v:+.3f}")

    # model: gradient boosting on the first 3 days, evaluated on the last 2
    from lightgbm import LGBMRegressor

    Xtr = np.vstack([data[d]["X"] for d in train])
    ytr = np.concatenate([data[d]["y"][horizon] for d in train])
    sub = np.arange(len(ytr))
    m = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=500, subsample=0.7,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, random_state=0, verbose=-1, n_jobs=4)
    m.fit(Xtr[sub], np.clip(ytr[sub], -5, 5))
    p = m.predict(Xte)
    out["gbm_ic"] = ic(p, yte)
    out["gbm_r2"] = 1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)
    q = np.quantile(p, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    spread = Xte[:, 1]
    top = p >= q[-2]
    bot = p <= q[1]
    out["top5_move"] = float(yte[top].mean())
    out["bot5_move"] = float(yte[bot].mean())
    out["top1_move"] = float(yte[p >= q[-1]].mean())
    out["spread_ticks_top5"] = float(spread[top].mean())
    out["imp"] = sorted(zip(FEATURES, m.feature_importances_), key=lambda r: -r[1])[:5]
    if verbose:
        print(f"GBM OOS: rank IC {out['gbm_ic']:+.3f}  R2 {out['gbm_r2']:+.4f}")
        print(f"mean realised {horizon} mid move when model is in its top 5% / bottom 5% / top 1%: "
              f"{out['top5_move']:+.3f} / {out['bot5_move']:+.3f} / {out['top1_move']:+.3f} ticks")
        print(f"for scale: a taker pays at least HALF the spread ({0.5 * out['spread_ticks_top5']:.2f} tick) on entry plus half on exit, "
              f"and ~0.67 tick/share of commissions+exchange fees each way at IBKR Pro tiered rates when taking")
        print("top features:", out["imp"])
    # decay across horizons for the model-free imbalance signal
    out["decay"] = {h: ic(Xte[:, 0], np.concatenate([data[d]["y"][h] for d in test])) for h in HORIZONS_MS}
    if verbose:
        print("imbalance IC by horizon:", {h: round(v, 3) for h, v in out["decay"].items()})
    return out


if __name__ == "__main__":
    for sym in (sys.argv[1:] or ["spy", "bac", "aig", "ibm"]):
        study(sym)
