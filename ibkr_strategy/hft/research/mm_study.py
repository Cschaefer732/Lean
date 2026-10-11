"""Passive market making on real SPY ticks: alpha value, latency decay, fee sensitivity, queue-model sensitivity.

Protocol: alpha and pull threshold are fitted/selected on Oct 7-9 only; Oct 10-11 are untouched until the final tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hft.data import available_days, load_day  # noqa: E402
from hft.fees import FREE, IBKR_PRO_TIERED, PROP_REBATE  # noqa: E402
from hft.features import FEATURES, build, on_grid  # noqa: E402
from hft.replay import Replay, ReplayConfig  # noqa: E402
from hft.risk import RiskLimits  # noqa: E402
from hft.signals import LinearAlpha, SIGNAL_FEATURES  # noqa: E402
from hft.strategies import PassiveMM  # noqa: E402

SYMBOL = "spy"
NOKILL = RiskLimits(max_loss=1e12)
BLOCK_MS = 30 * 60_000


def fit_alpha(days):
    cols = [FEATURES.index(f) for f in SIGNAL_FEATURES]
    X, y = [], []
    for d in days:
        g = on_grid(build(load_day(SYMBOL, d)))
        X.append(g["X"][:, cols])
        y.append(g["y"]["1s"])
    return LinearAlpha.fit(np.vstack(X), np.concatenate(y))


def run(days, make_strategy, **cfg):
    cfg.setdefault("limits", NOKILL)
    out = []
    for d in days:
        out.append(Replay(load_day(SYMBOL, d), make_strategy(), ReplayConfig(**cfg)).run())
    return out


def agg(results):
    shares = sum(sum(f.qty for f in r.fills) for r in results)
    pnl = sum(r.pnl_liq for r in results)
    mo = {k: np.mean([r.markout.get(k, np.nan) for r in results]) for k in ("spread_capture", "markout_1s", "markout_5s")}
    return {"pnl": pnl, "shares": shares, "c_per_sh": 100 * pnl / shares if shares else float("nan"),
            "fills": sum(len(r.fills) for r in results), "pos_abs": np.mean([r.pos_abs_mean for r in results]), **mo}


def blocks(results):
    """30-minute PnL increments (mark-to-mid, net of fees) concatenated across days."""
    out = []
    for r in results:
        if len(r.pnl_t) < 2:
            continue
        edges = np.arange(r.pnl_t[0], r.pnl_t[-1] + BLOCK_MS, BLOCK_MS)
        idx = np.clip(np.searchsorted(r.pnl_t, edges), 0, len(r.pnl_path) - 1)
        out.extend(np.diff(r.pnl_path[idx]))
    return np.array(out)


def paired_bootstrap(a, b, n=5000, seed=0):
    n_blocks = min(len(a), len(b))
    d = (a[:n_blocks] - b[:n_blocks])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, n_blocks).mean() for _ in range(n)])
    return d.mean(), float(np.mean(means <= 0))


def fmt(a):
    return (f"pnl ${a['pnl']:+9.2f} | {a['fills']:6d} fills {a['shares']:8d} sh | {a['c_per_sh']:+7.3f} c/sh | "
            f"capture {a['spread_capture']:+.3f} markout1s {a['markout_1s']:+.3f} 5s {a['markout_5s']:+.3f} ticks | mean|pos| {a['pos_abs']:.0f}")


def main():
    days = available_days(SYMBOL)
    train, test = days[:3], days[3:]
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    alpha = fit_alpha(train)
    P(f"# SPY passive market making (train {train[0]}..{train[-1]}, test {test[0]}..{test[-1]})\n")
    P(f"alpha weights {dict(zip(SIGNAL_FEATURES, alpha.w.round(4)))} intercept {alpha.b:+.4f}\n")

    # 1. choose pull threshold on TRAIN days, gross of fees, zero latency
    P("## 1. Pull-threshold selection (train days, fee-free, 0 latency)")
    best, best_pnl = None, -1e18
    for pull in (None, 0.03, 0.06, 0.10, 0.15, 0.25):
        a = agg(run(train, lambda: PassiveMM(alpha=alpha, pull=pull), fees=FREE))
        P(f"pull={pull}:", fmt(a))
        if a["pnl"] > best_pnl:
            best, best_pnl = pull, a["pnl"]
    P(f"\nselected pull = {best}\n")

    mk = {
        "naive (always quote)": lambda: PassiveMM(),
        "alpha pull": lambda: PassiveMM(alpha=alpha, pull=best),
        "shuffled-alpha control": lambda: PassiveMM(alpha=alpha, pull=best, shuffle_seed=1),
    }

    # 2. latency decay, test days
    P("## 2. Latency sweep (test days, prop-rebate fees). One-way latency applies to market data AND to each order/ack leg")
    lats = (0, 0.1, 1, 5, 20, 50, 100, 250, 1000)
    table = {}
    for L in lats:
        for name, f in mk.items():
            res = run(test, f, fees=PROP_REBATE, md_latency_ms=L, order_latency_ms=L)
            table[(L, name)] = (agg(res), blocks(res))
            P(f"L={L:>6}ms {name:24s}", fmt(table[(L, name)][0]))
        P("")

    # 3. paired bootstrap of alpha vs naive and vs shuffled at a low latency
    P("## 3. Does the alpha add value? paired block bootstrap over 30-minute PnL blocks (prop-rebate fees)")
    for L in (0, 1, 20):
        for other in ("naive (always quote)", "shuffled-alpha control"):
            res_a = run(test, mk["alpha pull"], fees=PROP_REBATE, md_latency_ms=L, order_latency_ms=L)
            res_o = run(test, mk[other], fees=PROP_REBATE, md_latency_ms=L, order_latency_ms=L)
            d, p = paired_bootstrap(blocks(res_a), blocks(res_o))
            P(f"L={L}ms alpha-pull minus {other:24s}: mean block pnl diff ${d:+.2f}, bootstrap P(diff<=0) = {p:.3f}  (n_blocks={min(len(blocks(res_a)), len(blocks(res_o)))})")
    P("")

    # 4. fees
    P("## 4. Fee sensitivity (test days, alpha pull, L=1ms)")
    for fee in (FREE, PROP_REBATE, IBKR_PRO_TIERED):
        P(f"{fee.name:18s}", fmt(agg(run(test, mk["alpha pull"], fees=fee, md_latency_ms=1, order_latency_ms=1))))
    P("")

    # 5. queue assumptions
    P("## 5. Queue-model sensitivity (test days, alpha pull, prop-rebate, L=1ms)")
    for qm, pess in (("prob", 0.5), ("prob", 1.0), ("prob", 2.0), ("none", 1.0), ("none", 2.0)):
        P(f"queue={qm:4s} pessimism={pess}:", fmt(agg(run(test, mk["alpha pull"], fees=PROP_REBATE, md_latency_ms=1, order_latency_ms=1,
                                                        queue_model=qm, queue_pessimism=pess))))
    Path(__file__).resolve().parents[1].joinpath("reports").mkdir(exist_ok=True)
    Path(__file__).resolve().parents[1].joinpath("reports", "mm_study.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
