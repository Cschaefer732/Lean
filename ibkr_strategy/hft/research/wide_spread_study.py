"""Can a wider-spread name make market making survive IBKR fees and latency? (IBM, ~3 tick spread in this sample.)

Same protocol as mm_study: alpha and thresholds from Oct 7-9, results on Oct 10-11, IBKR 50 msg/s cap enforced."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hft.research.mm_study as mm  # noqa: E402
from hft.data import available_days, load_day  # noqa: E402
from hft.fees import FREE, IBKR_PRO_TIERED, PROP_REBATE  # noqa: E402
from hft.risk import RiskLimits  # noqa: E402
from hft.strategies import InsideMM, PassiveMM  # noqa: E402

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "ibm"
mm.SYMBOL = SYMBOL
LIM = RiskLimits(max_loss=1e12)   # default 50 msg/s budget stays on


def main():
    days = available_days(SYMBOL)
    train, test = days[:3], days[3:]
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    d = load_day(SYMBOL, days[0])
    spread = np.median((d.q_ask - d.q_bid)[d.q_bid < d.q_ask]) / 0.01
    P(f"# {SYMBOL.upper()} wide-spread market making (median spread {spread:.0f} ticks, {len(d.q_t):,} NBBO events/day)\n")
    alpha = mm.fit_alpha(train)
    P(f"alpha {dict(zip(['imb', 'flow_100ms', 'flow_1s'], alpha.w.round(4)))}\n")

    P("## train: choose pull threshold (fee-free, 0 latency)")
    best, best_p = None, -1e18
    for pull in (None, 0.03, 0.06, 0.1, 0.2):
        a = mm.agg(mm.run(train, lambda: InsideMM(alpha=alpha, pull=pull), fees=FREE, limits=LIM))
        P(f"pull={pull}:", mm.fmt(a))
        if a["pnl"] > best_p:
            best, best_p = pull, a["pnl"]
    P(f"selected pull = {best}\n")

    strategies = {
        "touch MM (naive)": lambda **k: PassiveMM(**k),
        "inside MM (naive)": lambda **k: InsideMM(**k),
        "inside MM + alpha pull": lambda **k: InsideMM(alpha=alpha, pull=best, **k),
    }
    P("## test days: fees x latency (min requote 250 ms whenever latency > 0, to respect IBKR's message budget)")
    for fee in (FREE, PROP_REBATE, IBKR_PRO_TIERED):
        P(f"\n### fees: {fee.name}")
        for L in (0, 50, 250):
            for name, f in strategies.items():
                res = mm.run(test, lambda: f(min_requote_ms=250 if L else 0), fees=fee, md_latency_ms=L, order_latency_ms=L, limits=LIM)
                P(f"L={L:>4}ms {name:24s}", mm.fmt(mm.agg(res)))
    Path(__file__).resolve().parents[1].joinpath("reports", f"wide_spread_{SYMBOL}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
