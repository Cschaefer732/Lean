"""Measure this implementation's own speed: replay throughput and the tick-to-decision time of the strategy hot path."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hft.core import Quote, Trade  # noqa: E402
from hft.data import load_day  # noqa: E402
from hft.fees import PROP_REBATE  # noqa: E402
from hft.replay import Replay, ReplayConfig  # noqa: E402
from hft.risk import RiskLimits  # noqa: E402
from hft.signals import LinearAlpha  # noqa: E402
from hft.strategies import PassiveMM  # noqa: E402


class _Ctx:
    """Minimal in-process context so the strategy's own cost is measured without simulator overhead."""
    now = 0
    position = 0
    can_trade = True
    quote = None

    def __init__(self):
        self.n = 0

    def place(self, *a, **k):
        self.n += 1
        return None

    def cancel(self, o): pass
    def cancel_all(self): pass
    def open_orders(self, side=None): return []


def main():
    lines = []

    def P(s):
        print(s, flush=True)
        lines.append(s)

    day = load_day("spy", "20131010")
    t0 = time.time()
    r = Replay(day, PassiveMM(alpha=LinearAlpha(np.array([0.175, 0.0023, 0.0])), pull=0.06),
               ReplayConfig(fees=PROP_REBATE, md_latency_ms=1, order_latency_ms=1, limits=RiskLimits(max_loss=1e12)), measure_decisions=True).run()
    n_ev = len(day.t_t) + int((day.q_bid < day.q_ask).sum())
    P(f"replay SPY 20131010: {n_ev:,} market events in {r.wall_s:.1f}s -> {n_ev / r.wall_s / 1e3:,.0f}k events/s (incl. simulator, risk gate, strategy, latency timing)")
    d = r.decision_us
    P(f"strategy callbacks measured inside the simulator (us): n={len(d):,} p50={np.percentile(d, 50):.1f} p99={np.percentile(d, 99):.1f} p99.9={np.percentile(d, 99.9):.1f} max={d.max():.0f}")

    mm = PassiveMM(alpha=LinearAlpha(np.array([0.175, 0.0023, 0.0])), pull=0.06)
    ctx = _Ctx()
    valid = np.flatnonzero(day.q_bid < day.q_ask)[200_000:260_000]
    times = np.empty(len(valid))
    for k, i in enumerate(valid):
        q = Quote(int(day.q_t[i]), day.q_bid[i], day.q_bsz[i], day.q_ask[i], day.q_asz[i])
        ctx.now, ctx.quote = q.t, q
        a = time.perf_counter_ns()
        mm.on_quote(ctx, q)
        times[k] = (time.perf_counter_ns() - a) / 1e3
    P(f"isolated PassiveMM.on_quote (features + alpha + quote logic), CPython {sys.version.split()[0]}: p50={np.percentile(times, 50):.1f}us p99={np.percentile(times, 99):.1f}us p99.9={np.percentile(times, 99.9):.1f}us")
    Path(__file__).resolve().parents[1].joinpath("reports", "bench.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
