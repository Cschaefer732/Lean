"""CLI: run a market-making strategy (or just record ticks) against IB Gateway / TWS.

  python -m hft.run_live --symbol SPY --seconds 600 --record            # record only, no orders ever
  python -m hft.run_live --symbol SPY --seconds 600                     # dry-run: logs intended orders, sends nothing
  python -m hft.run_live --symbol SPY --seconds 600 --send              # sends orders, PAPER ports only
  python -m hft.run_live --symbol SPY --send --port 4001 --live-i-understand-the-risk
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .ibkr_venue import LiveVenue
from .recorder import TickRecorder
from .risk import RiskLimits
from .signals import LinearAlpha
from .strategies import PassiveMM

LIVE_PORTS = {7496, 4001}
PAPER_PORTS = {7497, 4002}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=31)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--send", action="store_true", help="actually place orders (default is dry-run)")
    p.add_argument("--live-i-understand-the-risk", action="store_true")
    p.add_argument("--record", action="store_true", help="record ticks to hft/cache/live and place NO orders")
    p.add_argument("--alpha", type=Path, help='json {"w": [imb, flow_100ms, flow_1s], "b": 0.0, "pull": 0.06}; refit on recorded data')
    p.add_argument("--qty", type=int, default=100)
    p.add_argument("--max-pos", type=int, default=200)
    p.add_argument("--max-loss", type=float, default=100.0)
    p.add_argument("--stale-feed-s", type=float, default=2.0, help="pull all orders if no quote arrives for this long in-session")
    p.add_argument("--min-requote-ms", type=int, default=250, help="keep cancel/replace well inside IBKR's 50 msg/s")
    return p.parse_args(argv)


def build_strategy(a):
    alpha = pull = None
    if a.alpha:
        cfg = json.loads(a.alpha.read_text())
        alpha, pull = LinearAlpha(np.array(cfg["w"]), cfg.get("b", 0.0)), cfg.get("pull", 0.06)
    return PassiveMM(qty=a.qty, max_pos=a.max_pos, alpha=alpha, pull=pull, min_requote_ms=a.min_requote_ms)


def main(argv=None, ib_factory=None) -> int:
    a = parse(argv)
    if a.send and a.port in LIVE_PORTS and not a.live_i_understand_the_risk:
        print(f"refusing: port {a.port} is a LIVE port; pass --live-i-understand-the-risk or use a paper port {sorted(PAPER_PORTS)}")
        return 2
    if a.record and a.send:
        print("refusing: --record never sends orders")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if ib_factory is None:
        from ib_async import IB, Stock
        ib_factory = IB
    else:
        Stock = lambda s, e, c: type("C", (), {"symbol": s})()  # noqa: E731
    ib = ib_factory()
    ib.connect(a.host, a.port, clientId=a.client_id, readonly=not a.send, timeout=15)
    try:
        contract = Stock(a.symbol, "SMART", "USD")
        recorder = TickRecorder(a.symbol, dt.date.today().strftime("%Y%m%d")) if a.record else None
        strat = build_strategy(a)
        if a.record:
            from .core import Strategy
            strat = Strategy()
        venue = LiveVenue(ib, contract, strat, dry_run=not a.send, recorder=recorder, stale_feed_s=a.stale_feed_s,
                          reconnect=lambda: ib.connect(a.host, a.port, clientId=a.client_id, readonly=not a.send, timeout=15),
                          limits=RiskLimits(max_position=a.max_pos, max_order_qty=a.qty, max_loss=a.max_loss))
        venue.start()
        venue.run(a.seconds)
        print(venue.lat.report())
        print("risk rejections:", venue.risk.state.rejected, "halted:", venue.risk.state.halt_reason or "no")
        print("reconciliation:", venue.recon)
        if recorder:
            print("recorded to", recorder.save(Path(__file__).parent / "cache" / "live"))
        if not a.send:
            print(f"dry-run intents: {len(venue.intents)}")
    finally:
        ib.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
