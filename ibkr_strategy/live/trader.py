"""Weekly rebalancer for the momentum rotation through the Interactive Brokers API.

Safe by default: connects read-only and only prints the plan. Orders are transmitted only with --execute, and only to
a paper account unless --live is also given. Schedule it daily at ~15:30 America/New_York; it exits quietly unless
today is the final trading session of the week.

    python -m live.trader --port 4002                      # dry run against IB Gateway paper
    python -m live.trader --port 4002 --execute            # place MOC orders in the paper account
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.broker import Broker, IBKRBroker  # noqa: E402
from momentum_rotation import StrategyConfig, plan_orders, target_weights  # noqa: E402

ET = ZoneInfo("America/New_York")
MOC_CUTOFF = dt.time(15, 50)


class SafetyError(RuntimeError):
    pass


def validate_prices(prices: pd.DataFrame, today: dt.date, cfg: StrategyConfig) -> None:
    missing = [s for s in cfg.symbols if s not in prices.columns]
    if missing:
        raise SafetyError(f"no price history for {missing}")
    if len(prices) < cfg.min_history:
        raise SafetyError(f"only {len(prices)} bars, need {cfg.min_history}")
    stale = (pd.Timestamp(today) - prices.index[-1]).days
    if stale > 4:
        raise SafetyError(f"latest bar is {prices.index[-1].date()}, {stale} days old")
    tail = prices.iloc[-cfg.min_history :]
    short = [s for s in cfg.symbols if tail[s].isna().any()]
    if short:
        raise SafetyError(f"gaps in recent history for {short}")
    jump = (prices.iloc[-1] / prices.iloc[-2] - 1).abs()
    bad = jump[jump > 0.25]
    if len(bad):
        raise SafetyError(f"implausible one-day moves, refusing to trade: {bad.round(3).to_dict()}")


def run(
    broker: Broker,
    cfg: StrategyConfig,
    *,
    now: dt.datetime,
    state_dir: Path,
    execute: bool = False,
    allow_live: bool = False,
    force: bool = False,
    order_type: str = "moc",
    limit_bps: float = 10.0,
    capital: float | None = None,
    leverage: float = 1.0,
    max_turnover: float = 1.25,
    cash_buffer: float = 0.01,
    min_trade_weight: float = 0.005,
) -> dict:
    today = now.astimezone(ET).date()
    state_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"date": str(today), "account": broker.account, "paper": broker.is_paper, "executed": False}

    if (state_dir / "HALT").exists():
        raise SafetyError(f"kill switch present: remove {state_dir / 'HALT'} to trade")
    if execute and not broker.is_paper and not allow_live:
        raise SafetyError("account is LIVE; refusing without --live")
    if leverage > 2.0 or leverage <= 0:
        raise SafetyError("leverage must be in (0, 2]")

    if not force:
        state = broker.session_state(today)
        if state != "last":
            report["skipped"] = "market closed today" if state == "closed" else "not the last session of the week"
            return report
    last_run = state_dir / "last_run.json"
    if execute and last_run.exists() and json.loads(last_run.read_text()).get("date") == str(today) and not force:
        report["skipped"] = "already executed today"
        return report
    if execute and order_type == "moc" and now.astimezone(ET).time() >= MOC_CUTOFF and not force:
        raise SafetyError("past the 15:50 ET MOC cutoff; use --order-type mkt or run earlier")

    prices = broker.daily_closes(list(cfg.symbols), cfg.min_history + 30)
    validate_prices(prices, today, cfg)
    weights = target_weights(prices, cfg)

    equity = broker.account_equity()
    if capital is not None:
        equity = min(equity, capital)
    held = {s: q for s, q in broker.positions().items() if s in cfg.symbols}
    last = broker.last_prices(list(cfg.symbols))
    orders = plan_orders(weights, last, held, equity, leverage=leverage, cash_buffer=cash_buffer,
                         min_trade_weight=min_trade_weight)

    turnover = sum(o.notional for o in orders) / equity
    report.update(
        equity=equity,
        weights={k: round(float(v), 4) for k, v in weights.items() if v > 0},
        positions=held,
        orders=[{**asdict(o), "side": o.side} for o in orders],
        turnover=round(turnover, 4),
        order_type=order_type,
    )
    (state_dir / f"decision-{today}.json").write_text(json.dumps(report, indent=2))
    if turnover > max_turnover:
        raise SafetyError(f"planned turnover {turnover:.2f}x equity exceeds --max-turnover {max_turnover}")

    if execute and orders:
        placed = broker.place(orders, order_type, tag=f"momrot-{today:%Y%m%d}", limit_bps=limit_bps)
        report["placed"] = [asdict(p) for p in placed]
        report["executed"] = True
        last_run.write_text(json.dumps({"date": str(today), "orders": len(placed)}))
    elif execute:
        report["executed"] = True
        last_run.write_text(json.dumps({"date": str(today), "orders": 0}))
    return report


def print_report(report: dict) -> None:
    if "skipped" in report:
        print(f"[{report['date']}] nothing to do: {report['skipped']}")
        return
    mode = "EXECUTED" if report["executed"] else "DRY RUN (no orders sent)"
    print(f"[{report['date']}] {mode}  account={report['account']} paper={report['paper']} equity=${report['equity']:,.0f}")
    print("target weights:", ", ".join(f"{k} {v:.1%}" for k, v in sorted(report["weights"].items(), key=lambda kv: -kv[1])))
    if not report["orders"]:
        print("no trades needed")
    for o in report["orders"]:
        print(f"  {o['side']:4s} {abs(o['quantity']):6d} {o['symbol']:5s} @ ~{o['ref_price']:.2f}  (${abs(o['quantity']) * o['ref_price']:,.0f})")
    print(f"turnover {report['turnover']:.1%} of equity, order type {report['order_type']}")
    for p in report.get("placed", []):
        print(f"  placed #{p['order_id']} {p['side']} {p['quantity']} {p['symbol']} -> {p['status']}")


def main(argv: list[str] | None = None, now: dt.datetime | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002, help="4002 Gateway paper, 7497 TWS paper, 4001/7496 live")
    ap.add_argument("--client-id", type=int, default=17)
    ap.add_argument("--account")
    ap.add_argument("--execute", action="store_true", help="transmit orders (default: dry run, read-only connection)")
    ap.add_argument("--live", action="store_true", help="allow trading a non-paper account")
    ap.add_argument("--force", action="store_true", help="ignore the last-session-of-week and already-ran guards")
    ap.add_argument("--order-type", choices=("moc", "mkt", "lmt"), default="moc")
    ap.add_argument("--limit-bps", type=float, default=10.0)
    ap.add_argument("--capital", type=float, help="cap the equity the strategy manages (e.g. for shared accounts)")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--max-turnover", type=float, default=1.25)
    ap.add_argument("--what-to-show", default="ADJUSTED_LAST", help="TRADES = split-adjusted only")
    ap.add_argument("--delayed-data", action="store_true", help="use delayed market data (no subscription)")
    ap.add_argument("--state-dir", default=str(Path(__file__).resolve().parents[1] / "state"))
    args = ap.parse_args(argv)

    broker = IBKRBroker(args.host, args.port, args.client_id, args.account, readonly=not args.execute,
                        market_data_type=3 if args.delayed_data else 1, what_to_show=args.what_to_show)
    try:
        report = run(
            broker, StrategyConfig(), now=now or dt.datetime.now(ET), state_dir=Path(args.state_dir), execute=args.execute,
            allow_live=args.live, force=args.force, order_type=args.order_type, limit_bps=args.limit_bps,
            capital=args.capital, leverage=args.leverage, max_turnover=args.max_turnover,
        )
    except SafetyError as e:
        print(f"REFUSING TO TRADE: {e}", file=sys.stderr)
        return 2
    finally:
        broker.close()
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
