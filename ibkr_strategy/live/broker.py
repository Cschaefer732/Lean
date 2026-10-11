"""Broker abstraction plus the Interactive Brokers (TWS / IB Gateway API via ib_async) implementation."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from momentum_rotation import Order

PREFERRED_EXCHANGES = ("ARCA", "NASDAQ", "NYSE", "BATS", "AMEX")
LIVE_PORTS = {7496, 4001}
PAPER_PORTS = {7497, 4002}


@dataclass(frozen=True)
class PlacedOrder:
    symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    order_id: int | None = None


class Broker(Protocol):
    account: str
    is_paper: bool

    def account_equity(self) -> float: ...
    def positions(self) -> dict[str, int]: ...
    def daily_closes(self, symbols: list[str], min_bars: int) -> pd.DataFrame:
        """Dividend/split-adjusted daily closes, last row = most recent (possibly in-progress) session."""
    def last_prices(self, symbols: list[str]) -> pd.Series:
        """Unadjusted tradable prices used for share sizing."""
    def session_state(self, today: dt.date) -> str:
        """'last' (final session of the week), 'open' (a session but not the last), or 'closed'."""
    def place(self, orders: list[Order], order_type: str, tag: str, limit_bps: float) -> list[PlacedOrder]: ...
    def close(self) -> None: ...


def parse_liquid_hours(liquid_hours: str) -> dict[dt.date, bool]:
    """Parse IB `liquidHours` ('20260928:0930-20260928:1600;20260929:CLOSED;...') into {date: is_open}."""
    sessions: dict[dt.date, bool] = {}
    for part in filter(None, liquid_hours.split(";")):
        m = re.match(r"(\d{8}):(\d{4}|CLOSED)", part)
        if not m:
            continue
        day = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        sessions[day] = sessions.get(day, False) or m.group(2) != "CLOSED"
    return sessions


def classify_session(sessions: dict[dt.date, bool], today: dt.date) -> str:
    """'last' if `today` is an open session and no later session exists in the same ISO week."""
    if not sessions.get(today, False):
        return "closed"
    week = today.isocalendar()[:2]
    later = [d for d, is_open in sessions.items() if is_open and d > today and d.isocalendar()[:2] == week]
    if later:
        return "open"
    covers_week_end = any(d.isocalendar()[:2] == week and d.weekday() == 4 for d in sessions)
    if not covers_week_end and today.weekday() < 4:
        return "open"  # calendar did not reach Friday, so we cannot prove this is the last session
    return "last"


class IBKRBroker:
    """Thin adapter over ib_async. `readonly=True` makes order placement impossible at the API level."""

    def __init__(self, host: str, port: int, client_id: int, account: str | None, readonly: bool,
                 market_data_type: int = 1, what_to_show: str = "ADJUSTED_LAST", timeout: float = 20.0):
        from ib_async import IB

        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id, readonly=readonly, timeout=timeout)
        accounts = self.ib.managedAccounts()
        if account is None:
            if len(accounts) != 1:
                raise RuntimeError(f"multiple accounts {accounts}; pass --account")
            account = accounts[0]
        if account not in accounts:
            raise RuntimeError(f"account {account} not in {accounts}")
        self.account = account
        self.is_paper = account.startswith("DU") or port in PAPER_PORTS
        self.what_to_show = what_to_show
        self.ib.reqMarketDataType(market_data_type)
        self._contracts: dict[str, object] = {}

    def _contract(self, symbol: str):
        if symbol in self._contracts:
            return self._contracts[symbol]
        from ib_async import Stock

        details = [d for d in self.ib.reqContractDetails(Stock(symbol, "SMART", "USD"))
                   if d.contract.secType == "STK" and d.contract.currency == "USD"]
        if not details:
            raise RuntimeError(f"IBKR has no US stock/ETF contract for {symbol}")
        if len(details) > 1:
            ranked = sorted(details, key=lambda d: PREFERRED_EXCHANGES.index(d.contract.primaryExchange)
                            if d.contract.primaryExchange in PREFERRED_EXCHANGES else 99)
            details = ranked[:1]
        contract = details[0].contract
        self._contracts[symbol] = contract
        return contract

    def account_equity(self) -> float:
        values = [v for v in self.ib.accountValues(self.account) if v.tag == "NetLiquidation"]
        for currency in ("USD", "BASE"):
            for v in values:
                if v.currency == currency:
                    return float(v.value)
        raise RuntimeError("NetLiquidation not available")

    def positions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.ib.positions(self.account):
            if p.contract.secType == "STK":
                out[p.contract.symbol] = out.get(p.contract.symbol, 0) + int(p.position)
        return out

    def _bars(self, symbol: str, duration: str, what: str) -> pd.Series:
        bars = self.ib.reqHistoricalData(
            self._contract(symbol), endDateTime="", durationStr=duration, barSizeSetting="1 day",
            whatToShow=what, useRTH=True, formatDate=1,
        )
        if not bars:
            raise RuntimeError(f"no {what} history for {symbol}")
        return pd.Series({pd.Timestamp(b.date): float(b.close) for b in bars}).sort_index()

    def daily_closes(self, symbols: list[str], min_bars: int) -> pd.DataFrame:
        years = max(2, -(-min_bars // 252) + 1)
        return pd.DataFrame({s: self._bars(s, f"{years} Y", self.what_to_show) for s in symbols}).sort_index()

    def last_prices(self, symbols: list[str]) -> pd.Series:
        return pd.Series({s: float(self._bars(s, "5 D", "TRADES").iloc[-1]) for s in symbols})

    def session_state(self, today: dt.date) -> str:
        details = self.ib.reqContractDetails(self._contract("SPY"))
        return classify_session(parse_liquid_hours(details[0].liquidHours), today)

    def place(self, orders: list[Order], order_type: str, tag: str, limit_bps: float) -> list[PlacedOrder]:
        from ib_async import LimitOrder, MarketOrder
        from ib_async import Order as IBOrder

        trades = []
        for o in orders:
            qty = abs(o.quantity)
            if order_type == "moc":
                ib_order = IBOrder(action=o.side, totalQuantity=qty, orderType="MOC", tif="DAY")
            elif order_type == "mkt":
                ib_order = MarketOrder(o.side, qty)
            elif order_type == "lmt":
                sign = 1 if o.side == "BUY" else -1
                ib_order = LimitOrder(o.side, qty, round(o.ref_price * (1 + sign * limit_bps / 1e4), 2))
            else:
                raise ValueError(order_type)
            ib_order.account = self.account
            ib_order.orderRef = tag
            trades.append((o, self.ib.placeOrder(self._contract(o.symbol), ib_order)))
        self.ib.sleep(2)
        return [PlacedOrder(o.symbol, o.side, abs(o.quantity), order_type, t.orderStatus.status or "Submitted",
                            t.order.orderId) for o, t in trades]

    def close(self) -> None:
        self.ib.disconnect()
