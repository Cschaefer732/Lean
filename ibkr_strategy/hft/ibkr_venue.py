"""Live adapter: IBKR (ib_async) tick-by-tick data + order entry behind the same `Context` the strategies see in replay.

What IBKR can and cannot do for HFT (details in docs/DESIGN.md): tick-by-tick data arrives over TCP through TWS/Gateway
(typically tens of ms), at most ~50 API messages/s, no post-only order type (emulated here with a local crossed-book
check, which can race), no queue-position information. This adapter is therefore suitable for the *slow* end of
microstructure strategies (decisions at O(100 ms - seconds)); the replay latency sweep quantifies what that costs.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from zoneinfo import ZoneInfo

from .core import BUY, SELL, Fill, Order, Quote, Status, Strategy, Trade, round_tick
from .data import CLOSE_MS, OPEN_MS
from .fees import IBKR_PRO_TIERED, FeeModel
from .latency import LatencyStats
from .risk import RiskGate, RiskLimits

log = logging.getLogger("hft.live")
ET = ZoneInfo("America/New_York")


def ms_since_midnight_et(ts: dt.datetime) -> int:
    ts = ts.astimezone(ET)
    return ((ts.hour * 60 + ts.minute) * 60 + ts.second) * 1000 + ts.microsecond // 1000


class LiveContext:
    def __init__(self, venue: "LiveVenue"):
        self._v = venue
        self.now = 0
        self.quote: Quote | None = None
        self.position = 0
        self.can_trade = False
        self._orders: dict[int, Order] = {}

    def place(self, side, px, qty, post_only=True, tag=""):
        return self._v.place(self, side, px, qty, post_only, tag)

    def cancel(self, order):
        self._v.cancel(self, order)

    def cancel_all(self):
        for o in list(self._orders.values()):
            if o.status == Status.WORKING:
                self._v.cancel(self, o)

    def open_orders(self, side=None):
        return [o for o in self._orders.values() if side is None or o.side == side]


class LiveVenue:
    def __init__(self, ib, contract, strategy: Strategy, *, dry_run: bool = True, limits: RiskLimits | None = None,
                 fees: FeeModel = IBKR_PRO_TIERED, recorder=None, stale_feed_s: float = 2.0, reconnect=None,
                 max_reconnects: int = 5, reconnect_every_s: float = 5.0, clock=time.time, trade_from_ms: int = OPEN_MS + 5 * 60_000, trade_until_ms: int = CLOSE_MS - 5 * 60_000):
        self.ib, self.contract, self.strategy, self.dry_run, self.fees = ib, contract, strategy, dry_run, fees
        self.risk = RiskGate(limits or RiskLimits(max_position=200, max_order_qty=100, max_loss=100.0))
        self.ctx = LiveContext(self)
        self.lat = LatencyStats()
        self.t_start, self.t_stop = trade_from_ms, trade_until_ms
        self.cash = 0.0
        self._ib_trades: dict[int, object] = {}
        self._next_id = 0
        self._q_idx = 0
        self._t_idx = 0
        self._quote_ticker = self._trade_ticker = None
        self.intents: list[tuple] = []
        self.recorder = recorder
        self.clock, self.stale_feed_s, self._reconnect = clock, stale_feed_s, reconnect
        self.max_reconnects, self.reconnect_every_s = max_reconnects, reconnect_every_s
        self.connected = True
        self.needs_reconcile = False
        self.feed_stale = False
        self.pos0 = 0                       # position held before this session; excluded from PnL
        self._last_tick_wall: float | None = None
        self._last_reconnect = -1e18
        self._reconnect_attempts = 0
        self._seen_exec: set = set()
        self._baseline_exec: set = set()
        self.recon = {"position_mismatch": 0, "orders_cancelled": 0, "reconnects": 0, "stale_halts": 0}

    # -- wiring ---------------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._baseline_exec = {self._exec_id(f) for f in self._broker_fills()}
        self._seen_exec |= self._baseline_exec
        self.reconcile(startup=True)
        self.pos0 = self.ctx.position
        ev = getattr(self.ib, "disconnectedEvent", None)
        if ev is not None:
            ev += self._on_disconnect
        self._subscribe()

    def _subscribe(self) -> None:
        self._q_idx = self._t_idx = 0
        self._quote_ticker = self.ib.reqTickByTickData(self.contract, "BidAsk", 0, False)
        self._trade_ticker = self.ib.reqTickByTickData(self.contract, "AllLast", 0, False)
        self._quote_ticker.updateEvent += self._on_quote_update
        self._trade_ticker.updateEvent += self._on_trade_update

    # -- connection health, staleness, reconciliation -----------------------------------------------------------------
    def _on_disconnect(self) -> None:
        self.connected = False
        self.ctx.can_trade = False
        self.needs_reconcile = True
        log.error("IB disconnected: quoting stopped. Orders already at the broker may still be live until reconciled.")

    def _broker_fills(self) -> list:
        try:
            return [f for f in self.ib.fills() if f.contract.symbol == self.contract.symbol]
        except AttributeError:
            return []

    @staticmethod
    def _exec_id(fill):
        return getattr(fill.execution, "execId", None) or id(fill)

    def reconcile(self, startup: bool = False) -> None:
        """Make local state match the broker: position, cash (from executions), and no stray resting orders.

        Everything resting at the broker for this symbol is cancelled rather than adopted: a market maker re-quotes from
        scratch, and an order whose state we can no longer vouch for is a liability."""
        sym = self.contract.symbol
        broker_pos = int(sum(p.position for p in self.ib.positions() if p.contract.symbol == sym))
        if not startup and broker_pos != self.ctx.position:
            self.recon["position_mismatch"] += 1
            log.warning("position mismatch after reconcile: local %d, broker %d (broker wins)", self.ctx.position, broker_pos)
        self.ctx.position = broker_pos
        if not startup:
            self.cash = 0.0
            for f in self._broker_fills():
                eid = self._exec_id(f)
                if eid in self._baseline_exec:
                    continue
                self._seen_exec.add(eid)
                side = BUY if f.execution.side.upper().startswith("B") else SELL
                self.cash -= side * float(f.execution.price) * int(f.execution.shares) + float(getattr(f.commissionReport, "commission", 0.0) or 0.0)
        if not self.dry_run:
            for tr in self.ib.openTrades():
                if tr.contract.symbol == sym:
                    self.ib.cancelOrder(tr.order)
                    self.recon["orders_cancelled"] += 1
        for o in self.ctx._orders.values():
            o.status = Status.CANCELLED
        self.ctx._orders.clear()
        self._ib_trades.clear()
        self.needs_reconcile = False

    def health_check(self) -> None:
        """Called from the run loop: reconnect with a bounded retry budget, reconcile, and halt quoting on a dead feed."""
        now = self.clock()
        is_up = getattr(self.ib, "isConnected", lambda: True)()
        if not is_up or not self.connected:
            self.connected = False
            self.ctx.can_trade = False
            if now - self._last_reconnect < self.reconnect_every_s:
                return
            self._last_reconnect = now
            if self._reconnect is None or self._reconnect_attempts >= self.max_reconnects:
                self.risk.halt("disconnected")
                return
            self._reconnect_attempts += 1
            try:
                self._reconnect()
                self._subscribe()
                self.connected = True
                self._reconnect_attempts = 0
                self.recon["reconnects"] += 1
                self.reconcile()
            except Exception as exc:  # noqa: BLE001 - any failure leaves us not-trading and retrying
                log.error("reconnect attempt %d failed: %s", self._reconnect_attempts, exc)
            return
        if self.needs_reconcile:
            self.reconcile()
        t_ms = ms_since_midnight_et(dt.datetime.fromtimestamp(now, dt.timezone.utc))
        if self._last_tick_wall is not None and self.t_start <= t_ms < self.t_stop and now - self._last_tick_wall > self.stale_feed_s:
            if not self.feed_stale:
                self.feed_stale = True
                self.recon["stale_halts"] += 1
                log.error("no quote for %.1fs: pulling all orders", now - self._last_tick_wall)
            self.ctx.can_trade = False
            if self.ctx.open_orders():
                self.ctx.cancel_all()

    def stop(self) -> None:
        self.ctx.cancel_all()
        self.strategy.on_end(self.ctx)
        for t in (self._quote_ticker, self._trade_ticker):
            if t is not None:
                self.ib.cancelTickByTickData(self.contract, "BidAsk" if t is self._quote_ticker else "AllLast")

    def run(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and not self.risk.state.halted:
            self.ib.sleep(0.05)
            self.health_check()
        self.stop()

    # -- market data ----------------------------------------------------------------------------------------------
    def _on_quote_update(self, ticker) -> None:
        ticks = ticker.tickByTicks
        new, self._q_idx = ticks[self._q_idx:], len(ticks)
        for tk in new:
            if tk.bidPrice <= 0 or tk.askPrice <= 0 or tk.bidPrice >= tk.askPrice:
                continue
            q = Quote(ms_since_midnight_et(tk.time), tk.bidPrice, float(tk.bidSize), tk.askPrice, float(tk.askSize))
            if self.recorder is not None:
                self.recorder.on_quote(q)
            t0 = time.perf_counter_ns()
            self.lat.record("feed_age_us", max(0.0, (dt.datetime.now(dt.timezone.utc) - tk.time).total_seconds() * 1e6))
            self.ctx.quote, self.ctx.now = q, q.t
            self._last_tick_wall = self.clock()
            self.feed_stale = False
            self.ctx.can_trade = self.t_start <= q.t < self.t_stop and not self.risk.state.halted and self.connected
            if self.risk.state.halted and self.ctx.open_orders():
                self.ctx.cancel_all()
            self.strategy.on_quote(self.ctx, q)
            self._check_pnl(q)
            self.lat.record("tick_to_decision_us", (time.perf_counter_ns() - t0) / 1e3)

    def _on_trade_update(self, ticker) -> None:
        ticks = ticker.tickByTicks
        new, self._t_idx = ticks[self._t_idx:], len(ticks)
        q = self.ctx.quote
        for tk in new:
            if q is None:
                continue
            mid = q.mid
            side = 1 if tk.price > mid else (-1 if tk.price < mid else 0)
            tr = Trade(ms_since_midnight_et(tk.time), tk.price, float(tk.size), side)
            if self.recorder is not None:
                self.recorder.on_trade(tr)
            self.strategy.on_trade(self.ctx, tr)

    def _check_pnl(self, q: Quote) -> None:
        self.risk.check_pnl(self.cash + (self.ctx.position - self.pos0) * q.mid)

    # -- orders ---------------------------------------------------------------------------------------------------
    def place(self, ctx: LiveContext, side, px, qty, post_only, tag):
        q = ctx.quote
        if not ctx.can_trade or q is None:
            return None
        px = round_tick(px)
        if post_only and ((side == BUY and px >= q.ask) or (side == SELL and px <= q.bid)):
            self.risk._reject("would_cross")
            return None
        wb = sum(o.remaining for o in ctx._orders.values() if o.side == BUY)
        ws = sum(o.remaining for o in ctx._orders.values() if o.side == SELL)
        if self.risk.check_order(time.time() * 1000, side, px, qty, q.mid, ctx.position, wb, ws):
            return None
        self._next_id += 1
        o = Order(self._next_id, side, px, qty, post_only, tag, t_created=ctx.now)
        ctx._orders[o.id] = o
        if self.dry_run:
            self.intents.append(("place", o.id, side, px, qty))
            log.info("DRY place %s %d@%.2f", "BUY" if side > 0 else "SELL", qty, px)
            return o
        from ib_async import LimitOrder

        t0 = time.perf_counter_ns()
        trade = self.ib.placeOrder(self.contract, LimitOrder("BUY" if side > 0 else "SELL", qty, px, tif="DAY", outsideRth=False))
        self.lat.record("place_call_us", (time.perf_counter_ns() - t0) / 1e3)
        self._ib_trades[o.id] = trade
        trade.fillEvent += lambda tr, fill, oid=o.id: self._on_fill(oid, fill)
        trade.statusEvent += lambda tr, oid=o.id: self._on_status(oid, tr)
        return o

    def cancel(self, ctx: LiveContext, order: Order) -> None:
        if order.status != Status.WORKING or not self.risk.check_cancel(time.time() * 1000):
            return
        order.status = Status.CANCEL_PENDING
        if self.dry_run:
            self.intents.append(("cancel", order.id))
            order.status = Status.CANCELLED
            ctx._orders.pop(order.id, None)
            return
        self.ib.cancelOrder(self._ib_trades[order.id].order)

    def _on_fill(self, oid: int, fill) -> None:
        eid = self._exec_id(fill)
        if eid in self._seen_exec:
            return
        self._seen_exec.add(eid)
        side = BUY if fill.execution.side.upper().startswith("B") else SELL
        qty, px = int(fill.execution.shares), float(fill.execution.price)
        fee = float(getattr(fill.commissionReport, "commission", 0.0) or 0.0)
        # every execution moves position and cash, even for an order we already forgot (e.g. cancelled during a reconcile)
        self.cash -= side * px * qty + fee
        self.ctx.position += side * qty
        o = self.ctx._orders.get(oid)
        if o is None:
            return
        o.filled += qty
        if o.remaining <= 0:
            o.status = Status.FILLED
            self.ctx._orders.pop(oid, None)
        self.strategy.on_fill(self.ctx, Fill(oid, self.ctx.now, side, px, qty, maker=True, fee=fee))

    def _on_status(self, oid: int, trade) -> None:
        st = trade.orderStatus.status
        o = self.ctx._orders.get(oid)
        if o is None:
            return
        if st in ("Cancelled", "ApiCancelled", "Inactive"):
            o.status = Status.REJECTED if st == "Inactive" else Status.CANCELLED
            self.ctx._orders.pop(oid, None)
