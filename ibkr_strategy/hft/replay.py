"""Event-driven replay venue: real NBBO + trade prints in, simulated fills for our orders out.

Timeline semantics (all times in data milliseconds):
  market event at t  -> venue state/fills updated at t -> strategy sees it at t + md_latency
  strategy order     -> reaches the venue at t_decision + order_latency (so cancels and fills race)
  venue fill at t    -> strategy learns of it at t + order_latency

Fill model for a passive order (the standard conservative L1 queue model):
  * joins the BACK of the visible aggregate queue at its price (queue_ahead = size at touch * `queue_pessimism`)
  * trades at our price consume queue_ahead first; the remainder fills us
  * trades strictly through our price fill us fully
  * size drops at our level not explained by trades are cancellations and shrink queue_ahead proportionally
    (`queue_model='prob'`) or not at all (`'none'`, the most pessimistic)
  * orders resting behind the touch (market moved away) cannot fill until the market trades through them
Known approximations (documented in docs/DESIGN.md): consolidated NBBO treated as one venue, orders do not affect the
market, no hidden liquidity, marketable orders fill fully at the touch.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from .core import BUY, SELL, TICK, Context, Fill, Order, Quote, Status, Strategy, Trade, round_tick
from .data import CLOSE_MS, OPEN_MS, Day
from .fees import IBKR_PRO_TIERED, FeeModel
from .risk import RiskGate, RiskLimits

INF = 1e18
Q_DELIVER, T_DELIVER, ORDER_ARRIVE, CANCEL_ARRIVE, FILL_DELIVER, NOTE_DELIVER = range(6)


@dataclass
class ReplayConfig:
    md_latency_ms: float = 0.0       # market data: venue -> strategy
    order_latency_ms: float = 0.0    # one way: strategy -> venue and venue -> strategy
    fees: FeeModel = IBKR_PRO_TIERED
    queue_model: str = "prob"        # 'prob' or 'none'
    queue_pessimism: float = 1.0     # multiplies the queue ahead when joining a level
    start_min: float = 5.0           # minutes after the open before we may trade
    end_min: float = 5.0             # minutes before the close when we stop and cancel
    limits: RiskLimits = field(default_factory=RiskLimits)


@dataclass
class _VOrder:
    o: Order
    px: float
    side: int
    remaining: int
    ahead: float = 0.0
    rel: str = "at"          # 'at' | 'inside' | 'behind'
    exp_size: float = 0.0


@dataclass
class Result:
    symbol: str
    date: str
    fills: list
    pnl_mid: float
    pnl_liq: float
    fees: float
    position_end: int
    pos_abs_mean: float
    pos_abs_max: int
    pnl_path: np.ndarray
    pnl_t: np.ndarray
    counts: dict
    wall_s: float
    decision_us: np.ndarray
    halted: str
    markout: dict

    def summary(self) -> dict:
        shares = sum(f.qty for f in self.fills)
        maker = sum(f.qty for f in self.fills if f.maker)
        return {
            "date": self.date, "pnl_mid": round(self.pnl_mid, 2), "pnl_liq": round(self.pnl_liq, 2), "fees": round(self.fees, 2),
            "fills": len(self.fills), "shares": shares, "maker_share": round(maker / shares, 3) if shares else 0.0,
            "pnl_per_share_c": round(100 * self.pnl_liq / shares, 3) if shares else 0.0,
            "pos_abs_mean": round(self.pos_abs_mean, 1), "pos_abs_max": self.pos_abs_max, **{f"mo_{k}": round(v, 3) for k, v in self.markout.items()},
        }


class ReplayContext:
    def __init__(self, engine: "Replay"):
        self._e = engine
        self.now = 0
        self.quote: Quote | None = None
        self.position = 0
        self.can_trade = False
        self._orders: dict[int, Order] = {}   # active orders only (terminal ones are dropped)

    def place(self, side, px, qty, post_only=True, tag=""):
        return self._e._strategy_place(self, side, px, qty, post_only, tag)

    def cancel(self, order):
        self._e._strategy_cancel(self, order)

    def cancel_all(self):
        for o in list(self._orders.values()):
            if o.status == Status.WORKING:
                self._e._strategy_cancel(self, o)

    def open_orders(self, side=None):
        return [o for o in self._orders.values() if side is None or o.side == side]


class Replay:
    def __init__(self, day: Day, strategy: Strategy, cfg: ReplayConfig | None = None, measure_decisions: bool = True):
        self.day, self.strategy, self.cfg = day, strategy, cfg or ReplayConfig()
        self.risk = RiskGate(self.cfg.limits)
        self.ctx = ReplayContext(self)
        self.measure = measure_decisions
        self._heap: list = []
        self._seq = 0
        self._oid = 0
        # venue truth
        self.bid = self.ask = None
        self.bsz = self.asz = 0.0
        self.working: dict[int, _VOrder] = {}
        self.cash = 0.0
        self.pos = 0
        self.fees = 0.0
        self.fills: list[Fill] = []
        self.counts = {"sent": 0, "cancels": 0, "venue_rejects": 0, "risk_rejects": 0, "fills_maker": 0, "fills_taker": 0, "late_cancel": 0}
        self._dec_ns: list[int] = []
        self._pnl_path: list[float] = []
        self._pnl_t: list[int] = []
        self._pos_samples: list[int] = []
        self._next_sample = 0
        self.t_stop = CLOSE_MS - int(self.cfg.end_min * 60_000)
        self.t_start = OPEN_MS + int(self.cfg.start_min * 60_000)
        self._last_trade_sign = 0

    # ---- scheduling -------------------------------------------------------------------------------------------
    def _push(self, t, kind, payload):
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, kind, payload))

    # ---- strategy-side API ------------------------------------------------------------------------------------
    def _mtm(self) -> float:
        if self.bid is None:
            return self.cash
        return self.cash + self.pos * 0.5 * (self.bid + self.ask)

    def _strategy_place(self, ctx, side, px, qty, post_only, tag):
        t = ctx.now
        if not ctx.can_trade:
            self.counts["risk_rejects"] += 1
            return None
        px = round_tick(px)
        wb = sum(o.remaining for o in ctx._orders.values() if o.side == BUY)
        ws = sum(o.remaining for o in ctx._orders.values() if o.side == SELL)
        ref = ctx.quote.mid if ctx.quote else None
        why = self.risk.check_order(t, side, px, qty, ref, ctx.position, wb, ws)
        if why:
            self.counts["risk_rejects"] += 1
            return None
        self._oid += 1
        o = Order(self._oid, side, px, qty, post_only, tag, t_created=t)
        ctx._orders[o.id] = o
        self.counts["sent"] += 1
        self._push(t + self.cfg.order_latency_ms, ORDER_ARRIVE, o)
        return o

    def _strategy_cancel(self, ctx, order):
        if order.status != Status.WORKING:
            return
        if not self.risk.check_cancel(ctx.now):
            return
        order.status = Status.CANCEL_PENDING
        self.counts["cancels"] += 1
        self._push(ctx.now + self.cfg.order_latency_ms, CANCEL_ARRIVE, order)

    # ---- venue side -------------------------------------------------------------------------------------------
    def _fill(self, vo: _VOrder, qty: int, px: float, t: int, maker: bool):
        qty = min(qty, vo.remaining)
        if qty <= 0:
            return
        vo.remaining -= qty
        fee = self.cfg.fees.fee(vo.side, px, qty, maker)
        self.cash -= vo.side * px * qty + fee
        self.pos += vo.side * qty
        self.fees += fee
        f = Fill(vo.o.id, t, vo.side, px, qty, maker, fee)
        self.fills.append(f)
        self.counts["fills_maker" if maker else "fills_taker"] += 1
        if vo.remaining == 0:
            self.working.pop(vo.o.id, None)
        self._push(t + self.cfg.order_latency_ms, FILL_DELIVER, (f, vo.remaining == 0))

    def _on_order_arrive(self, o: Order, t: int):
        b, a = self.bid, self.ask
        vo = _VOrder(o, o.px, o.side, o.qty)
        if (o.side == BUY and o.px >= a) or (o.side == SELL and o.px <= b):
            if o.post_only:
                self.counts["venue_rejects"] += 1
                self._push(t + self.cfg.order_latency_ms, NOTE_DELIVER, (o, Status.REJECTED))
                return
            px = a if o.side == BUY else b
            self.working[o.id] = vo
            self._fill(vo, o.qty, px, t, maker=False)
            return
        pess = self.cfg.queue_pessimism
        if o.side == BUY:
            vo.rel, vo.ahead, vo.exp_size = ("inside", 0.0, 0.0) if o.px > b else (("at", self.bsz * pess, self.bsz) if o.px == b else ("behind", INF, 0.0))
        else:
            vo.rel, vo.ahead, vo.exp_size = ("inside", 0.0, 0.0) if o.px < a else (("at", self.asz * pess, self.asz) if o.px == a else ("behind", INF, 0.0))
        self.working[o.id] = vo

    def _on_cancel_arrive(self, o: Order, t: int):
        vo = self.working.pop(o.id, None)
        if vo is None:
            self.counts["late_cancel"] += 1
            self._push(t + self.cfg.order_latency_ms, NOTE_DELIVER, (o, None))
            return
        self._push(t + self.cfg.order_latency_ms, NOTE_DELIVER, (o, Status.CANCELLED))

    def _cancel_all_venue(self, t):
        for oid, vo in list(self.working.items()):
            self.working.pop(oid)
            self._push(t + self.cfg.order_latency_ms, NOTE_DELIVER, (vo.o, Status.CANCELLED))

    def _on_market_quote(self, q: Quote):
        self.bid, self.bsz, self.ask, self.asz = q.bid, q.bsz, q.ask, q.asz
        pess = self.cfg.queue_pessimism
        for vo in list(self.working.values()):
            if vo.side == BUY:
                if q.ask < vo.px:
                    self._fill(vo, vo.remaining, vo.px, q.t, maker=True)
                    continue
                lvl, size = q.bid, q.bsz
                rel = "inside" if vo.px > lvl else ("at" if vo.px == lvl else "behind")
            else:
                if q.bid > vo.px:
                    self._fill(vo, vo.remaining, vo.px, q.t, maker=True)
                    continue
                lvl, size = q.ask, q.asz
                rel = "inside" if vo.px < lvl else ("at" if vo.px == lvl else "behind")
            if rel == "inside":
                vo.ahead, vo.exp_size = 0.0, 0.0
            elif rel == "behind":
                vo.ahead = INF
            else:
                if vo.rel != "at":
                    vo.ahead, vo.exp_size = size * pess, size
                else:
                    if size < vo.exp_size and self.cfg.queue_model == "prob" and vo.exp_size > 0:
                        vo.ahead *= size / vo.exp_size
                    vo.ahead = min(vo.ahead, size * max(pess, 1.0))
                    vo.exp_size = size
            vo.rel = rel

    def _on_market_trade(self, tr: Trade):
        for vo in list(self.working.values()):
            if vo.side == BUY:
                through, touch = tr.px < vo.px, tr.px == vo.px
            else:
                through, touch = tr.px > vo.px, tr.px == vo.px
            if through:
                self._fill(vo, vo.remaining, vo.px, tr.t, maker=True)
            elif touch and vo.rel in ("at", "inside"):
                vo.ahead -= tr.sz
                vo.exp_size = max(0.0, vo.exp_size - tr.sz)
                if vo.ahead < 0:
                    avail = int(-vo.ahead)
                    vo.ahead = 0.0
                    self._fill(vo, avail, vo.px, tr.t, maker=True)

    # ---- strategy delivery ------------------------------------------------------------------------------------
    def _call(self, fn, *args):
        if self.measure:
            t0 = time.perf_counter_ns()
            fn(self.ctx, *args)
            self._dec_ns.append(time.perf_counter_ns() - t0)
        else:
            fn(self.ctx, *args)

    def _drain(self, until: float):
        h = self._heap
        ctx = self.ctx
        while h and h[0][0] <= until:
            t, _, kind, p = heapq.heappop(h)
            ctx.now = int(t)
            if kind == Q_DELIVER:
                ctx.quote = p
                ctx.can_trade = (self.t_start <= p.t < self.t_stop) and not self.risk.state.halted
                if self.risk.state.halted and ctx.open_orders():
                    ctx.cancel_all()
                self._call(self.strategy.on_quote, p)
            elif kind == T_DELIVER:
                self._call(self.strategy.on_trade, p)
            elif kind == ORDER_ARRIVE:
                self._on_order_arrive(p, int(t))
            elif kind == CANCEL_ARRIVE:
                self._on_cancel_arrive(p, int(t))
            elif kind == FILL_DELIVER:
                f, done = p
                o = ctx._orders.get(f.order_id)
                if o is not None:
                    o.filled += f.qty
                    if done:
                        o.status = Status.FILLED
                        del ctx._orders[f.order_id]
                ctx.position += f.side * f.qty
                self._call(self.strategy.on_fill, f)
            elif kind == NOTE_DELIVER:
                o, status = p
                if status is None:
                    pass  # cancel arrived after the order was already done; fill notification tells the story
                elif o.status != Status.FILLED:
                    o.status = status
                    ctx._orders.pop(o.id, None)

    # ---- main loop --------------------------------------------------------------------------------------------
    def run(self) -> Result:
        d = self.day
        w0 = time.time()
        qv = d.q_bid < d.q_ask
        qi = np.flatnonzero(qv & (d.q_t <= self.t_stop))
        ti = np.flatnonzero(d.t_t <= self.t_stop)
        # same-millisecond ordering: trades first, they cause the quote updates stamped in the same ms
        t_all = np.concatenate([d.t_t[ti], d.q_t[qi]])
        kind = np.concatenate([np.zeros(len(ti), np.int8), np.ones(len(qi), np.int8)])
        src = np.concatenate([ti, qi])
        order = np.lexsort((kind, t_all))
        md = self.cfg.md_latency_ms
        QT, QB, QBS, QA, QAS = d.q_t, d.q_bid, d.q_bsz, d.q_ask, d.q_asz
        TT, TP, TS = d.t_t, d.t_px, d.t_sz
        have_quote = False
        for k in order:
            t = int(t_all[k])
            i = int(src[k])
            self._drain(t)
            if t >= self._next_sample and have_quote:
                self._pnl_path.append(self._mtm())
                self._pnl_t.append(t)
                self._pos_samples.append(abs(self.pos))
                self._next_sample = t + 1000
            if t >= self.t_stop and self.working:
                self._cancel_all_venue(t)
            if kind[k] == 1:
                q = Quote(t, QB[i], QBS[i], QA[i], QAS[i])
                self._on_market_quote(q)
                have_quote = True
                self._push(t + md, Q_DELIVER, q)
            else:
                if not have_quote:
                    continue
                px = TP[i]
                mid = 0.5 * (self.bid + self.ask)
                side = 1 if px > mid else (-1 if px < mid else self._last_trade_sign)
                self._last_trade_sign = side or self._last_trade_sign
                tr = Trade(t, px, TS[i], side)
                self._on_market_trade(tr)
                self._push(t + md, T_DELIVER, tr)
            if not self.risk.state.halted:
                self.risk.check_pnl(self._mtm())
        self._drain(self.t_stop + 10 * (md + self.cfg.order_latency_ms) + 1)
        self._cancel_all_venue(self.t_stop)
        self._drain(float("inf"))
        self.strategy.on_end(self.ctx)
        return self._result(time.time() - w0)

    # ---- reporting --------------------------------------------------------------------------------------------
    def _markouts(self) -> dict:
        d = self.day
        if not self.fills:
            return {}
        valid = d.q_bid < d.q_ask
        qt, mid = d.q_t[valid], 0.5 * (d.q_bid + d.q_ask)[valid]

        def mid_at(ts):
            return mid[np.clip(np.searchsorted(qt, ts, side="right") - 1, 0, len(mid) - 1)]
        ft = np.array([f.t for f in self.fills]); fpx = np.array([f.px for f in self.fills]); fs = np.array([f.side for f in self.fills])
        fq = np.array([f.qty for f in self.fills], float); mk = np.array([f.maker for f in self.fills])
        w = fq * mk
        if w.sum() == 0:
            return {}
        out = {"spread_capture": float(np.sum(w * fs * (mid_at(ft) - fpx)) / w.sum() / TICK)}
        for name, ms in (("1s", 1000), ("5s", 5000)):
            out[f"markout_{name}"] = float(np.sum(w * fs * (mid_at(ft + ms) - mid_at(ft))) / w.sum() / TICK)
        return out

    def _result(self, wall: float) -> Result:
        mid = 0.5 * (self.bid + self.ask) if self.bid is not None else 0.0
        liq = 0.0
        if self.pos and self.bid is not None:
            px = self.bid if self.pos > 0 else self.ask
            liq = self.cfg.fees.fee(-1 if self.pos > 0 else 1, px, abs(self.pos), maker=False)
            pnl_liq = self.cash + self.pos * px - liq
        else:
            pnl_liq = self.cash
        pos_abs = np.array(self._pos_samples) if self._pos_samples else np.zeros(1)
        return Result(
            self.day.symbol, self.day.date, self.fills, self.cash + self.pos * mid, pnl_liq, self.fees + liq, self.pos,
            float(pos_abs.mean()), int(pos_abs.max()), np.array(self._pnl_path), np.array(self._pnl_t), dict(self.counts) | {"risk": dict(self.risk.state.rejected)},
            wall, np.array(self._dec_ns) / 1e3, self.risk.state.halt_reason, self._markouts(),
        )
