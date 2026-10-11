"""Market-making strategies. They only use `Context`, so they run unchanged in replay and live."""
from __future__ import annotations

import numpy as np

from .core import BUY, SELL, TICK, Context, Fill, Order, Quote, Status, Strategy, Trade, round_tick
from .signals import LinearAlpha, OnlineFeatures


class PassiveMM(Strategy):
    """Joins the touch on both sides with post-only limits, one order per side.

    `alpha` (optional) predicts the next-second mid move in ticks. It is used ONLY to avoid being picked off:
      bid is pulled when predicted move < -pull   (price likely to fall through our bid)
      ask is pulled when predicted move > +pull
    Inventory control: no new bid at +max_pos, no new ask at -max_pos. `pull=None` gives the naive always-quote baseline.
    """

    def __init__(self, qty=100, max_pos=300, alpha: LinearAlpha | None = None, pull: float | None = None,
                 min_requote_ms=0, shuffle_seed: int | None = None, name="pmm"):
        self.qty, self.max_pos, self.alpha, self.pull = qty, max_pos, alpha, pull
        self.min_requote_ms = min_requote_ms
        self.feat = OnlineFeatures()
        self.name = name
        self._last_action = {BUY: -10**12, SELL: -10**12}
        self._rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        self.n_pulls = 0

    def on_trade(self, ctx: Context, tr: Trade) -> None:
        self.feat.on_trade(tr)

    def on_fill(self, ctx: Context, f: Fill) -> None:
        pass

    def _score(self) -> float:
        if self.alpha is None:
            return 0.0
        s = self.alpha.score(self.feat.vector())
        if self._rng is not None:  # control: same score distribution, no information about the market
            s = s * (1 if self._rng.random() < 0.5 else -1)
        return s

    def on_quote(self, ctx: Context, q: Quote) -> None:
        self.feat.on_quote(q)
        if not ctx.can_trade:
            if ctx.open_orders():
                ctx.cancel_all()
            return
        score = self._score()
        pos = ctx.position
        want = {BUY: pos < self.max_pos, SELL: pos > -self.max_pos}
        if self.pull is not None:
            if score < -self.pull:
                want[BUY] = False
            if score > self.pull:
                want[SELL] = False
            if not (want[BUY] and want[SELL]):
                self.n_pulls += 1
        targets = self._targets(q)
        for side in (BUY, SELL):
            px = targets[side]
            if px is None:
                want[side] = False
            orders = ctx.open_orders(side)
            if any(o.status == Status.CANCEL_PENDING for o in orders):
                continue
            if orders:
                o = orders[0]
                if (not want[side] or o.px != px) and ctx.now - self._last_action[side] >= self.min_requote_ms:
                    ctx.cancel(o)
                    self._last_action[side] = ctx.now
            elif want[side] and ctx.now - self._last_action[side] >= self.min_requote_ms:
                if ctx.place(side, px, self.qty, post_only=True, tag=self.name):
                    self._last_action[side] = ctx.now

    def _targets(self, q: Quote) -> dict:
        return {BUY: q.bid, SELL: q.ask}

    def on_end(self, ctx: Context) -> None:
        ctx.cancel_all()


class InsideMM(PassiveMM):
    """Wide-spread variant: improve the touch by one tick on each side, but only when the spread is wide enough that
    both improved quotes still earn `min_edge_ticks` before fees (otherwise quote nothing)."""

    def __init__(self, min_spread_ticks: int = 3, **kw):
        super().__init__(**kw)
        self.min_spread_ticks = min_spread_ticks

    def _targets(self, q: Quote) -> dict:
        if round(q.spread / TICK) < self.min_spread_ticks:
            return {BUY: None, SELL: None}
        return {BUY: round_tick(q.bid + TICK), SELL: round_tick(q.ask - TICK)}
