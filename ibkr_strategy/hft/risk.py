"""Pre-trade risk gate, message-rate limiter and kill switch. Used identically in replay and live."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    max_order_qty: int = 500
    max_position: int = 500             # |position + working buys| and |position - working sells| must stay under this
    max_notional: float = 250_000.0     # per order
    price_band_ticks: float = 20.0      # fat-finger: reject limit prices this far from the reference mid
    max_msgs_per_sec: float = 50.0      # IBKR hard limit on API messages/s; strategies must live inside it
    max_loss: float = 500.0             # session loss (mark-to-mid, after fees) that trips the kill switch


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: float | None = None):
        self.rate = rate_per_sec
        self.cap = burst if burst is not None else rate_per_sec
        self.tokens = self.cap
        self.t = None

    def allow(self, t_ms: float, cost: float = 1.0) -> bool:
        if self.t is not None:
            self.tokens = min(self.cap, self.tokens + (t_ms - self.t) / 1000.0 * self.rate)
        self.t = t_ms
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


@dataclass
class RiskState:
    rejected: dict = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""


class RiskGate:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        # the IBKR budget is split so a burst of new orders can never starve cancels (a dropped cancel is a risk event)
        self.bucket = TokenBucket(self.limits.max_msgs_per_sec / 2)
        self.cancel_bucket = TokenBucket(self.limits.max_msgs_per_sec / 2)
        self.state = RiskState()

    def _reject(self, why: str) -> str:
        self.state.rejected[why] = self.state.rejected.get(why, 0) + 1
        return why

    def halt(self, reason: str) -> None:
        if not self.state.halted:
            self.state.halted, self.state.halt_reason = True, reason

    def check_order(self, t_ms, side, px, qty, ref_mid, position, working_buy, working_sell) -> str | None:
        """Returns a rejection reason or None. Consumes one message token only when the order passes."""
        L = self.limits
        if self.state.halted:
            return self._reject("halted")
        if qty <= 0 or qty > L.max_order_qty:
            return self._reject("order_qty")
        if px * qty > L.max_notional:
            return self._reject("notional")
        if ref_mid is not None and abs(px - ref_mid) > L.price_band_ticks * 0.01:
            return self._reject("price_band")
        if side > 0 and position + working_buy + qty > L.max_position:
            return self._reject("position_limit")
        if side < 0 and -(position - working_sell - qty) > L.max_position:
            return self._reject("position_limit")
        if not self.bucket.allow(t_ms):
            return self._reject("rate_limit")
        return None

    def check_cancel(self, t_ms) -> bool:
        """Cancels draw on their own reserved budget; when halted they always pass (flattening risk is worse than a breach)."""
        ok = self.cancel_bucket.allow(t_ms)
        if not ok:
            self._reject("rate_limit_cancel")
        return ok or self.state.halted

    def check_pnl(self, pnl: float) -> None:
        if pnl <= -self.limits.max_loss:
            self.halt(f"max_loss {pnl:.2f}")
