"""Shared types and the Strategy/Context contract. Strategies only touch `Context`, so the identical object runs on the
replay simulator and on the live IBKR adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

BUY, SELL = 1, -1
TICK = 0.01


class Status(str, Enum):
    WORKING = "working"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass(slots=True)
class Quote:
    t: int          # ms since midnight (data timestamp)
    bid: float
    bsz: float
    ask: float
    asz: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(slots=True)
class Trade:
    t: int
    px: float
    sz: float
    side: int = 0   # +1 buyer initiated, -1 seller initiated (Lee-Ready against prevailing quote)


@dataclass(slots=True)
class Order:
    id: int
    side: int
    px: float
    qty: int
    post_only: bool = True
    tag: str = ""
    filled: int = 0
    status: Status = Status.WORKING
    t_created: int = 0

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    @property
    def active(self) -> bool:
        return self.status in (Status.WORKING, Status.CANCEL_PENDING)


@dataclass(slots=True)
class Fill:
    order_id: int
    t: int
    side: int
    px: float
    qty: int
    maker: bool
    fee: float = 0.0


class Context(Protocol):
    now: int
    quote: Quote | None
    position: int
    can_trade: bool

    def place(self, side: int, px: float, qty: int, post_only: bool = True, tag: str = "") -> Order | None: ...
    def cancel(self, order: Order) -> None: ...
    def cancel_all(self) -> None: ...
    def open_orders(self, side: int | None = None) -> list[Order]: ...


class Strategy:
    def on_quote(self, ctx: Context, q: Quote) -> None: ...
    def on_trade(self, ctx: Context, tr: Trade) -> None: ...
    def on_fill(self, ctx: Context, f: Fill) -> None: ...
    def on_end(self, ctx: Context) -> None: ...


def round_tick(px: float) -> float:
    return round(round(px / TICK) * TICK, 4)
