from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: int  # signed: >0 buy, <0 sell
    ref_price: float

    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.ref_price


def plan_orders(
    weights: pd.Series,
    prices: pd.Series,
    positions: dict[str, int],
    equity: float,
    *,
    leverage: float = 1.0,
    cash_buffer: float = 0.01,
    min_trade_weight: float = 0.005,
) -> list[Order]:
    """Whole-share orders that move the managed symbols to `weights * equity * leverage`.

    Only symbols in `weights` are ever touched, so unrelated holdings in the account are left alone. Trades smaller
    than `min_trade_weight` of equity are skipped (except full exits) to avoid paying commissions on drift.
    Sells are returned before buys so proceeds fund purchases.
    """
    if equity <= 0:
        raise ValueError("equity must be positive")
    budget = equity * leverage * (1.0 - cash_buffer)
    orders: list[Order] = []
    for sym, w in weights.items():
        px = float(prices[sym])
        if not math.isfinite(px) or px <= 0:
            raise ValueError(f"bad price for {sym}: {px}")
        target = math.floor(w * budget / px) if w > 0 else 0
        current = int(positions.get(sym, 0))
        delta = target - current
        if delta == 0:
            continue
        exits_position = target == 0 and current > 0
        if not exits_position and abs(delta) * px < min_trade_weight * equity:
            continue
        orders.append(Order(sym, delta, px))
    return sorted(orders, key=lambda o: (o.quantity > 0, o.symbol))
