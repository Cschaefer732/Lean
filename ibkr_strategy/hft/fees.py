"""Per-share fee schedules. Numbers are approximations of published schedules and must be re-checked against the
broker/venue schedule in force; the research question is how fast the edge dies as fees move, so they are parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    name: str
    commission: float = 0.0          # $/share
    min_commission: float = 0.0      # $/fill (IBKR applies it per order; per-fill is a slightly conservative proxy)
    clearing: float = 0.0            # $/share
    exchange_remove: float = 0.0     # $/share paid when taking liquidity
    exchange_add: float = 0.0        # $/share paid when adding liquidity (negative = rebate)
    sec_rate: float = 0.0            # fraction of sale notional (sells only)
    taf: float = 0.0                 # $/share on sells (FINRA trading activity fee)

    def fee(self, side: int, px: float, qty: int, maker: bool) -> float:
        per_share = self.commission + self.clearing + (self.exchange_add if maker else self.exchange_remove)
        fee = max(self.min_commission, self.commission * qty) + (per_share - self.commission) * qty
        if side < 0:
            fee += self.sec_rate * px * qty + self.taf * qty
        return fee


FREE = FeeModel("free")
IBKR_PRO_TIERED = FeeModel("ibkr_pro_tiered", commission=0.0035, min_commission=0.35, clearing=0.0002,
                           exchange_remove=0.0030, exchange_add=-0.0020, sec_rate=0.0000278, taf=0.000119)
PROP_REBATE = FeeModel("prop_rebate_tier", commission=0.0002, clearing=0.0001, exchange_remove=0.0030,
                       exchange_add=-0.0030, sec_rate=0.0000278, taf=0.000119)
PRESETS = {m.name: m for m in (FREE, IBKR_PRO_TIERED, PROP_REBATE)}
