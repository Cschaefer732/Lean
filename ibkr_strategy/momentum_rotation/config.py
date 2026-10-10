from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the multi-asset momentum rotation.

    Defaults are the plateau values found in research/ (not the in-sample optimum): top_k 4-6 and the blended
    1-3-6-12 month lookbacks were all within noise of each other.
    """

    risk_universe: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "TIP", "GLD", "DBC",
    )
    defensive: tuple[str, ...] = ("IEF", "TLT", "GLD")
    cash: str = "SHY"
    top_k: int = 5
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    vol_window: int = 63
    smoothing_weeks: int = 4

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.risk_universe, *self.defensive, self.cash)))

    @property
    def min_history(self) -> int:
        """Daily bars needed so the longest lookback is defined for the oldest smoothing date."""
        return max(max(self.lookbacks), self.vol_window) + 1 + 5 * self.smoothing_weeks
