import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AlgorithmImports import *  # noqa: E402,F401,F403

from momentum_rotation import StrategyConfig, target_weights  # noqa: E402


class IBKRMomentumRotation(QCAlgorithm):
    """Multi-asset momentum rotation on liquid ETFs, executed through the Interactive Brokers brokerage model.

    Signal logic lives in `momentum_rotation` and is shared verbatim with the direct IBKR API runner
    (live/ibkr_trader.py), so backtest, LEAN live-trading and API trading all trade the same book.
    Parameters (all optional, `--parameters key=value`): start_date, end_date (YYYY-MM-DD), cash, top_k,
    smoothing_weeks, leverage.
    """

    def initialize(self):
        start = self.get_parameter("start_date", "2007-02-01")
        end = self.get_parameter("end_date", "")
        y, m, d = (int(x) for x in start.split("-"))
        self.set_start_date(y, m, d)
        if end:
            y, m, d = (int(x) for x in end.split("-"))
            self.set_end_date(y, m, d)
        self.set_cash(self.get_parameter("cash", 100000))

        self.cfg = StrategyConfig(
            top_k=self.get_parameter("top_k", 5),
            smoothing_weeks=self.get_parameter("smoothing_weeks", 4),
        )
        self.leverage = self.get_parameter("leverage", 1.0)
        self.cash_buffer = 0.01

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0.005
        self.settings.free_portfolio_value_percentage = 0.0

        self.symbols = {t: self.add_equity(t, Resolution.DAILY).symbol for t in self.cfg.symbols}
        self.set_benchmark("SPY")
        self.set_warm_up(timedelta(days=15))

        anchor = self.symbols["SPY"]
        self.schedule.on(
            self.date_rules.week_end(anchor),
            self.time_rules.before_market_close(anchor, 15),
            self.rebalance,
        )

    def rebalance(self):
        if self.is_warming_up:
            return
        need = self.cfg.min_history + 5
        history = self.history(list(self.symbols.values()), need, Resolution.DAILY)
        if history.empty:
            return
        prices = history["close"].unstack(level=0)
        prices.columns = [str(c).split(" ")[0] for c in prices.columns]
        prices = prices.sort_index().dropna(how="all")
        if len(prices) < self.cfg.min_history:
            return

        weights = target_weights(prices, self.cfg)
        scale = self.leverage * (1.0 - self.cash_buffer)
        priced = {t: self.securities[self.symbols[t]].price > 0 for t in weights.index}
        missing = [t for t, w in weights.items() if w > 0 and not priced[t]]
        if missing:
            self.debug(f"{self.time:%Y-%m-%d} no price yet for {missing}; skipping rebalance")
            return
        targets = [PortfolioTarget(self.symbols[t], round(float(w) * scale, 6)) for t, w in weights.items() if priced[t]]
        self.log("WEIGHTS " + prices.index[-1].strftime("%Y-%m-%d") + " " + " ".join(f"{t}={w:.4f}" for t, w in weights.items() if w > 0))
        self.set_holdings(targets, liquidate_existing_holdings=True)
        self.plot("Allocation", "Invested", sum(h.holdings_value for h in self.portfolio.values()) / self.portfolio.total_portfolio_value)
