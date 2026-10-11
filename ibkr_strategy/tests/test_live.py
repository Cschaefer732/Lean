import datetime as dt
import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from live import trader
from live.broker import IBKRBroker, PlacedOrder, classify_session, parse_liquid_hours
from live.trader import SafetyError, run
from momentum_rotation import StrategyConfig, plan_orders

ET = ZoneInfo("America/New_York")
CFG = StrategyConfig()
FRIDAY_1530 = dt.datetime(2024, 6, 14, 15, 30, tzinfo=ET)


def make_prices(end="2024-06-14", n=420, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    drift = {"SPY": 0.0008, "QQQ": 0.0009, "GLD": 0.0005}
    return pd.DataFrame(
        {s: 100 * np.cumprod(1 + rng.normal(drift.get(s, 0.0001), 0.01, n)) for s in CFG.symbols}, index=idx
    )


class FakeBroker:
    def __init__(self, prices, equity=100_000.0, positions=None, paper=True, session="last"):
        self.account, self.is_paper = "DU123456", paper
        self._prices, self._equity, self._positions, self._session = prices, equity, positions or {}, session
        self.placed = []

    def account_equity(self): return self._equity
    def positions(self): return dict(self._positions)
    def daily_closes(self, symbols, min_bars): return self._prices[symbols]
    def last_prices(self, symbols): return self._prices[symbols].iloc[-1]
    def session_state(self, today): return self._session
    def close(self): pass

    def place(self, orders, order_type, tag, limit_bps):
        self.placed.append((orders, order_type, tag))
        return [PlacedOrder(o.symbol, o.side, abs(o.quantity), order_type, "PreSubmitted", i) for i, o in enumerate(orders)]


def kwargs(tmp_path, **kw):
    return dict(now=FRIDAY_1530, state_dir=tmp_path, **kw)


# ---------- order planning ----------
def test_plan_orders_respects_budget_and_sells_first():
    w = pd.Series({"SPY": 0.5, "TLT": 0.5, "GLD": 0.0})
    px = pd.Series({"SPY": 400.0, "TLT": 100.0, "GLD": 200.0})
    orders = plan_orders(w, px, {"GLD": 100}, 100_000)
    assert [o.symbol for o in orders][0] == "GLD" and orders[0].quantity == -100
    buy_value = sum(o.notional for o in orders if o.quantity > 0)
    assert buy_value <= 100_000 * 0.99


def test_plan_orders_skips_small_drift_but_always_exits():
    w = pd.Series({"SPY": 0.5, "GLD": 0.0})
    px = pd.Series({"SPY": 100.0, "GLD": 100.0})
    orders = plan_orders(w, px, {"SPY": 494, "GLD": 1}, 100_000)  # SPY target 495 -> $100 diff, below 0.5%
    assert [(o.symbol, o.quantity) for o in orders] == [("GLD", -1)]


def test_plan_orders_never_touches_unmanaged_symbols():
    orders = plan_orders(pd.Series({"SPY": 1.0}), pd.Series({"SPY": 100.0}), {"AAPL": 500}, 50_000)
    assert {o.symbol for o in orders} == {"SPY"}


def test_plan_orders_rejects_bad_price():
    with pytest.raises(ValueError):
        plan_orders(pd.Series({"SPY": 1.0}), pd.Series({"SPY": float("nan")}), {}, 1000)


# ---------- session calendar ----------
def test_liquid_hours_parsing_and_last_session_detection():
    lh = ("20240610:0930-20240610:1600;20240611:0930-20240611:1600;20240612:0930-20240612:1600;"
          "20240613:0930-20240613:1600;20240614:0930-20240614:1600;20240615:CLOSED;20240616:CLOSED")
    s = parse_liquid_hours(lh)
    assert classify_session(s, dt.date(2024, 6, 14)) == "last"
    assert classify_session(s, dt.date(2024, 6, 13)) == "open"
    assert classify_session(s, dt.date(2024, 6, 15)) == "closed"


def test_holiday_friday_makes_thursday_the_last_session():
    lh = ("20240701:0930-20240701:1600;20240702:0930-20240702:1600;20240703:0930-20240703:1300;"
          "20240704:CLOSED;20240705:0930-20240705:1600;20240706:CLOSED")
    assert classify_session(parse_liquid_hours(lh), dt.date(2024, 7, 3)) == "open"
    lh2 = lh.replace("20240705:0930-20240705:1600", "20240705:CLOSED")
    assert classify_session(parse_liquid_hours(lh2), dt.date(2024, 7, 3)) == "last"


def test_calendar_that_does_not_reach_friday_is_not_assumed_last():
    lh = "20240610:0930-20240610:1600;20240611:0930-20240611:1600"
    assert classify_session(parse_liquid_hours(lh), dt.date(2024, 6, 11)) == "open"


# ---------- runner ----------
def test_dry_run_sends_nothing_and_reports_plan(tmp_path):
    b = FakeBroker(make_prices())
    rep = run(b, CFG, **kwargs(tmp_path))
    assert not b.placed and rep["executed"] is False
    assert rep["orders"] and abs(sum(rep["weights"].values()) - 1) < 1e-3
    assert all(o["quantity"] > 0 for o in rep["orders"])
    assert (tmp_path / "decision-2024-06-14.json").exists()


def test_execute_places_orders_once_per_day(tmp_path):
    b = FakeBroker(make_prices())
    rep = run(b, CFG, **kwargs(tmp_path, execute=True))
    assert rep["executed"] and len(b.placed) == 1 and b.placed[0][1] == "moc"
    again = run(b, CFG, **kwargs(tmp_path, execute=True))
    assert again["skipped"] == "already executed today" and len(b.placed) == 1


def test_second_run_after_fill_is_a_noop(tmp_path):
    prices = make_prices()
    b = FakeBroker(prices)
    rep = run(b, CFG, **kwargs(tmp_path))
    held = {o["symbol"]: o["quantity"] for o in rep["orders"]}
    b2 = FakeBroker(prices, positions=held)
    rep2 = run(b2, CFG, **kwargs(tmp_path / "x"))
    assert rep2["orders"] == []


def test_live_account_refused_without_flag(tmp_path):
    with pytest.raises(SafetyError, match="LIVE"):
        run(FakeBroker(make_prices(), paper=False), CFG, **kwargs(tmp_path, execute=True))
    rep = run(FakeBroker(make_prices(), paper=False), CFG, **kwargs(tmp_path / "ok", execute=True, allow_live=True))
    assert rep["executed"]


def test_dry_run_on_live_account_is_allowed(tmp_path):
    assert run(FakeBroker(make_prices(), paper=False), CFG, **kwargs(tmp_path))["executed"] is False


def test_halt_file_stops_everything(tmp_path):
    (tmp_path / "HALT").write_text("")
    with pytest.raises(SafetyError, match="kill switch"):
        run(FakeBroker(make_prices()), CFG, **kwargs(tmp_path))


@pytest.mark.parametrize("state,msg", [("open", "not the last"), ("closed", "closed")])
def test_skips_when_not_final_session(tmp_path, state, msg):
    b = FakeBroker(make_prices(), session=state)
    rep = run(b, CFG, **kwargs(tmp_path, execute=True))
    assert msg in rep["skipped"] and not b.placed


def test_stale_data_refused(tmp_path):
    with pytest.raises(SafetyError, match="days old"):
        run(FakeBroker(make_prices(end="2024-06-03")), CFG, **kwargs(tmp_path))


def test_price_glitch_refused(tmp_path):
    prices = make_prices()
    prices.iloc[-1, prices.columns.get_loc("SPY")] *= 0.6
    with pytest.raises(SafetyError, match="implausible"):
        run(FakeBroker(prices), CFG, **kwargs(tmp_path))


def test_missing_symbol_refused(tmp_path):
    with pytest.raises(SafetyError, match="no price history"):
        trader.validate_prices(make_prices().drop(columns=["GLD"]), dt.date(2024, 6, 14), CFG)
    with pytest.raises(SafetyError, match="gaps"):
        run(FakeBroker(make_prices().assign(GLD=np.nan)), CFG, **kwargs(tmp_path))


def test_turnover_cap(tmp_path):
    with pytest.raises(SafetyError, match="turnover"):
        run(FakeBroker(make_prices()), CFG, **kwargs(tmp_path, max_turnover=0.2))


def test_moc_refused_after_cutoff(tmp_path):
    late = dt.datetime(2024, 6, 14, 15, 55, tzinfo=ET)
    with pytest.raises(SafetyError, match="cutoff"):
        run(FakeBroker(make_prices()), CFG, now=late, state_dir=tmp_path, execute=True)
    rep = run(FakeBroker(make_prices()), CFG, now=late, state_dir=tmp_path / "m", execute=True, order_type="mkt")
    assert rep["executed"]


def test_capital_cap_limits_position_sizes(tmp_path):
    full = run(FakeBroker(make_prices(), equity=1_000_000), CFG, **kwargs(tmp_path / "a"))
    capped = run(FakeBroker(make_prices(), equity=1_000_000), CFG, **kwargs(tmp_path / "b", capital=100_000))
    tot = lambda r: sum(abs(o["quantity"]) * o["ref_price"] for o in r["orders"])
    assert tot(capped) == pytest.approx(tot(full) / 10, rel=0.05)


def test_leverage_bounds(tmp_path):
    with pytest.raises(SafetyError):
        run(FakeBroker(make_prices()), CFG, **kwargs(tmp_path, leverage=3.0))


# ---------- IBKR adapter against a faked ib_async.IB ----------
class FakeIB:
    instances = []

    def __init__(self):
        self.connect_args = None
        self.orders = []
        FakeIB.instances.append(self)

    def connect(self, host, port, clientId, readonly, timeout):
        self.connect_args = dict(host=host, port=port, clientId=clientId, readonly=readonly)

    def managedAccounts(self): return ["DU999"]
    def reqMarketDataType(self, t): self.mdt = t
    def sleep(self, s): pass
    def disconnect(self): self.disconnected = True

    def accountValues(self, account):
        return [SimpleNamespace(tag="NetLiquidation", currency="USD", value="250000.5"),
                SimpleNamespace(tag="NetLiquidation", currency="BASE", value="1")]

    def positions(self, account):
        stock = lambda s: SimpleNamespace(symbol=s, secType="STK")
        return [SimpleNamespace(contract=stock("SPY"), position=10.0),
                SimpleNamespace(contract=SimpleNamespace(symbol="ES", secType="FUT"), position=2.0)]

    def reqContractDetails(self, contract):
        c = SimpleNamespace(symbol=contract.symbol, secType="STK", currency="USD", primaryExchange="ARCA")
        other = SimpleNamespace(symbol=contract.symbol, secType="STK", currency="USD", primaryExchange="BATS")
        lh = "20240612:0930-20240612:1600;20240613:0930-20240613:1600;20240614:0930-20240614:1600;20240615:CLOSED"
        return [SimpleNamespace(contract=other, liquidHours=lh), SimpleNamespace(contract=c, liquidHours=lh)]

    def reqHistoricalData(self, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate):
        self.last_hist = (durationStr, whatToShow, useRTH)
        days = pd.bdate_range(end="2024-06-14", periods=5)
        return [SimpleNamespace(date=d.date(), close=100.0 + i) for i, d in enumerate(days)]

    def placeOrder(self, contract, order):
        self.orders.append((contract, order))
        return SimpleNamespace(order=SimpleNamespace(orderId=len(self.orders)), orderStatus=SimpleNamespace(status="PreSubmitted"))


@pytest.fixture
def fake_ib(monkeypatch):
    import ib_async

    FakeIB.instances.clear()
    monkeypatch.setattr(ib_async, "IB", FakeIB)
    return FakeIB


def test_adapter_connects_readonly_for_dry_run(fake_ib):
    b = IBKRBroker("127.0.0.1", 4002, 5, None, readonly=True)
    assert fake_ib.instances[0].connect_args["readonly"] is True
    assert b.account == "DU999" and b.is_paper


def test_adapter_reads_account_and_filters_non_stock_positions(fake_ib):
    b = IBKRBroker("127.0.0.1", 4002, 5, None, readonly=True)
    assert b.account_equity() == 250000.5
    assert b.positions() == {"SPY": 10}


def test_adapter_prefers_primary_listing_and_history_params(fake_ib):
    b = IBKRBroker("127.0.0.1", 4002, 5, None, readonly=True)
    assert b._contract("SPY").primaryExchange == "ARCA"
    closes = b.daily_closes(["SPY"], 300)
    assert len(closes) == 5 and fake_ib.instances[0].last_hist == ("3 Y", "ADJUSTED_LAST", True)
    assert b.last_prices(["SPY"])["SPY"] == 104.0
    assert fake_ib.instances[0].last_hist[1] == "TRADES"
    assert b.session_state(dt.date(2024, 6, 14)) == "last"


@pytest.mark.parametrize("otype,expected", [("moc", "MOC"), ("mkt", "MKT"), ("lmt", "LMT")])
def test_adapter_builds_orders(fake_ib, otype, expected):
    from momentum_rotation import Order

    b = IBKRBroker("127.0.0.1", 4002, 5, None, readonly=False)
    placed = b.place([Order("SPY", -5, 500.0), Order("GLD", 3, 200.0)], otype, "momrot-test", 10.0)
    sent = fake_ib.instances[0].orders
    assert [o.orderType for _, o in sent] == [expected, expected]
    assert [o.action for _, o in sent] == ["SELL", "BUY"]
    assert all(o.orderRef == "momrot-test" and o.account == "DU999" for _, o in sent)
    assert [p.quantity for p in placed] == [5, 3]
    if otype == "lmt":
        assert sent[0][1].lmtPrice == 499.5 and sent[1][1].lmtPrice == 200.2


def test_adapter_flags_live_ports_as_non_paper(fake_ib, monkeypatch):
    monkeypatch.setattr(FakeIB, "managedAccounts", lambda self: ["U1234567"])
    assert IBKRBroker("127.0.0.1", 4001, 5, None, readonly=True).is_paper is False


def test_cli_dry_run_end_to_end(fake_ib, tmp_path, monkeypatch, capsys):
    prices = make_prices()
    monkeypatch.setattr(IBKRBroker, "daily_closes", lambda self, s, n: prices[s])
    monkeypatch.setattr(IBKRBroker, "last_prices", lambda self, s: prices[s].iloc[-1])
    rc = trader.main(["--state-dir", str(tmp_path)], now=FRIDAY_1530)
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out and "BUY" in out
    assert fake_ib.instances[0].connect_args["readonly"] is True and not fake_ib.instances[0].orders
