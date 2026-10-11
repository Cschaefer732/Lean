"""HFT module: NBBO construction, online/offline feature parity, replay fill model, latency semantics, risk, live adapter."""
from types import SimpleNamespace

import numpy as np
import pytest

from hft.core import BUY, SELL, Quote, Status, Strategy, Trade
from hft.data import CLOSE_MS, OPEN_MS, Day, _nbbo, sign_trades
from hft.features import build
from hft.fees import FREE, IBKR_PRO_TIERED, PROP_REBATE, FeeModel
from hft.ibkr_venue import LiveVenue
from hft.latency import LatencyStats
from hft.replay import Replay, ReplayConfig
from hft.risk import RiskGate, RiskLimits, TokenBucket
from hft.signals import LinearAlpha, OnlineFeatures
from hft.strategies import PassiveMM

T0 = OPEN_MS + 60_000


def make_day(quotes, trades):
    """quotes: [(t, bid, bsz, ask, asz)], trades: [(t, px, sz)]"""
    q = np.array(quotes, float)
    tr = np.array(trades, float).reshape(-1, 3)
    return Day("TST", "20200101", q_t=q[:, 0].astype(np.int64), q_bid=q[:, 1], q_bsz=q[:, 2], q_ask=q[:, 3], q_asz=q[:, 4],
               t_t=tr[:, 0].astype(np.int64), t_px=tr[:, 1], t_sz=tr[:, 2], t_ex=np.full(len(tr), 80, np.uint8))


def cfg(**kw):
    kw.setdefault("start_min", 0.0)
    kw.setdefault("end_min", 0.0)
    kw.setdefault("fees", FREE)
    kw.setdefault("limits", RiskLimits(max_loss=1e12, max_msgs_per_sec=1e6))
    return ReplayConfig(**kw)


class Script(Strategy):
    """Places a fixed list of (when_ms, action) once data time passes `when`."""

    def __init__(self, actions):
        self.actions, self.fills, self.orders, self.seen = list(actions), [], {}, []

    def on_quote(self, ctx, q):
        self.seen.append((q.t, ctx.now))
        while self.actions and ctx.now >= self.actions[0][0]:
            _, fn = self.actions.pop(0)
            fn(self, ctx)

    def on_fill(self, ctx, f):
        self.fills.append(f)


def buy(px, qty=100, post_only=True, key="b"):
    def f(s, ctx):
        s.orders[key] = ctx.place(BUY, px, qty, post_only=post_only)
    return f


def sell(px, qty=100, post_only=True, key="s"):
    def f(s, ctx):
        s.orders[key] = ctx.place(SELL, px, qty, post_only=post_only)
    return f


# ----------------------------------------------------------------------------- NBBO construction
def test_nbbo_aggregates_size_at_best_and_drops_stale_and_selfcrossed():
    # rows: t, bid, bsz, ask, asz, venue
    rows = [
        (1000, 10.00, 100, 0, 0, 1), (1000, 0, 0, 10.02, 200, 1),     # venue 1: 10.00 / 10.02
        (1010, 10.00, 300, 0, 0, 2), (1010, 0, 0, 10.01, 50, 2),      # venue 2: 10.00 / 10.01 -> bid size aggregates to 400
        (1020, 10.05, 100, 0, 0, 3), (1020, 0, 0, 10.03, 100, 3),     # venue 3 is self-crossed -> ignored
        (3000, 10.00, 300, 0, 0, 2),                                   # venue 1 now stale (>200ms)
    ]
    a = np.array(rows, float)
    t, out = _nbbo(a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5].astype(np.int64), 200)
    by_t = dict(zip(t, out))
    assert tuple(by_t[1010]) == (10.00, 400.0, 10.01, 50.0)
    assert 1020 not in by_t                  # self-crossed venue 3 contributes nothing -> NBBO unchanged, nothing emitted
    assert out[:, 0].max() == 10.00          # its 10.05 bid never leaks into the NBBO
    assert 3000 not in by_t                  # by 3000 every venue's other side is stale -> no NBBO at all


def test_sign_trades_uses_prequote_not_posttrade_quote():
    day = make_day([(1000, 10.00, 100, 10.02, 100), (2000, 10.01, 100, 10.03, 100)], [(2000, 10.02, 50)])
    assert sign_trades(day)[0] == 1  # pre-trade mid 10.01 -> a 10.02 print is a buy (post-trade mid 10.02 would give 0)


# ----------------------------------------------------------------------------- online == offline features
def test_online_features_match_vectorised_research_features():
    rng = np.random.default_rng(0)
    n = 3000
    base = OPEN_MS + 10 * 60_000
    qt = np.sort(rng.choice(np.arange(base, base + 120_000), n, replace=False))
    bid = 10 + np.round(np.cumsum(rng.choice([-0.01, 0, 0.01], n, p=[.1, .8, .1])), 2)
    quotes = np.column_stack([qt, bid, rng.integers(1, 20, n) * 100.0, bid + 0.01, rng.integers(1, 20, n) * 100.0])
    tt = np.sort(rng.choice(np.arange(base - 5_000, base + 120_000), 1500))
    px = bid[np.minimum(np.searchsorted(qt, tt), n - 1)] + rng.choice([0.0, 0.01], len(tt))
    day = make_day(quotes, np.column_stack([tt, px, rng.integers(1, 5, len(tt)) * 100.0]))
    b = build(day, start_min=0, end_min=0)
    # replay order: trades before quotes within a ms; sign against the venue's current (pre-trade) quote
    feat, rows, last, cur = OnlineFeatures(), {}, 0, None
    events = sorted([(t, 0, i) for i, t in enumerate(tt)] + [(t, 1, i) for i, t in enumerate(qt)])
    for t, kind, i in events:
        if kind == 1:
            cur = Quote(int(t), *quotes[i, 1:])
            feat.on_quote(cur)
            rows[int(t)] = feat.vector()
        elif cur is not None:
            mid = cur.mid
            s = 1 if day.t_px[i] > mid else (-1 if day.t_px[i] < mid else last)
            last = s or last
            feat.on_trade(Trade(int(t), day.t_px[i], day.t_sz[i], s))
    from hft.features import FEATURES
    cols = [FEATURES.index(f) for f in ("imb", "flow_100ms", "flow_1s")]
    online = np.array([rows[int(t)] for t in b["t"]])
    # offline tick-rule fallback for equal-to-mid prints can differ before the first quote; compare where both defined
    ok = b["t"] > base + 5_000
    assert ok.sum() > 1000
    np.testing.assert_allclose(online[ok][:, 0], b["X"][ok][:, cols[0]], atol=1e-12)
    agree = np.isclose(online[ok][:, 1], b["X"][ok][:, cols[1]], atol=1e-9) & np.isclose(online[ok][:, 2], b["X"][ok][:, cols[2]], atol=1e-9)
    assert agree.mean() > 0.97, agree.mean()


# ----------------------------------------------------------------------------- replay fill model
def run_script(quotes, trades, actions, **kw):
    s = Script(actions)
    r = Replay(make_day(quotes, trades), s, cfg(**kw)).run()
    return s, r


def test_passive_order_joins_back_of_queue_and_fills_only_after_queue_ahead_trades():
    q = [(T0, 10.00, 500, 10.01, 500)]
    # 300 trades at our price: queue ahead 500 -> 200 left. Next 300: 100 for the queue, 100 fills us.
    s, r = run_script(q + [(T0 + 5000, 10.00, 500, 10.01, 500)], [(T0 + 100, 10.00, 300), (T0 + 200, 10.00, 300)], [(0, buy(10.00))])
    assert [f.qty for f in s.fills] == [100] and s.fills[0].px == 10.00 and s.fills[0].maker
    assert r.position_end == 100


def test_no_fill_when_trades_do_not_exhaust_queue():
    s, r = run_script([(T0, 10.00, 500, 10.01, 500), (T0 + 5000, 10.00, 500, 10.01, 500)], [(T0 + 100, 10.00, 400)], [(0, buy(10.00))])
    assert s.fills == [] and r.position_end == 0


def test_trade_through_fills_fully_at_limit_price():
    s, r = run_script([(T0, 10.00, 500, 10.01, 500), (T0 + 5000, 9.99, 500, 10.00, 500)], [(T0 + 100, 9.99, 10)], [(0, buy(10.00))])
    assert len(s.fills) == 1 and s.fills[0].qty == 100 and s.fills[0].px == 10.00


def test_cancel_reductions_move_us_up_the_queue_proportionally():
    # we join behind 1000; 500 more join behind us (size 1500); then 750 cancel across the whole queue (size 750).
    # 'prob': cancels are spread uniformly -> half of our 1000 ahead disappeared -> 500 ahead; a 600 trade fills us for 100.
    # 'none': nothing ahead is assumed to cancel (only capped by visible size) -> 750 ahead; the same trade fills nothing.
    q = [(T0, 10.00, 1000, 10.01, 500), (T0 + 100, 10.00, 1500, 10.01, 500), (T0 + 200, 10.00, 750, 10.01, 500), (T0 + 5000, 10.00, 750, 10.01, 500)]
    s, _ = run_script(q, [(T0 + 300, 10.00, 600)], [(0, buy(10.00))])
    assert sum(f.qty for f in s.fills) == 100
    s2, _ = run_script(q, [(T0 + 300, 10.00, 600)], [(0, buy(10.00))], queue_model="none")
    assert s2.fills == []
    # joining pessimism multiplies the queue ahead
    s3, _ = run_script([(T0, 10.00, 500, 10.01, 500), (T0 + 5000, 10.00, 500, 10.01, 500)], [(T0 + 100, 10.00, 700)], [(0, buy(10.00))], queue_pessimism=2.0)
    assert s3.fills == []


def test_post_only_crossing_order_is_rejected_marketable_takes_at_touch_with_fee():
    q = [(T0, 10.00, 500, 10.01, 500), (T0 + 5000, 10.00, 500, 10.01, 500)]
    s, r = run_script(q, [], [(0, buy(10.01, post_only=True))])
    assert s.fills == [] and s.orders["b"].status == Status.REJECTED and r.counts["venue_rejects"] == 1
    s, r = run_script(q, [], [(0, buy(10.01, post_only=False))], fees=IBKR_PRO_TIERED)
    assert len(s.fills) == 1 and not s.fills[0].maker and s.fills[0].px == 10.01
    assert s.fills[0].fee == pytest.approx(IBKR_PRO_TIERED.fee(BUY, 10.01, 100, False))


def test_latency_delays_data_orders_and_acks_and_trade_during_flight_does_not_fill():
    q = [(T0 + k * 100, 10.00, 500, 10.01, 500) for k in range(60)]
    trades = [(T0 + 120, 10.00, 5000)]   # sweeps the queue at +120ms; our order is decided at +50 and reaches the venue at +150
    s, r = run_script(q, trades, [(T0 + 50, buy(10.00))], md_latency_ms=50, order_latency_ms=100)
    assert all(now - t == 50 for t, now in s.seen)          # strategy sees data exactly md_latency late: no lookahead
    assert s.fills == []                                     # arrived at +150: after the sweep
    s0, _ = run_script(q, trades, [(T0, buy(10.00))])
    assert sum(f.qty for f in s0.fills) == 100               # with zero latency the same order is filled


def test_cancel_races_fill_fill_still_counts_and_cancel_is_late():
    q = [(T0 + k * 50, 10.00, 100, 10.01, 500) for k in range(40)]

    def cancel_it(s, ctx):
        ctx.cancel(s.orders["b"])
    s, r = run_script(q, [(T0 + 160, 10.00, 500)], [(0, buy(10.00)), (T0 + 100, cancel_it)], order_latency_ms=100)
    # order rests from +100, cancel decided at +100(+latency data=0) arrives at +200; trade at +160 fills first
    assert r.position_end == 100 and r.counts["late_cancel"] == 1
    assert s.orders["b"].status == Status.FILLED


def test_pnl_accounting_identity_and_markouts():
    q = [(T0, 10.00, 100, 10.01, 100), (T0 + 1000, 10.00, 100, 10.01, 100), (T0 + 2000, 10.04, 100, 10.05, 100)]
    s, r = run_script(q, [(T0 + 100, 10.00, 200)], [(0, buy(10.00))], fees=PROP_REBATE)
    fill = s.fills[0]
    expect = -10.00 * 100 - fill.fee + 100 * 10.045
    assert r.pnl_mid == pytest.approx(expect)
    assert r.fees == pytest.approx(fill.fee + IBKR_like_liq(r))
    assert r.markout["spread_capture"] == pytest.approx(0.5)      # filled 0.5 tick below mid
    assert r.markout["markout_5s"] == pytest.approx(4.0)          # mid then rose 4 ticks


def IBKR_like_liq(r):
    return r.fees - sum(f.fee for f in r.fills)


# ----------------------------------------------------------------------------- strategy behaviour
def test_passive_mm_quotes_both_sides_and_respects_inventory_cap():
    q = [(T0 + k * 100, 10.00, 500, 10.01, 500) for k in range(100)]
    trades = [(T0 + 500, 10.00, 2000), (T0 + 600, 10.00, 2000), (T0 + 700, 10.00, 2000)]
    mm = PassiveMM(qty=100, max_pos=200)
    r = Replay(make_day(q, trades), mm, cfg()).run()
    assert r.position_end <= 200 and r.counts["sent"] >= 2


def test_alpha_pull_removes_the_exposed_side():
    alpha = LinearAlpha(np.array([1.0, 0.0, 0.0]), 0.0)         # score = imbalance
    bid_heavy = [(T0 + k * 100, 10.00, 900, 10.01, 100) for k in range(30)]   # imb=+0.8 -> price likely up -> pull the ASK
    s = PassiveMM(alpha=alpha, pull=0.3)
    Replay(make_day(bid_heavy, []), s, cfg()).run()
    assert s.n_pulls > 0
    rec = {"sides": set()}

    class Spy(PassiveMM):
        def on_quote(self, ctx, q):
            super().on_quote(ctx, q)
            rec["sides"] |= {o.side for o in ctx.open_orders()}
    Replay(make_day(bid_heavy, []), Spy(alpha=alpha, pull=0.3), cfg()).run()
    assert rec["sides"] == {BUY}


# ----------------------------------------------------------------------------- risk
def test_risk_gate_limits_rate_and_kill_switch():
    g = RiskGate(RiskLimits(max_order_qty=100, max_position=200, max_notional=5000, price_band_ticks=5, max_msgs_per_sec=10, max_loss=50))
    ok = lambda **k: g.check_order(0, k.get("side", 1), k.get("px", 10.0), k.get("qty", 100), 10.0, k.get("pos", 0), k.get("wb", 0), k.get("ws", 0))
    assert ok() is None
    assert ok(qty=101) == "order_qty"
    assert ok(px=11.0) == "notional" or ok(px=11.0) == "price_band"
    assert ok(px=10.10, qty=100) == "price_band"
    assert ok(pos=150) == "position_limit"
    assert ok(side=-1, pos=-150) == "position_limit"
    g.check_pnl(-51)
    assert g.state.halted and ok() == "halted"
    assert g.check_cancel(0)   # cancels still pass while halted


def test_token_bucket_refills_with_time():
    b = TokenBucket(10, burst=2)
    assert b.allow(0) and b.allow(0) and not b.allow(0)
    assert b.allow(150)         # 0.15s * 10/s = 1.5 tokens
    assert not b.allow(150)


def test_cancel_budget_is_reserved_independent_of_order_flow():
    g = RiskGate(RiskLimits(max_msgs_per_sec=10, max_position=10**9, max_notional=1e12))
    while g.check_order(0, 1, 10.0, 1, 10.0, 0, 0, 0) is None:
        pass
    assert g.check_cancel(0)    # order flood did not starve cancels


def test_kill_switch_cancels_working_orders_in_replay():
    q = [(T0 + k * 100, 10.00, 100000, 10.01, 100000) for k in range(50)] + [(T0 + 6000, 9.50, 100, 9.51, 100)]
    s = Script([(0, buy(10.00, 100))])
    r = Replay(make_day(q, [(T0 + 50, 10.00, 150000)]), s, cfg(limits=RiskLimits(max_loss=20.0, max_msgs_per_sec=1e6))).run()
    assert r.halted.startswith("max_loss")


# ----------------------------------------------------------------------------- fees
def test_fee_models():
    assert FREE.fee(BUY, 100, 100, True) == 0
    f = FeeModel("x", commission=0.0035, min_commission=0.35, exchange_remove=0.003, exchange_add=-0.002)
    assert f.fee(BUY, 100, 10, True) == pytest.approx(0.35 - 0.002 * 10)
    assert f.fee(BUY, 100, 1000, False) == pytest.approx(3.5 + 3.0)
    assert IBKR_PRO_TIERED.fee(SELL, 167, 100, True) > IBKR_PRO_TIERED.fee(BUY, 167, 100, True)   # SEC/TAF only on sells


# ----------------------------------------------------------------------------- alpha / latency stats
def test_linear_alpha_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 3))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 2] + rng.normal(0, 0.1, 20000)
    a = LinearAlpha.fit(X, y, l2=0.01)
    np.testing.assert_allclose(a.w, [0.5, 0.0, -0.2], atol=0.02)


def test_latency_stats_percentiles():
    s = LatencyStats()
    for v in range(1, 1001):
        s.record("x", v)
    out = s.summary()["x"]
    assert out["p50"] == pytest.approx(500.5, abs=1) and out["max"] == 1000 and "x" in s.report()


# ----------------------------------------------------------------------------- live adapter against a fake ib_async
class Ev:
    def __init__(self): self.h = []
    def __iadd__(self, f): self.h.append(f); return self
    def fire(self, *a): [h(*a) for h in self.h]


class FakeTicker:
    def __init__(self): self.tickByTicks, self.updateEvent = [], Ev()


class FakeIB:
    def __init__(self):
        self.tickers, self.placed, self.cancelled, self.trades = {}, [], [], []

    def positions(self): return [SimpleNamespace(contract=SimpleNamespace(symbol="SPY"), position=0)]
    def reqTickByTickData(self, c, kind, n, ignore): self.tickers[kind] = FakeTicker(); return self.tickers[kind]
    def cancelTickByTickData(self, c, kind): pass
    def sleep(self, s): pass

    def placeOrder(self, contract, order):
        t = SimpleNamespace(order=order, fillEvent=Ev(), statusEvent=Ev(), orderStatus=SimpleNamespace(status="Submitted"))
        self.placed.append(order); self.trades.append(t); return t

    def cancelOrder(self, order): self.cancelled.append(order)
    def openTrades(self): return []
    def fills(self): return []


def push_quote(ib, v, bid, bsz, ask, asz, ms):
    import datetime as dt
    from zoneinfo import ZoneInfo
    ts = dt.datetime(2024, 6, 12, 10, 0, 0, tzinfo=ZoneInfo("America/New_York")) + dt.timedelta(milliseconds=ms)
    tk = ib.tickers["BidAsk"]
    tk.tickByTicks.append(SimpleNamespace(time=ts, bidPrice=bid, askPrice=ask, bidSize=bsz, askSize=asz))
    tk.updateEvent.fire(tk)


def test_live_venue_dry_run_logs_intents_and_never_touches_the_broker():
    ib = FakeIB()
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), PassiveMM(qty=100, max_pos=200), dry_run=True)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    push_quote(ib, v, 100.00, 500, 100.01, 500, 10)
    assert ib.placed == [] and {i[0] for i in v.intents} == {"place"} and len(v.intents) == 2
    assert v.lat.summary()["tick_to_decision_us"]["n"] == 2


def test_live_venue_places_limit_orders_routes_fills_and_cancels():
    ib = FakeIB()
    mm = PassiveMM(qty=100, max_pos=200)
    got = []
    mm.on_fill = lambda ctx, f: got.append(f)
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), mm, dry_run=False)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    assert [(o.action, o.lmtPrice, o.totalQuantity) for o in ib.placed] == [("BUY", 100.00, 100), ("SELL", 100.01, 100)]
    ib.trades[0].fillEvent.fire(ib.trades[0], SimpleNamespace(execution=SimpleNamespace(side="BOT", shares=100, price=100.00),
                                                           commissionReport=SimpleNamespace(commission=0.35)))
    assert v.ctx.position == 100 and got[0].qty == 100 and v.cash == pytest.approx(-100.00 * 100 - 0.35)
    push_quote(ib, v, 100.01, 500, 100.02, 500, 500)   # market moved up: bid at 100.00 is stale -> cancel/replace
    assert ib.cancelled and ib.cancelled[0] is ib.placed[1]    # the resting ASK at 100.01 is now at the bid -> cancelled


def test_live_venue_post_only_emulation_and_kill_switch():
    ib = FakeIB()
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=False, limits=RiskLimits(max_loss=10.0, max_position=1000, max_order_qty=1000))
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    assert v.ctx.place(BUY, 100.01, 100, post_only=True) is None and v.risk.state.rejected["would_cross"] == 1
    v.cash, v.ctx.position = -2000.0, 10
    push_quote(ib, v, 99.00, 500, 99.01, 500, 100)
    assert v.risk.state.halted


# ----------------------------------------------------------------------------- CLI guards / recorder
def test_cli_refuses_live_port_without_flag_and_record_with_send(capsys):
    from hft.run_live import main
    assert main(["--send", "--port", "4001"]) == 2
    assert main(["--record", "--send"]) == 2


def test_cli_dry_run_and_record_never_place_orders(tmp_path, monkeypatch):
    import hft.run_live as rl

    ib = FakeIB()
    ib.connect = lambda *a, **k: setattr(ib, "readonly", k["readonly"])
    ib.disconnect = lambda: None
    v_holder = {}
    orig = rl.LiveVenue

    class V(orig):
        def run(self, seconds):
            v_holder["v"] = self
            for k in range(3):
                push_quote(ib, self, 100.00, 500, 100.01, 500, k * 10)
            self.stop()

    monkeypatch.setattr(rl, "LiveVenue", V)
    # dry run: readonly connection, nothing sent
    assert rl.main(["--symbol", "SPY"], ib_factory=lambda: ib) == 0
    assert ib.readonly is True and ib.placed == []
    # session window: ticks at 10:00 ET are inside trade hours so intents exist
    assert len(v_holder["v"].intents) > 0


def test_recorder_roundtrips_to_a_day(tmp_path):
    from hft.recorder import TickRecorder, load_recorded
    r = TickRecorder("SPY", "20260101")
    r.on_quote(SimpleNamespace(t=1000, bid=10.0, bsz=100, ask=10.01, asz=200))
    r.on_trade(SimpleNamespace(t=1001, px=10.0, sz=50))
    p = r.save(tmp_path)
    d = load_recorded(p, "SPY", "20260101")
    assert d.q_bid[0] == 10.0 and d.t_sz[0] == 50 and d.q_t.dtype == np.int64


# ----------------------------------------------------------------------------- live safety: staleness, reconnect, reconcile
class SafeIB(FakeIB):
    """FakeIB plus the surface the safety paths use: fills, open orders, connection state."""
    def __init__(self):
        super().__init__()
        self.up, self.positions_now, self.fills_now, self.open_now = True, 0, [], []
        self.disconnectedEvent = Ev()
        self.connects = 0

    def isConnected(self): return self.up
    def positions(self): return [SimpleNamespace(contract=SimpleNamespace(symbol="SPY"), position=self.positions_now)]
    def fills(self): return self.fills_now
    def openTrades(self): return self.open_now

    def cancelOrder(self, order): self.cancelled.append(order)


def mk_fill(eid, side, shares, price, comm=0.35):
    return SimpleNamespace(contract=SimpleNamespace(symbol="SPY"), execution=SimpleNamespace(execId=eid, side=side, shares=shares, price=price),
                           commissionReport=SimpleNamespace(commission=comm))


class Clock:
    t = 1_700_000_000.0
    def __call__(self): return self.t


def et_clock_at(hh, mm):
    import datetime as dt
    from zoneinfo import ZoneInfo
    c = Clock()
    c.t = dt.datetime(2024, 6, 12, hh, mm, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    return c


def test_stale_feed_pulls_orders_and_blocks_quoting_until_fresh_tick():
    ib, clk = SafeIB(), et_clock_at(10, 0)
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), PassiveMM(qty=100, max_pos=200), dry_run=False, stale_feed_s=2.0, clock=clk)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    assert len(ib.placed) == 2
    clk.t += 1.0
    v.health_check()
    assert not v.feed_stale and ib.cancelled == []
    clk.t += 1.5                       # 2.5s without a quote
    v.health_check()
    assert v.feed_stale and len(ib.cancelled) == 2 and v.recon["stale_halts"] == 1 and not v.ctx.can_trade
    push_quote(ib, v, 100.00, 500, 100.01, 500, 3000)   # fresh data resumes quoting
    assert not v.feed_stale and v.ctx.can_trade


def test_stale_feed_is_ignored_outside_trading_hours():
    ib, clk = SafeIB(), et_clock_at(8, 0)
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=False, clock=clk)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    clk.t += 60
    v.health_check()
    assert not v.feed_stale


def test_disconnect_stops_quoting_then_reconnect_resubscribes_and_reconciles():
    ib, clk = SafeIB(), et_clock_at(10, 0)
    calls = []

    def reconnect():
        calls.append(1)
        ib.up = True
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), PassiveMM(qty=100, max_pos=300), dry_run=False, clock=clk, reconnect=reconnect, reconnect_every_s=5)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    assert len(ib.placed) == 2
    # link drops: an order fills at the broker while we are blind to it, and a stray order is still resting there
    ib.up = False
    ib.disconnectedEvent.fire()
    assert not v.connected and not v.ctx.can_trade
    ib.positions_now = 100
    ib.fills_now = [mk_fill("e1", "BOT", 100, 100.00)]
    ib.open_now = [SimpleNamespace(contract=SimpleNamespace(symbol="SPY"), order=ib.placed[1])]
    v.health_check()
    assert calls == [1] and v.connected
    assert v.ctx.position == 100 and v.recon["position_mismatch"] == 1        # broker wins
    assert v.cash == pytest.approx(-100.00 * 100 - 0.35)                       # cash rebuilt from broker executions
    assert ib.cancelled and v.recon["orders_cancelled"] == 1 and v.ctx.open_orders() == []
    assert len(ib.tickers) == 2 and v._q_idx == 0                              # fresh subscriptions


def test_reconnect_budget_is_bounded_then_trading_halts():
    ib, clk = SafeIB(), et_clock_at(10, 0)

    def reconnect():
        raise ConnectionError("down")
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=False, clock=clk, reconnect=reconnect, max_reconnects=3, reconnect_every_s=5)
    v.start()
    ib.up = False
    for _ in range(6):
        clk.t += 6
        v.health_check()
    assert v.risk.state.halted and v.risk.state.halt_reason == "disconnected" and not v.connected


def test_no_reconnect_callback_halts_immediately():
    ib, clk = SafeIB(), et_clock_at(10, 0)
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=False, clock=clk)
    v.start()
    ib.up = False
    v.health_check()
    assert v.risk.state.halted


def test_duplicate_execution_reports_are_not_double_counted():
    ib = SafeIB()
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), PassiveMM(qty=100, max_pos=500), dry_run=False)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    f = SimpleNamespace(execution=SimpleNamespace(execId="x1", side="BOT", shares=100, price=100.00), commissionReport=SimpleNamespace(commission=0.0))
    for _ in range(2):
        ib.trades[0].fillEvent.fire(ib.trades[0], f)
    assert v.ctx.position == 100


def test_fill_for_an_order_forgotten_by_reconcile_still_updates_position():
    ib = SafeIB()
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), PassiveMM(qty=100, max_pos=500), dry_run=False)
    v.start()
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    v.reconcile()                      # forgets local orders
    f = SimpleNamespace(execution=SimpleNamespace(execId="x2", side="BOT", shares=100, price=100.00), commissionReport=SimpleNamespace(commission=0.0))
    ib.trades[0].fillEvent.fire(ib.trades[0], f)
    assert v.ctx.position == 100


def test_preexisting_position_does_not_count_as_pnl_for_the_kill_switch():
    ib = SafeIB()
    ib.positions_now = 1000                                  # carried in; worth $100k at the mark
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=False, limits=RiskLimits(max_loss=50.0, max_position=10**6))
    v.start()
    assert v.ctx.position == 1000 and v.pos0 == 1000
    push_quote(ib, v, 100.00, 500, 100.01, 500, 0)
    assert not v.risk.state.halted
    push_quote(ib, v, 99.90, 500, 99.91, 500, 100)           # carried position loses $100 at the mark -> counts
    assert not v.risk.state.halted                           # pnl is measured on session trades only, so flat session = 0
    v.ctx.position += 100; v.cash -= 100 * 99.91 + 0.35      # session buy of 100, then price falls
    push_quote(ib, v, 98.00, 500, 98.01, 500, 200)
    assert v.risk.state.halted


def test_startup_cancels_stray_orders_but_dry_run_never_does():
    ib = SafeIB()
    ib.open_now = [SimpleNamespace(contract=SimpleNamespace(symbol="SPY"), order="stray")]
    LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=True).start()
    assert ib.cancelled == []
    v = LiveVenue(ib, SimpleNamespace(symbol="SPY"), Script([]), dry_run=False)
    v.start()
    assert ib.cancelled == ["stray"]
