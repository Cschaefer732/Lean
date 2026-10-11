import numpy as np
import pandas as pd
import pytest

from lab.construct import scores_to_weights, smooth
from lab.data import CASH, RISK, build_market
from lab.metrics import (backtest_weights, block_bootstrap_sharpe_diff, deflated_sharpe, information_coefficient)
from lab.models import REGISTRY, Ensemble, get_factory, register
from lab.models.ga import GAConfig, decode, evolve, fast_returns, fitness, genome_weights
from lab.models.rl import RLConfig, RLStrategy
from lab.walkforward import schedule, walk_forward

ALL = RISK + [CASH]


def make_px(n=3300, seed=0, drifts=None, trending=False):
    """Synthetic prices. trending=True gives each asset a slowly-varying drift (momentum is then genuinely predictive)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    cols = {}
    for s in ALL:
        vol = 0.002 if s == CASH else 0.011
        if trending and s != CASH:
            d = np.zeros(n)
            for t in range(1, n):  # slow Ornstein-Uhlenbeck drift, ~6 month memory
                d[t] = d[t - 1] * 0.992 + rng.normal(0, 0.00035)
            mu = d
        else:
            mu = (drifts or {}).get(s, 0.0002 if s == CASH else 0.0002)
        cols[s] = 100 * np.cumprod(1 + mu + rng.normal(0, vol, n))
    px = pd.DataFrame(cols, index=idx)
    return px.pct_change().dropna(), px


@pytest.fixture(scope="module")
def noise_mkt():
    rets, px = make_px(seed=1)
    return build_market(px, rets)


@pytest.fixture(scope="module")
def trend_mkt():
    rets, px = make_px(seed=2, trending=True)
    return build_market(px, rets)


def test_features_do_not_use_future_prices():
    rets, px = make_px(n=900, seed=3)
    cut = 600
    a = build_market(px, rets)
    px2 = px.copy()
    px2.iloc[cut + 1 :] *= np.random.default_rng(9).uniform(0.5, 2.0, px2.iloc[cut + 1 :].shape)
    b = build_market(px2, px2.pct_change().dropna())
    ok = np.where(a.dates <= px.index[cut])[0]
    assert len(ok) > 20
    for name in ("X", "vol", "seq"):
        np.testing.assert_allclose(getattr(a, name)[ok], getattr(b, name)[ok], err_msg=name)
    np.testing.assert_allclose(a.mom[:, ok], b.mom[:, ok])


def test_walkforward_blocks_are_purged_and_contiguous(noise_mkt):
    blocks = schedule(noise_mkt, "2012-01-01", 52)
    prev_end = None
    for train, test in blocks:
        assert train[-1] + noise_mkt.horizon < test[0]  # last training label window ended before the test block
        assert train[0] == 0 and np.all(np.diff(train) == 1)
        if prev_end is not None:
            assert test[0] == prev_end + 1
        prev_end = test[-1]
    assert prev_end == noise_mkt.T - 1


def test_leak_detector_no_skill_on_pure_noise(noise_mkt):
    """On driftless random-walk data nothing is predictable; a leaking harness would show a large positive IC."""
    for spec in ("rf", "gbm"):
        W, S = walk_forward(get_factory(spec), noise_mkt, "2012-01-01", 104, return_scores=True)
        ic = information_coefficient(S, noise_mkt)
        assert abs(ic["IC"]) < 0.06, (spec, ic)


def test_models_find_planted_momentum(trend_mkt):
    W, S = walk_forward(get_factory("rf"), trend_mkt, "2012-01-01", 104, return_scores=True)
    ic = information_coefficient(S, trend_mkt)
    assert ic["IC"] > 0.05 and ic["IC_t"] > 2, ic


def test_weights_are_long_only_and_capped(noise_mkt):
    W = walk_forward(get_factory("gbm"), noise_mkt, "2012-01-01", 104)
    assert (W.values >= -1e-12).all() and (W.sum(axis=1) <= 1 + 1e-9).all()


def test_scores_to_weights_rules(noise_mkt):
    rows = np.arange(300, 303)
    S = np.full((3, noise_mkt.N), -1.0)
    S[:, 0] = 0.05
    W = scores_to_weights(S, noise_mkt, rows, top_k=5)
    assert W[:, 0].sum() == pytest.approx(3 * 0.2)  # one pick owns only 1 of 5 slots
    defensive = noise_mkt.defensive_mask
    assert W[:, ~defensive].sum() == pytest.approx(0.6)  # remaining slots stay cash when nothing defensive is positive
    S[:, 6] = 0.01  # a positive defensive asset (TLT) absorbs the idle slots
    assert scores_to_weights(S, noise_mkt, rows, top_k=5).sum(axis=1) == pytest.approx(1.0)


def test_smoothing_is_trailing_mean():
    W = np.arange(10, dtype=float).reshape(10, 1)
    out = smooth(W, 4)
    assert out[9, 0] == pytest.approx(7.5) and out[0, 0] == 0.0 and out[1, 0] == 0.5


def test_ga_decode_and_weight_invariants(trend_mkt):
    rng = np.random.default_rng(0)
    rows = np.arange(260, 500)
    for _ in range(25):
        g = rng.random(11)
        p = decode(g)
        assert 2 <= p["top_k"] <= 8 and p["lb_w"].sum() == pytest.approx(1)
        W = genome_weights(trend_mkt, g, rows)
        assert (W >= -1e-12).all() and (W.sum(axis=1) <= 1 + 1e-9).all()
        assert (W > 1e-9).sum(axis=1).max() <= p["top_k"] + 1


def test_ga_improves_fitness_and_beats_random_population(trend_mkt):
    rows = np.arange(260, 560)
    hist = []
    pop, fit = evolve(trend_mkt, rows, 0, GAConfig(pop=40, generations=12, runs=1), hist)
    assert hist[-1][1] >= hist[0][1] and hist[-1][2] > hist[0][2]  # best never regresses (elitism), mean improves
    assert fit[0] == pytest.approx(fitness(fast_returns(trend_mkt, genome_weights(trend_mkt, pop[0], rows), rows)))


def test_ga_strategy_adapts_out_of_sample(trend_mkt):
    W = walk_forward(lambda: get_factory("ga")(), trend_mkt, "2012-01-01", 104)
    assert W.shape[1] == trend_mkt.N and W.sum(axis=1).max() > 0.5


def test_rl_agent_learns_to_overweight_the_winning_asset():
    drifts = {s: 0.0 for s in RISK}
    drifts["SPY"] = 0.0015
    rets, px = make_px(n=2800, seed=4, drifts=drifts)
    mkt = build_market(px, rets)
    rl = RLStrategy(RLConfig(iterations=150, batch=128, seeds=1), seed=0)
    rl.fit(mkt, np.arange(0, 330))
    w = rl.weights(mkt, np.arange(340, 430))
    mean_w = w.mean(axis=0)
    assert mean_w.argmax() == mkt.assets.index("SPY") and mean_w.max() > 3 / (mkt.N + 1)
    assert (w >= 0).all() and (w.sum(axis=1) <= 1 + 1e-6).all()


def test_torch_mlp_fits_and_predicts_on_planted_signal(trend_mkt):
    from lab.models.nn import TorchStrategy

    s = TorchStrategy("mlp", seeds=1, epochs=15)
    s.fit(trend_mkt, np.arange(0, 330))
    S = s.scores(trend_mkt, np.arange(340, 460))
    assert S.shape == (120, trend_mkt.N) and np.isfinite(S).all()


def test_ensemble_averages_members_and_registry():
    class Fixed:
        name = "fixed"

        def __init__(self, v):
            self.v = v

        def fit(self, mkt, rows):
            self.fitted = True

        def weights(self, mkt, rows):
            return np.full((len(rows), 2), self.v)

    e = Ensemble([Fixed(0.2), Fixed(0.4)], "ens")
    e.fit(None, np.arange(3))
    np.testing.assert_allclose(e.weights(None, np.arange(3)), 0.3)

    register("fixed_test")(lambda: Fixed(0.1))
    assert "fixed_test" in REGISTRY and get_factory("fixed_test")().v == 0.1
    assert isinstance(get_factory("ens:fixed_test+fixed_test")(), Ensemble)
    with pytest.raises(KeyError):
        get_factory("nope")
    REGISTRY.pop("fixed_test")


def test_backtest_weights_matches_engine_for_constant_book(noise_mkt):
    dates = noise_mkt.dates[300:400]
    W = pd.DataFrame(0.0, index=dates, columns=noise_mkt.assets)
    W["SPY"] = 1.0
    bt = backtest_weights(W, noise_mkt, smooth_weeks=1).loc[dates[0] :]
    spy = noise_mkt.rets["SPY"].loc[bt.index]
    assert np.corrcoef(bt["ret"].iloc[5:], spy.iloc[5:])[0, 1] > 0.999


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    ret = pd.Series(rng.normal(0.0006, 0.01, 1500), index=idx)
    rf = pd.Series(0.0, index=idx)
    few = deflated_sharpe(ret, rf, rng.normal(0.02, 0.03, 3))
    many = deflated_sharpe(ret, rf, rng.normal(0.02, 0.03, 300))
    assert many < few
    noise = pd.Series(rng.normal(0.0, 0.01, 1500), index=idx)
    assert deflated_sharpe(noise, rf, rng.normal(0.0, 0.03, 50)) < 0.5


def test_bootstrap_identical_series_is_a_coin_flip():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=800)
    a = pd.Series(rng.normal(0.0004, 0.01, 800), index=idx)
    out = block_bootstrap_sharpe_diff(a, a, pd.Series(0.0, index=idx), n=200)
    assert abs(out["diff"]) < 1e-9


def test_information_coefficient_perfect_and_inverse(noise_mkt):
    d = noise_mkt.dates[300:340]
    y = pd.DataFrame(noise_mkt.y, index=noise_mkt.dates, columns=noise_mkt.assets).loc[d]
    assert information_coefficient(y, noise_mkt)["IC"] == pytest.approx(1.0)
    assert information_coefficient(-y, noise_mkt)["IC"] == pytest.approx(-1.0)
