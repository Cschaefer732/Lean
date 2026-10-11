from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_rotation import StrategyConfig, target_weights
from momentum_rotation.signals import decision_dates, single_signal

CFG = StrategyConfig()
DATA = Path(__file__).resolve().parents[1] / "research" / "data"


def synthetic_prices(n=400, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    drift = drift or {}
    cols = {}
    for s in CFG.symbols:
        mu = drift.get(s, 0.0003)
        cols[s] = 100 * np.cumprod(1 + rng.normal(mu, 0.01, n))
    return pd.DataFrame(cols, index=idx)


def test_weights_are_long_only_and_fully_allocated():
    w = target_weights(synthetic_prices(seed=1), CFG)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    assert set(w.index) == set(CFG.symbols)


def test_no_lookahead():
    prices = synthetic_prices(seed=2)
    a = target_weights(prices.iloc[:300], CFG)
    mutated = prices.copy()
    mutated.iloc[300:] *= 5.0
    b = target_weights(mutated.iloc[:300], CFG)
    pd.testing.assert_series_equal(a, b)


def test_bear_market_goes_to_cash():
    prices = synthetic_prices(seed=3, drift={s: -0.003 for s in CFG.symbols})
    w = single_signal(prices, CFG)
    assert w.sum() == 0.0
    tw = target_weights(prices, CFG)
    assert tw[CFG.cash] == pytest.approx(1.0)


def test_idle_slots_go_to_best_defensive_asset():
    drift = {s: -0.003 for s in CFG.symbols}
    drift["SPY"] = 0.002
    drift["GLD"] = 0.0015
    drift["IEF"] = 0.0005
    w = single_signal(synthetic_prices(seed=4, drift=drift), CFG)
    assert w["SPY"] > 0
    assert w["GLD"] >= (CFG.top_k - 1) / CFG.top_k - 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_each_pick_capped_by_slot_when_fully_invested():
    w = single_signal(synthetic_prices(seed=5, drift={s: 0.002 for s in CFG.symbols}), CFG)
    assert (w[list(CFG.risk_universe)] > 0).sum() == CFG.top_k
    assert w.sum() == pytest.approx(1.0)


def test_asset_with_short_history_is_ignored_not_fatal():
    prices = synthetic_prices(seed=6)
    prices.loc[prices.index[:200], "DBC"] = np.nan
    w = target_weights(prices, CFG)
    assert w["DBC"] == 0.0
    assert w.sum() == pytest.approx(1.0)


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        target_weights(synthetic_prices(n=100), CFG)


def test_decision_dates_use_last_trading_day_of_prior_weeks():
    idx = pd.bdate_range("2024-01-01", periods=12)  # ends Tue 2024-01-16
    idx = idx.delete(idx.get_loc(pd.Timestamp("2024-01-12")))  # Friday holiday -> Thursday is the week's last bar
    dates = decision_dates(idx, 3)
    assert dates == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-11"), pd.Timestamp("2024-01-16")]
    assert decision_dates(idx, 1) == [idx[-1]]


@pytest.mark.skipif(not (DATA / "SPY.csv").exists(), reason="run research/fetch_data.py first")
def test_parity_with_research_implementation():
    """The production code must reproduce the research strategy bit-for-bit, otherwise research numbers are moot."""
    from engine import CASH, CORE, load, week_ends
    from strategies import rotation

    rets, px = load(CORE + [CASH])
    sig = rotation(px, rets, CORE, freq="W", top_k=5, defensive=["IEF", "TLT", "GLD"])
    research = (sum(sig.shift(j).fillna(0) for j in range(4)).iloc[3:] / 4)
    dates = research.index[[300, 450, 640, 800, 1000]]
    for d in dates:
        ours = target_weights(px.loc[:d], CFG)
        theirs = research.loc[d]
        pd.testing.assert_series_equal(
            ours[CORE].sort_index(), theirs[CORE].sort_index(), check_names=False, atol=1e-10, rtol=0
        )
