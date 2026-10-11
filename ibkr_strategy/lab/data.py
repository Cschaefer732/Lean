from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402  (research backtest engine)

RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "TIP", "GLD", "DBC"]
DEFENSIVE = ["IEF", "TLT", "GLD"]
CASH = "SHY"
CLASS = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "bond": ["TLT", "IEF", "LQD", "TIP"],
    "real": ["VNQ", "GLD", "DBC"],
}
MOM_LOOKBACKS = (5, 21, 63, 126, 252)
SEQ_LEN = 64


@dataclass
class Market:
    """Everything the strategies need, sampled once at weekly decision dates (last trading day of each week).

    Arrays are indexed [t, asset] over `dates`; `t` is a decision date and uses only information up to that close.
    """

    px: pd.DataFrame            # daily adjusted closes (risk assets + cash)
    rets: pd.DataFrame          # daily total returns
    dates: pd.DatetimeIndex     # weekly decision dates
    assets: list[str]
    feat_names: list[str]
    X: np.ndarray               # [T, N, F] features
    y: np.ndarray               # [T, N] forward excess return (vs cash) over `horizon` weeks, NaN where unknown
    r1: np.ndarray              # [T, N] return over the NEXT week (t -> t+1)
    r1c: np.ndarray             # [T] next-week cash return
    vol: np.ndarray             # [T, N] 63d annualised vol
    mom: np.ndarray             # [L, T, N] trailing returns for MOM_LOOKBACKS
    above_sma: dict[int, np.ndarray]  # sma length -> bool [T, N]
    seq: np.ndarray             # [T, N, SEQ_LEN] vol-scaled daily returns ending at t
    horizon: int
    defensive_mask: np.ndarray  # [N] bool

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.assets)


def load_prices(symbols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return engine.load(symbols or RISK + [CASH])


def _rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_market(px: pd.DataFrame | None = None, rets: pd.DataFrame | None = None, horizon: int = 4) -> Market:
    if px is None:
        rets, px = load_prices()
    assert rets is not None
    assets = [a for a in RISK if a in px.columns]
    P = px[assets]
    R = rets[assets]

    mom = {lb: P / P.shift(lb) - 1 for lb in MOM_LOOKBACKS}
    vol21 = R.rolling(21).std() * np.sqrt(252)
    vol63 = R.rolling(63).std() * np.sqrt(252)
    dd252 = P / P.rolling(252).max() - 1
    sma50 = P / P.rolling(50).mean() - 1
    sma200 = P / P.rolling(200).mean() - 1
    rsi = _rsi(P)
    avg_mom = sum(mom[lb] for lb in (21, 63, 126, 252)) / 4
    cs_rank = avg_mom.rank(axis=1, pct=True)
    cs_z21 = (mom[21].sub(mom[21].mean(axis=1), axis=0)).div(mom[21].std(axis=1), axis=0)

    spy = "SPY"
    mkt = pd.DataFrame(
        {
            "mkt_vol63": vol63[spy],
            "mkt_vol_ratio": vol21[spy] / vol63[spy],
            "mkt_ret252": mom[252][spy],
            "mkt_dd": dd252[spy],
            "mkt_bond_vs_stock": mom[63]["TLT"] - mom[63][spy],
        }
    )

    per_asset = {
        "ret_5": mom[5], "ret_21": mom[21], "ret_63": mom[63], "ret_126": mom[126], "ret_252": mom[252],
        "vol_21": vol21, "vol_63": vol63, "vol_ratio": vol21 / vol63, "dd_252": dd252,
        "sma50_dist": sma50, "sma200_dist": sma200, "rsi_14": rsi / 100, "cs_rank": cs_rank, "cs_z_ret21": cs_z21,
    }
    names = list(per_asset) + list(mkt.columns) + [f"cls_{c}" for c in CLASS]

    dates = engine.week_ends(px.index)
    first_ok = px.index[np.where(np.arange(len(px)) >= 252)[0][0]]
    dates = dates[dates >= first_ok]
    T, N = len(dates), len(assets)

    X = np.zeros((T, N, len(names)), dtype=np.float64)
    for k, df in enumerate(per_asset.values()):
        X[:, :, k] = df.loc[dates].values
    for j, c in enumerate(mkt.columns):
        X[:, :, len(per_asset) + j] = mkt.loc[dates, c].values[:, None]
    for j, members in enumerate(CLASS.values()):
        X[:, :, len(per_asset) + len(mkt.columns) + j] = np.array([a in members for a in assets], dtype=float)[None, :]

    pw = px.loc[dates]
    nxt = pw.shift(-1) / pw - 1
    fwd = pw.shift(-horizon) / pw - 1
    y = (fwd[assets].sub(fwd[CASH], axis=0)).values

    zs = {n: (R[n] / vol63[n].replace(0, np.nan)) for n in assets}
    ret_scaled = pd.DataFrame(zs) / np.sqrt(252)
    pos = ret_scaled.index.get_indexer(dates)
    seq = np.zeros((T, N, SEQ_LEN), dtype=np.float32)
    rs = ret_scaled.values
    for t, p in enumerate(pos):
        seq[t] = rs[p - SEQ_LEN + 1 : p + 1].T

    return Market(
        px=px, rets=rets, dates=dates, assets=assets, feat_names=names,
        X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y=y,
        r1=nxt[assets].values, r1c=nxt[CASH].values, vol=vol63.loc[dates].values,
        mom=np.stack([mom[lb].loc[dates].values for lb in MOM_LOOKBACKS]),
        above_sma={n: (P > P.rolling(n).mean()).loc[dates].values for n in (100, 150, 200)},
        seq=np.nan_to_num(seq), horizon=horizon,
        defensive_mask=np.array([a in DEFENSIVE for a in assets]),
    )


def rank_target(y: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank of each row's forward excess return, centred at 0 (NaN preserved)."""
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(len(y)):
        ok = np.isfinite(y[i])
        if ok.sum() > 1:
            r = y[i][ok].argsort().argsort()
            out[i, ok] = r / (ok.sum() - 1) - 0.5
    return out


def momentum_gate(mkt: "Market", rows: np.ndarray) -> np.ndarray:
    """Production absolute-momentum rule: blended 1/3/6/12-month return > 0. Returns bool [len(rows), N]."""
    avg = sum(mkt.mom[k][rows] for k in (1, 2, 3, 4)) / 4
    return np.isfinite(avg) & (avg > 0)
