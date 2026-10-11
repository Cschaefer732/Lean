"""Small vectorised research engine: drift-aware, lagged execution, proportional costs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
CORE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "TIP", "GLD", "DBC"]
CASH = "SHY"


def load(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (total-return daily returns, adjusted closes) on the common date range."""
    px = pd.DataFrame(
        {s: pd.read_csv(DATA / f"{s}.csv", index_col=0, parse_dates=True)["adjclose"] for s in symbols}
    ).dropna()
    return px.pct_change().dropna(), px


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).last().values)


def backtest(
    target: pd.DataFrame,
    rets: pd.DataFrame,
    cost_bps: float = 5.0,
    lag: int = 1,
    borrow_bps: float = 0.0,
    cash_rets: pd.Series | None = None,
) -> pd.DataFrame:
    """target: rows = decision dates (close), cols = assets, values = portfolio weights (may sum to <1; remainder in
    cash earning `cash_rets`, or may sum to >1; the excess is borrowed at cash + `borrow_bps`/yr).

    `lag` counts trading days between the decision close and the execution close:
      lag=1 -> trade at the same close the decision was made on (market-on-close order; new weights earn from d+1)
      lag=2 -> trade at the next day's close (conservative: signal is one full day stale)
    lag=0 would let weights earn the decision day's own return (look-ahead) and is rejected.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1; lag=0 is look-ahead bias")
    idx = rets.index
    cols = list(rets.columns)
    R = rets.values
    cash = np.zeros(len(idx)) if cash_rets is None else cash_rets.reindex(idx).fillna(0).values
    pos_of = {d: i for i, d in enumerate(idx)}
    exec_at: dict[int, np.ndarray] = {}
    for d, row in target.reindex(columns=cols).fillna(0.0).iterrows():
        if d in pos_of and pos_of[d] + lag < len(idx):
            exec_at[pos_of[d] + lag] = row.values

    w = np.zeros(len(cols))
    out_ret = np.zeros(len(idx))
    turnover = np.zeros(len(idx))
    exposure = np.zeros(len(idx))
    for t in range(len(idx)):
        if t in exec_at:
            new = exec_at[t]
            turnover[t] = np.abs(new - w).sum()
            w = new.copy()
        gross = (w * R[t]).sum()
        invested = w.sum()
        cash_w = 1.0 - invested
        fin = cash[t] * cash_w if cash_w >= 0 else (cash[t] + borrow_bps / 1e4 / 252) * cash_w
        out_ret[t] = gross + fin - turnover[t] * cost_bps / 1e4
        exposure[t] = invested
        w = w * (1 + R[t])
        w = w / (1 + gross + fin) if (1 + gross + fin) != 0 else w
    return pd.DataFrame({"ret": out_ret, "turnover": turnover, "exposure": exposure}, index=idx)


def stats(ret: pd.Series, rf: pd.Series | None = None) -> dict[str, float]:
    ret = ret.dropna()
    n = len(ret)
    if n < 30:
        return {}
    eq = (1 + ret).cumprod()
    years = n / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    ex = ret - (rf.reindex(ret.index).fillna(0) if rf is not None else 0)
    sharpe = ex.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    dd = eq / eq.cummax() - 1
    downside = ex[ex < 0].std() * np.sqrt(252)
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": ex.mean() * 252 / downside if downside > 0 else np.nan,
        "MaxDD": dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
    }


def fmt(name: str, s: dict[str, float], turnover: float | None = None) -> str:
    if not s:
        return f"{name:38s} n/a"
    t = f"  TO/yr {turnover:5.1f}x" if turnover is not None else ""
    return (
        f"{name:38s} CAGR {s['CAGR']*100:6.2f}%  Vol {s['Vol']*100:5.2f}%  Sharpe {s['Sharpe']:5.2f}  "
        f"Sortino {s['Sortino']:5.2f}  MaxDD {s['MaxDD']*100:6.2f}%  Calmar {s['Calmar']:5.2f}{t}"
    )
