"""Real tick data loading (AlgoSeek format as shipped in LEAN's Data/equity/usa/tick) and NBBO construction.

Quote rows are per-exchange top-of-book updates: `ms_since_midnight, bid_px*1e4, bid_sz, ask_px*1e4, ask_sz, exch, cond, flag`
where a bid update carries zeros on the ask side and vice versa. We keep each venue's latest bid/ask and emit a
consolidated NBBO (best price across venues, size aggregated across venues at that price) whenever it changes.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
TICK_DIR = ROOT.parent / "Data" / "equity" / "usa" / "tick"
CACHE = ROOT / "hft" / "cache"
OPEN_MS, CLOSE_MS = 34_200_000, 57_600_000  # 09:30 / 16:00 ET in ms since midnight


@dataclass
class Day:
    symbol: str
    date: str
    # consolidated NBBO change events
    q_t: np.ndarray      # int64 ms since midnight
    q_bid: np.ndarray    # float64 dollars
    q_bsz: np.ndarray    # float64 shares (aggregate at best)
    q_ask: np.ndarray
    q_asz: np.ndarray
    # trades
    t_t: np.ndarray
    t_px: np.ndarray
    t_sz: np.ndarray
    t_ex: np.ndarray     # uint8 exchange code (ASCII)


@njit(cache=True)
def _nbbo(ts, bpx, bsz, apx, asz, ex, stale_ms, n_venues=64):
    n = len(ts)
    vb_px = np.zeros(n_venues)
    vb_sz = np.zeros(n_venues)
    va_px = np.zeros(n_venues)
    va_sz = np.zeros(n_venues)
    vb_t = np.zeros(n_venues, np.int64)
    va_t = np.zeros(n_venues, np.int64)
    out_t = np.empty(n, np.int64)
    out = np.empty((n, 4))
    k = 0
    last_b, last_bs, last_a, last_as = -1.0, -1.0, -1.0, -1.0
    for i in range(n):
        v = ex[i] % n_venues
        if bpx[i] > 0:
            vb_px[v] = bpx[i]
            vb_sz[v] = bsz[i]
            vb_t[v] = ts[i]
        if apx[i] > 0:
            va_px[v] = apx[i]
            va_sz[v] = asz[i]
            va_t[v] = ts[i]
        best_b, best_bs = 0.0, 0.0
        best_a, best_as = 1e18, 0.0
        for j in range(n_venues):
            # a venue contributes only while both sides are fresh and not self-crossed
            fresh = ts[i] - vb_t[j] <= stale_ms and ts[i] - va_t[j] <= stale_ms and vb_px[j] < va_px[j]
            if not fresh:
                continue
            if vb_sz[j] > 0 and vb_px[j] > 0:
                if vb_px[j] > best_b:
                    best_b, best_bs = vb_px[j], vb_sz[j]
                elif vb_px[j] == best_b:
                    best_bs += vb_sz[j]
            if va_sz[j] > 0 and va_px[j] > 0:
                if va_px[j] < best_a:
                    best_a, best_as = va_px[j], va_sz[j]
                elif va_px[j] == best_a:
                    best_as += va_sz[j]
        if best_b > 0 and best_a < 1e17 and (best_b != last_b or best_bs != last_bs or best_a != last_a or best_as != last_as):
            out_t[k] = ts[i]
            out[k, 0], out[k, 1], out[k, 2], out[k, 3] = best_b, best_bs, best_a, best_as
            k += 1
            last_b, last_bs, last_a, last_as = best_b, best_bs, best_a, best_as
    return out_t[:k], out[:k]


def _read_zip(path: Path, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, header=None, names=names, dtype={"ex": str})


def available_days(symbol: str) -> list[str]:
    return sorted(p.name.split("_")[0] for p in (TICK_DIR / symbol.lower()).glob("*_quote.zip"))


def load_day(symbol: str, date: str, use_cache: bool = True, stale_ms: int | None = None) -> Day:
    """stale_ms=None picks 200ms for dense symbols (e.g. SPY) and 3s for sparsely quoted ones."""
    if stale_ms is None:
        stale_ms = 200 if symbol.lower() == "spy" else 3000
    cache = CACHE / f"{symbol.lower()}_{date}_{stale_ms}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return Day(symbol, date, **{k: z[k] for k in z.files})
    d = TICK_DIR / symbol.lower()
    q = _read_zip(d / f"{date}_quote.zip", ["t", "bpx", "bsz", "apx", "asz", "ex", "cond", "flag"])
    q = q[(q.t >= OPEN_MS - 60_000) & (q.t <= CLOSE_MS)]  # warm the book one minute before the open
    ex = q.ex.fillna("?").str.slice(0, 1).to_numpy().astype("S1").view("u1")
    t, out = _nbbo(q.t.to_numpy(np.int64), q.bpx.to_numpy(np.float64) / 1e4, q.bsz.to_numpy(np.float64),
                   q.apx.to_numpy(np.float64) / 1e4, q.asz.to_numpy(np.float64), ex.astype(np.int64), stale_ms)
    tr = _read_zip(d / f"{date}_trade.zip", ["t", "px", "sz", "ex", "cond", "flag"])
    tr = tr[(tr.t >= OPEN_MS) & (tr.t <= CLOSE_MS) & (tr.px > 0) & (tr.sz > 0)]
    day = Day(
        symbol, date, q_t=t, q_bid=out[:, 0], q_bsz=out[:, 1], q_ask=out[:, 2], q_asz=out[:, 3],
        t_t=tr.t.to_numpy(np.int64), t_px=tr.px.to_numpy(np.float64) / 1e4, t_sz=tr.sz.to_numpy(np.float64),
        t_ex=tr.ex.fillna("?").str.slice(0, 1).to_numpy().astype("S1").view("u1"),
    )
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{k: getattr(day, k) for k in day.__dataclass_fields__ if k not in ("symbol", "date")})
    return day


def regular_session(day: Day, start_min: float = 5.0, end_min: float = 5.0) -> slice:
    """Index slice of NBBO events inside the session, excluding the first/last minutes (opening/closing auction noise)."""
    lo = OPEN_MS + start_min * 60_000
    hi = CLOSE_MS - end_min * 60_000
    return slice(int(np.searchsorted(day.q_t, lo)), int(np.searchsorted(day.q_t, hi)))


def sign_trades(day: Day) -> np.ndarray:
    """Lee-Ready: +1 buyer-initiated if above the prevailing midpoint, -1 below, tick rule at the midpoint.

    "Prevailing" means the last quote strictly BEFORE the trade's millisecond: a trade that lifts the offer is stamped
    with the same ms as the quote update it causes, and using that post-trade quote would mis-sign it. The replay and
    live paths sign against the pre-trade quote too, which keeps offline and online features identical."""
    idx = np.searchsorted(day.q_t, day.t_t, side="left") - 1
    ok = idx >= 0
    mid = np.where(ok, 0.5 * (day.q_bid[np.maximum(idx, 0)] + day.q_ask[np.maximum(idx, 0)]), np.nan)
    s = np.sign(day.t_px - mid)
    prev = np.concatenate([[day.t_px[0]], day.t_px[:-1]])
    tick = np.sign(day.t_px - prev)
    last = 0.0
    out = np.zeros(len(s))
    for i in range(len(s)):  # tick-rule carry for zeros
        v = s[i] if s[i] != 0 and not np.isnan(s[i]) else (tick[i] if tick[i] != 0 else last)
        out[i] = v
        last = v
    return out
