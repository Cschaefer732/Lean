"""Vectorised L1 microstructure features and forward-return targets on NBBO event time (research path)."""
from __future__ import annotations

import numpy as np

from .data import Day, regular_session, sign_trades

TICK = 0.01
HORIZONS_MS = {"100ms": 100, "500ms": 500, "1s": 1000, "5s": 5000, "30s": 30000}
FEATURES = ["imb", "spread", "depth", "d_imb_100ms", "flow_100ms", "flow_1s", "flow_10s", "ret_100ms", "ret_1s", "ret_10s", "bsz_chg_1s", "asz_chg_1s"]


def _at(t: np.ndarray, x: np.ndarray, when: np.ndarray) -> np.ndarray:
    """Value of the step series (t, x) at times `when` (last observation at or before)."""
    return x[np.maximum(np.searchsorted(t, when, side="right") - 1, 0)]


def build(day: Day, start_min: float = 5.0, end_min: float = 5.0):
    """Returns dict with event times, mid, features [n, F] and targets {horizon: [n]} (in ticks) for valid events."""
    s = regular_session(day, start_min, end_min)
    valid_all = day.q_bid < day.q_ask
    t_all, bid, ask = day.q_t[valid_all], day.q_bid[valid_all], day.q_ask[valid_all]
    bsz, asz = day.q_bsz[valid_all], day.q_asz[valid_all]
    mid = 0.5 * (bid + ask)

    sign = sign_trades(day)
    cum = np.concatenate([[0.0], np.cumsum(sign * day.t_sz)])

    def flow(t, w):
        hi = np.searchsorted(day.t_t, t, side="right")
        lo = np.searchsorted(day.t_t, t - w, side="right")
        return cum[hi] - cum[lo]

    lo_t = day.q_t[s.start] if s.start < len(day.q_t) else 0
    hi_t = day.q_t[min(s.stop, len(day.q_t)) - 1]
    sel = (t_all >= lo_t) & (t_all <= hi_t - 30_000)  # leave room for the longest horizon
    t = t_all[sel]
    imb = (bsz - asz) / (bsz + asz)
    feats = {
        "imb": imb[sel],
        "spread": ((ask - bid) / TICK)[sel],
        "depth": np.log(bsz + asz)[sel],
        "d_imb_100ms": imb[sel] - _at(t_all, imb, t - 100),
        "flow_100ms": flow(t, 100) / 1e3,
        "flow_1s": flow(t, 1000) / 1e3,
        "flow_10s": flow(t, 10_000) / 1e3,
        "ret_100ms": (mid[sel] - _at(t_all, mid, t - 100)) / TICK,
        "ret_1s": (mid[sel] - _at(t_all, mid, t - 1000)) / TICK,
        "ret_10s": (mid[sel] - _at(t_all, mid, t - 10_000)) / TICK,
        "bsz_chg_1s": (bsz[sel] - _at(t_all, bsz, t - 1000)) / 1e3,
        "asz_chg_1s": (asz[sel] - _at(t_all, asz, t - 1000)) / 1e3,
    }
    X = np.column_stack([feats[k] for k in FEATURES])
    y = {h: (_at(t_all, mid, t + ms) - mid[sel]) / TICK for h, ms in HORIZONS_MS.items()}
    return {"t": t, "mid": mid[sel], "spread_px": (ask - bid)[sel], "X": X, "y": y}


def on_grid(b: dict, step_ms: int = 100) -> dict:
    """Resample feature rows onto a uniform time grid (last event at/before each grid time).

    Event-time sampling over-weights bursts of quote flicker and inflates predictability; the grid is the honest basis
    for economics."""
    t = b["t"]
    g = np.arange(t[0], t[-1], step_ms)
    gi = np.searchsorted(t, g, side="right") - 1
    return {"t": g, "mid": b["mid"][gi], "spread_px": b["spread_px"][gi], "X": b["X"][gi], "y": {h: v[gi] for h, v in b["y"].items()}}
