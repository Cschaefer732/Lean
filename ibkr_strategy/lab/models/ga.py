"""Genetic algorithm that evolves the *rules* of a rotation strategy (not just two knobs), re-evolved at every
walk-forward retrain so it adapts to the regime of the most recent data.

Genome (11 genes in [0, 1]): 5 lookback weights, vol-normalisation exponent, top_k, absolute-momentum threshold,
inverse-vol weighting exponent, trend-filter SMA length, defensive-fill switch.
Fitness = Sharpe + 0.5 * Calmar on net-of-cost weekly returns over the *training* window only.
To limit selection bias the final book averages the weights of the top distinct genomes from several independent runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import MOM_LOOKBACKS, Market

N_GENES = 11
SMA_CHOICES = (0, 100, 150, 200)


def decode(g: np.ndarray) -> dict:
    lw = np.asarray(g[:5], dtype=float) + 1e-9
    return {
        "lb_w": lw / lw.sum(),
        "vol_p": float(g[5]),
        "top_k": 2 + int(round(g[6] * 6)),
        "theta": -0.02 + float(g[7]) * 0.07,
        "inv_q": float(g[8]) * 1.5,
        "sma": SMA_CHOICES[min(int(g[9] * 4), 3)],
        "defensive": bool(g[10] > 0.5),
    }


def genome_weights(mkt: Market, g: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Vectorised weights [len(rows), N] for a genome (sum <= 1; remainder is cash)."""
    p = decode(g)
    mom = mkt.mom[:, rows, :]                              # [L, R, N]
    raw = np.tensordot(p["lb_w"], mom, axes=(0, 0))        # [R, N]
    vol = np.maximum(mkt.vol[rows], 1e-4)
    rank_score = raw / vol ** p["vol_p"]
    elig = raw > p["theta"]
    if p["sma"]:
        elig &= mkt.above_sma[p["sma"]][rows]
    elig &= np.isfinite(raw)
    masked = np.where(elig, rank_score, -np.inf)
    ranks = np.argsort(np.argsort(-masked, axis=1), axis=1)
    sel = (ranks < p["top_k"]) & elig
    wt = np.where(sel, vol ** -p["inv_q"], 0.0)
    n_sel = sel.sum(axis=1)
    denom = wt.sum(axis=1, keepdims=True)
    W = np.where(denom > 0, wt / np.where(denom > 0, denom, 1.0), 0.0) * (n_sel / p["top_k"])[:, None]
    if p["defensive"]:
        idle = (p["top_k"] - n_sel) / p["top_k"]
        d = np.where(mkt.defensive_mask[None, :] & np.isfinite(raw), raw, -np.inf)
        best = np.argmax(d, axis=1)
        ok = d[np.arange(len(rows)), best] > p["theta"]
        W[np.arange(len(rows))[ok], best[ok]] += idle[ok]
    return W


def fast_returns(mkt: Market, W: np.ndarray, rows: np.ndarray, cost_bps: float = 5.0, smooth_weeks: int = 4) -> np.ndarray:
    """Net weekly returns of weights held from decision date to the next week's close."""
    if smooth_weeks > 1:
        c = np.cumsum(np.vstack([np.zeros((1, W.shape[1])), W]), axis=0)
        lo = np.maximum(np.arange(len(W)) - smooth_weeks + 1, 0)
        W = (c[np.arange(len(W)) + 1] - c[lo]) / (np.arange(len(W)) + 1 - lo)[:, None]
    r = mkt.r1[rows]
    gross = np.nansum(W * r, axis=1) + (1 - W.sum(axis=1)) * np.nan_to_num(mkt.r1c[rows])
    turn = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
    return gross - turn * cost_bps / 1e4


def fitness(weekly: np.ndarray) -> float:
    if len(weekly) < 20 or weekly.std() < 1e-9:
        return -10.0
    sharpe = weekly.mean() / weekly.std() * np.sqrt(52)
    eq = np.cumprod(1 + weekly)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    cagr = eq[-1] ** (52 / len(weekly)) - 1
    calmar = cagr / max(abs(mdd), 0.03)
    return float(sharpe + 0.5 * min(calmar, 3.0))


@dataclass
class GAConfig:
    pop: int = 100
    generations: int = 35
    runs: int = 3
    elite: int = 5
    tournament: int = 3
    crossover_p: float = 0.8
    mut_p: float = 0.2
    mut_sigma: float = 0.15
    top_final: int = 5


def evolve(mkt: Market, rows: np.ndarray, seed: int, cfg: GAConfig, history: list | None = None):
    rng = np.random.default_rng(seed)
    pop = rng.random((cfg.pop, N_GENES))

    def score(g):
        return fitness(fast_returns(mkt, genome_weights(mkt, g, rows), rows))

    fit = np.array([score(g) for g in pop])
    for gen in range(cfg.generations):
        if history is not None:
            history.append((gen, fit.max(), fit.mean()))
        order = np.argsort(-fit)
        new = [pop[i].copy() for i in order[: cfg.elite]]
        while len(new) < cfg.pop:
            a = pop[max(rng.integers(0, cfg.pop, cfg.tournament), key=lambda i: fit[i])]
            b = pop[max(rng.integers(0, cfg.pop, cfg.tournament), key=lambda i: fit[i])]
            if rng.random() < cfg.crossover_p:
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                span = hi - lo
                child = rng.uniform(lo - 0.3 * span, hi + 0.3 * span)
            else:
                child = a.copy()
            mut = rng.random(N_GENES) < cfg.mut_p
            child = child + mut * rng.normal(0, cfg.mut_sigma, N_GENES)
            new.append(np.clip(child, 0.0, 1.0))
        pop = np.array(new)
        fit = np.array([score(g) for g in pop])
    if history is not None:
        history.append((cfg.generations, fit.max(), fit.mean()))
    order = np.argsort(-fit)
    return pop[order], fit[order]


class GeneticStrategy:
    name = "ga"

    def __init__(self, cfg: GAConfig | None = None, seed: int = 0):
        self.cfg, self.seed = cfg or GAConfig(), seed

    def fit(self, mkt: Market, train_rows: np.ndarray) -> None:
        pool, fits = [], []
        for run in range(self.cfg.runs):
            pop, f = evolve(mkt, train_rows, self.seed * 1000 + run, self.cfg)
            pool.extend(pop[: self.cfg.top_final])
            fits.extend(f[: self.cfg.top_final])
        order = np.argsort(-np.array(fits))
        chosen, seen = [], set()
        for i in order:  # drop (near-)duplicate genomes so the average is over different rule sets
            key = tuple(np.round(pool[i], 1))
            if key not in seen:
                seen.add(key)
                chosen.append(pool[i])
            if len(chosen) >= self.cfg.top_final:
                break
        self.genomes = chosen
        self.train_fitness = float(np.mean([fits[i] for i in order[: len(chosen)]]))

    def weights(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        return np.mean([genome_weights(mkt, g, rows) for g in self.genomes], axis=0)
