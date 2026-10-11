"""Policy-gradient (REINFORCE with baseline) allocation agent.

Environment: weekly steps over the training window. State: per-asset standardised features plus market features.
Policy: a shared scoring network maps each asset's features to a logit; a separate head produces the cash (SHY) logit
from market features; weights = softmax over [assets, cash]. Exploration: Gaussian noise on the logits.
Reward: net portfolio return after turnover cost minus a variance penalty (mean-variance utility), so the agent
is paid for risk-adjusted rather than raw return. Training samples random 26-week episodes from the training
window only; deployment is deterministic (softmax of the mean logits). Several seeds are averaged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..data import Market


@dataclass
class RLConfig:
    iterations: int = 300
    batch: int = 256
    episode: int = 26
    gamma: float = 0.9
    risk_aversion: float = 8.0
    cost_bps: float = 5.0
    sigma_start: float = 0.8
    sigma_end: float = 0.25
    lr: float = 5e-3
    hidden: int = 32
    seeds: int = 3


class PolicyNet(nn.Module):
    def __init__(self, n_feat: int, n_market: int, hidden: int):
        super().__init__()
        self.asset = nn.Sequential(nn.Linear(n_feat, hidden), nn.Tanh(), nn.Linear(hidden, hidden // 2), nn.Tanh(),
                                   nn.Linear(hidden // 2, 1))
        self.cash = nn.Linear(n_market, 1)
        nn.init.zeros_(self.cash.weight)
        nn.init.zeros_(self.cash.bias)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x [..., N, F], m [..., M] -> logits [..., N+1] (last = cash)."""
        a = self.asset(x).squeeze(-1)
        c = self.cash(m)
        return torch.cat([a, c], dim=-1)


MARKET_COLS = ("mkt_vol63", "mkt_vol_ratio", "mkt_ret252", "mkt_dd", "mkt_bond_vs_stock")


class RLStrategy:
    name = "rl"

    def __init__(self, cfg: RLConfig | None = None, seed: int = 0):
        self.cfg, self.seed = cfg or RLConfig(), seed

    def _prep(self, mkt: Market, rows: np.ndarray):
        X = (mkt.X[rows] - self.mu) / self.sd
        X = np.clip(X, -5, 5)
        m = X[:, 0, [mkt.feat_names.index(c) for c in MARKET_COLS]]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)

    def fit(self, mkt: Market, train_rows: np.ndarray) -> None:
        cfg = self.cfg
        rows = train_rows[np.isfinite(mkt.r1[train_rows]).all(axis=1) & np.isfinite(mkt.r1c[train_rows])]
        flat = mkt.X[rows].reshape(-1, mkt.X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        X, M = self._prep(mkt, rows)
        R = torch.tensor(np.concatenate([mkt.r1[rows], mkt.r1c[rows, None]], axis=1), dtype=torch.float32)
        n = len(rows)
        L = min(cfg.episode, n - 2)
        self.nets = []
        for s in range(cfg.seeds):
            torch.manual_seed(self.seed * 100 + s)
            gen = torch.Generator().manual_seed(self.seed * 100 + s)
            net = PolicyNet(X.shape[-1], M.shape[-1], cfg.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
            disc = cfg.gamma ** torch.arange(L, dtype=torch.float32)
            for it in range(cfg.iterations):
                sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * it / max(cfg.iterations - 1, 1)
                starts = torch.randint(0, n - L, (cfg.batch,), generator=gen)
                idx = starts[:, None] + torch.arange(L)[None, :]               # [B, L]
                mu_logits = net(X[idx], M[idx])                                 # [B, L, N+1]
                noise = torch.randn(mu_logits.shape, generator=gen) * sigma
                action = (mu_logits + noise).detach()
                w = torch.softmax(action, dim=-1)
                w_prev = torch.cat([torch.zeros_like(w[:, :1]), w[:, :-1]], dim=1)
                turn = (w - w_prev).abs().sum(-1)
                port = (w * R[idx]).sum(-1) - turn * cfg.cost_bps / 1e4
                reward = port - 0.5 * cfg.risk_aversion * port**2             # [B, L]
                # reward-to-go with discounting
                togo = torch.flip(torch.cumsum(torch.flip(reward * disc, [1]), 1), [1]) / disc
                adv = togo - togo.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-8)
                logp = -0.5 * (((action - mu_logits) / sigma) ** 2).sum(-1)
                loss = -(logp * adv.detach()).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            self.nets.append(net.eval())

    @torch.no_grad()
    def weights(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        X, M = self._prep(mkt, rows)
        ws = [torch.softmax(net(X, M), dim=-1).numpy() for net in self.nets]
        return np.mean(ws, axis=0)[:, :-1]
