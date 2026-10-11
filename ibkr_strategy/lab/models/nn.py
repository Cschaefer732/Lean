"""Deep-learning regressors (PyTorch): an MLP on engineered features and a GRU on raw volatility-scaled return
sequences. Regularised heavily (dropout, weight decay, early stopping on the most recent 20% of the training window,
seed ensembling) because the sample is small and noisy."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data import Market, rank_target
from .base import gated_weights

torch.set_num_threads(4)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x, seq=None):
        return self.net(x).squeeze(-1)


class SeqNet(nn.Module):
    def __init__(self, n_static: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x, seq):
        _, h = self.gru(seq.unsqueeze(-1))
        return self.head(torch.cat([h[-1], self.static(x)], dim=-1)).squeeze(-1)


class TorchStrategy:
    top_k = 5

    def __init__(self, kind: str = "mlp", seeds: int = 3, epochs: int = 40, patience: int = 6, lr: float = 1e-3,
                 weight_decay: float = 1e-2, seed: int = 0, hybrid: bool = False):
        self.kind, self.seeds, self.epochs, self.patience = kind, seeds, epochs, patience
        self.lr, self.wd, self.seed = lr, weight_decay, seed
        self.hybrid = hybrid
        self.name = kind + ("_h" if hybrid else "")

    def _tensors(self, mkt: Market, rows: np.ndarray):
        X = np.clip((mkt.X[rows].reshape(-1, mkt.X.shape[-1]) - self.mu) / self.sd, -5, 5)
        S = mkt.seq[rows].reshape(-1, mkt.seq.shape[-1])
        S = np.clip(S / self.seq_sd, -5, 5)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(S, dtype=torch.float32)

    def fit(self, mkt: Market, train_rows: np.ndarray) -> None:
        flat = mkt.X[train_rows].reshape(-1, mkt.X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        self.seq_sd = float(mkt.seq[train_rows].std()) + 1e-8
        y = (rank_target(mkt.y[train_rows]) if self.hybrid else mkt.y[train_rows]).reshape(-1)
        ok = np.isfinite(y)
        lo, hi = np.nanpercentile(y[ok], [1, 99])
        self.y_sd = float(np.clip(y[ok], lo, hi).std())
        X, S = self._tensors(mkt, train_rows)
        Y = torch.tensor(np.clip(np.nan_to_num(y, nan=0.0), lo, hi) / self.y_sd, dtype=torch.float32)
        keep = torch.tensor(ok)
        n_rows = len(train_rows)
        row_of = torch.arange(len(Y)) // mkt.N
        cut = int(n_rows * 0.8)
        tr, va = keep & (row_of < cut), keep & (row_of >= cut)
        self.models = []
        for s in range(self.seeds):
            torch.manual_seed(self.seed * 100 + s)
            net = MLP(X.shape[1]) if self.kind == "mlp" else SeqNet(X.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            best, best_state, bad = np.inf, None, 0
            idx = torch.nonzero(tr).squeeze(1)
            for ep in range(self.epochs):
                net.train()
                perm = idx[torch.randperm(len(idx))]
                for b in range(0, len(perm), 256):
                    j = perm[b : b + 256]
                    loss = nn.functional.huber_loss(net(X[j], S[j]), Y[j])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = nn.functional.huber_loss(net(X[va], S[va]), Y[va]).item()
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            self.models.append(net.eval())

    @torch.no_grad()
    def scores(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        X, S = self._tensors(mkt, rows)
        pred = np.mean([m(X, S).numpy() for m in self.models], axis=0) * self.y_sd
        return pred.reshape(len(rows), mkt.N)

    def weights(self, mkt: Market, rows: np.ndarray) -> np.ndarray:
        S = self.scores(mkt, rows)
        self.last_scores = S
        return gated_weights(S, mkt, rows, self.top_k, self.hybrid, self.hybrid)
