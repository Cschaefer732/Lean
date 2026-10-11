"""Record live IBKR tick-by-tick data into the same `Day` structure the research stack consumes, so the study/replay
code can be re-run on current data (the shipped 2013 sample is a different microstructure regime)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import Day


class TickRecorder:
    def __init__(self, symbol: str, date: str):
        self.symbol, self.date = symbol, date
        self.q: list[tuple] = []
        self.t: list[tuple] = []

    def on_quote(self, q) -> None:
        self.q.append((q.t, q.bid, q.bsz, q.ask, q.asz))

    def on_trade(self, tr) -> None:
        self.t.append((tr.t, tr.px, tr.sz))

    def to_day(self) -> Day:
        q = np.array(self.q, float).reshape(-1, 5)
        t = np.array(self.t, float).reshape(-1, 3)
        return Day(self.symbol, self.date, q[:, 0].astype(np.int64), q[:, 1], q[:, 2], q[:, 3], q[:, 4],
                   t[:, 0].astype(np.int64), t[:, 1], t[:, 2], np.zeros(len(t), np.uint8))

    def save(self, directory: Path) -> Path:
        d = self.to_day()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.symbol.lower()}_{self.date}_rec.npz"
        np.savez_compressed(path, **{k: getattr(d, k) for k in d.__dataclass_fields__ if k not in ("symbol", "date")})
        return path


def load_recorded(path: Path, symbol: str, date: str) -> Day:
    z = np.load(path)
    return Day(symbol, date, **{k: z[k] for k in z.files})
