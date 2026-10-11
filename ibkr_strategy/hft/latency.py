"""Latency instrumentation: record stage timings in microseconds and report tail percentiles (the tail is what matters)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class LatencyStats:
    def __init__(self):
        self._d: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, us: float) -> None:
        self._d[stage].append(us)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self._d.items():
            a = np.asarray(v)
            out[k] = {"n": len(a), "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)),
                      "p999": float(np.percentile(a, 99.9)), "max": float(a.max())}
        return out

    def report(self) -> str:
        rows = ["stage                n      p50us     p99us    p99.9us       maxus"]
        for k, s in sorted(self.summary().items()):
            rows.append(f"{k:18s} {s['n']:6d} {s['p50']:10.1f} {s['p99']:9.1f} {s['p999']:10.1f} {s['max']:11.1f}")
        return "\n".join(rows)
