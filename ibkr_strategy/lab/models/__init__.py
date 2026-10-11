"""Strategy registry and ensemble builder.

A strategy is any object with `fit(mkt, train_rows)` and `weights(mkt, rows)` (see lab.walkforward.Strategy).
Register your own with `@register("name")` on a zero-argument factory, then use it by name from the CLI;
combine any registered strategies with the spec `ens:rf+gbm+ga`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

REGISTRY: dict[str, Callable[[], object]] = {}


def register(name: str):
    def deco(factory):
        REGISTRY[name] = factory
        return factory

    return deco


class Ensemble:
    """Equal-weight average of the members' target weights (works for scorers, GA and RL alike)."""

    def __init__(self, members: list, name: str):
        self.members, self.name = members, name

    def fit(self, mkt, train_rows):
        for m in self.members:
            m.fit(mkt, train_rows)

    def weights(self, mkt, rows):
        return np.mean([m.weights(mkt, rows) for m in self.members], axis=0)


def get_factory(spec: str) -> Callable[[], object]:
    _load_builtin()
    if spec.startswith("ens:"):
        parts = spec[4:].split("+")
        facs = [get_factory(p) for p in parts]
        return lambda: Ensemble([f() for f in facs], spec)
    if spec not in REGISTRY:
        raise KeyError(f"unknown strategy '{spec}'. available: {sorted(REGISTRY)}")
    return REGISTRY[spec]


def _load_builtin():
    if REGISTRY:
        return
    from .ga import GeneticStrategy
    from .nn import TorchStrategy
    from .rl import RLStrategy
    from .trees import ExtraTrees, GradientBoosting, GradientBoostingHybrid, RandomForest, RandomForestHybrid, RidgeLinear

    REGISTRY.update(
        {
            "rf": RandomForest,
            "extra_trees": ExtraTrees,
            "gbm": GradientBoosting,
            "ridge": RidgeLinear,
            "ga": GeneticStrategy,
            "rl": RLStrategy,
            "rf_h": RandomForestHybrid,
            "gbm_h": GradientBoostingHybrid,
            "mlp_h": lambda: TorchStrategy("mlp", seeds=3, hybrid=True),
            "mlp": lambda: TorchStrategy("mlp", seeds=3),
            "gru": lambda: TorchStrategy("gru", seeds=2),
        }
    )
