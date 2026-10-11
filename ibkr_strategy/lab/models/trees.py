"""Classical supervised learners. Hyper-parameters are deliberately conservative (shallow, heavily regularised):
financial signal-to-noise is tiny and ~3-12k correlated samples punish flexible models."""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import SupervisedStrategy


class RandomForest(SupervisedStrategy):
    name = "rf"

    def make_model(self):
        return RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=80, max_features=0.5,
                                     n_jobs=-1, random_state=0)


class ExtraTrees(SupervisedStrategy):
    name = "extra_trees"

    def make_model(self):
        return ExtraTreesRegressor(n_estimators=300, max_depth=6, min_samples_leaf=80, max_features=0.5,
                                   n_jobs=-1, random_state=0)


class GradientBoosting(SupervisedStrategy):
    name = "gbm"

    def make_model(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100,
                             subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                             random_state=0, n_jobs=4, verbose=-1)


class RidgeLinear(SupervisedStrategy):
    name = "ridge"

    def make_model(self):
        return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))


class RandomForestHybrid(RandomForest):
    name = "rf_h"
    use_rank, gate = True, True


class GradientBoostingHybrid(GradientBoosting):
    name = "gbm_h"
    use_rank, gate = True, True
