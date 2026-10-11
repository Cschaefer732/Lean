# Strategy lab

A leak-free tester and builder for ML / GA / RL allocation strategies on the IBKR ETF universe. Every strategy,
whatever its algorithm, is run through the *same* pipeline so results are comparable:

1. **Walk-forward**: expanding training window, retrained every 52 weeks (configurable), with the label window purged
   (a training row is only used once its 4-week forward return has fully elapsed). First model trains on 5 years.
2. **Same portfolio construction** for score-based models (top 5, inverse-vol, idle slots to defensives/cash) and the
   same 4-week weight smoothing for every candidate.
3. **Same execution/cost engine** as the production research: daily drift, 5 bps per unit turnover, trade at the decision close.
4. **Honest statistics**: information coefficient (with overlap-adjusted t-stat), paired block-bootstrap probability of beating the
   production rotation and SPY, **deflated Sharpe ratio** (penalises best-of-N selection), and a **null distribution** of
   unfitted random rotation rule sets.

```bash
python -m lab.run                         # everything, ~6 min on 4 cores (add --cache to reuse fits)
python -m lab.run --list
python -m lab.run --strategies rf_h,gbm_h,ga,ens:rf_h+gbm_h+ga --oos-start 2012-01-01
```

Outputs `lab/reports/leaderboard.{md,csv}` and `equity.png`.

## Strategies

| Name | Family | What it does |
|---|---|---|
| `rf`, `extra_trees` | Random forest / extra-trees | Shallow, heavily regularised; predict 4-week excess return per asset |
| `gbm` | Gradient boosting (LightGBM) | 7 leaves, 200 rounds, subsampling, L2 |
| `ridge` | Linear baseline | Tells you whether non-linearity helps at all |
| `mlp`, `gru` | Deep learning (PyTorch) | MLP on 22 features; GRU on 64 days of vol-scaled returns; dropout, weight decay, early stopping, seed ensembles |
| `ga` | Genetic algorithm | Evolves the rules (lookback blend, top-k, thresholds, trend filter, weighting, defensive fill); re-evolved every retrain; final book averages the top distinct genomes of 3 independent runs |
| `rl` | Policy-gradient RL (REINFORCE + baseline) | Shared asset-scoring network + cash head, Gaussian exploration, mean-variance reward net of costs, 3 seeds |
| `rf_h`, `gbm_h`, `mlp_h` | Hybrids | Train on *cross-sectional rank* targets and keep the production absolute-momentum rule as a risk-off gate: ML only chooses *which* assets, the rule decides *whether* to be invested |
| `ens:a+b+c` | Ensemble | Equal-weight average of the members' target weights |

Features (22): 1/5/21/63/126/252-day returns, 21/63-day vol and ratio, drawdown from 252d high, distance to 50/200d SMA,
RSI, cross-sectional momentum rank/z-score, five market-regime features (SPY vol, vol ratio, 12m return, drawdown;
bond-vs-stock momentum) and asset-class flags. All are computed strictly from data up to the decision close
(tested by mutating future prices).

## Results (out-of-sample 2012-01 to 2026-10, 771 weeks, 15 retrains)

| Strategy | CAGR | Vol | Sharpe | MaxDD | IC (t) | DSR | P(Sharpe > production) | P(> SPY) |
|---|---|---|---|---|---|---|---|---|
| *Production rotation (hand-built)* | 9.4% | 10.7% | 0.78 | -17.5% | | | | |
| *SPY buy & hold* | 15.0% | 16.5% | 0.85 | -33.7% | | | | |
| *60/40* | 9.6% | 9.8% | 0.86 | -21.1% | | | | |
| rf / extra_trees / gbm / ridge | 4-7% | 11-13% | 0.29-0.46 | -22 to -31% | 0.01-0.02 (<1) | 0.5-0.76 | 0.01-0.04 | 0.00 |
| mlp / gru | 7-8% | 13-14% | 0.49-0.51 | -28 to -29% | 0.04-0.05 (1.4-1.6) | 0.80 | 0.04-0.05 | 0.00 |
| ga | 5.8% | 7.2% | 0.64 | **-11.9%** | | 0.92 | 0.11 | 0.21 |
| rl | 3.5% | 7.4% | 0.32 | -21.3% | | 0.60 | 0.01 | 0.00 |
| rf_h | 8.9% | 11.2% | 0.70 | -26.4% | 0.09 (2.8) | 0.95 | 0.21 | 0.21 |
| gbm_h | 8.9% | 10.9% | 0.72 | -24.9% | 0.08 (2.4) | 0.95 | 0.23 | 0.24 |
| mlp_h | 9.8% | 11.7% | 0.75 | -25.6% | **0.11 (3.3)** | 0.96 | 0.31 | 0.26 |
| ens:rf_h+gbm_h+mlp_h | 9.3% | 11.2% | 0.73 | -25.7% | | 0.96 | 0.28 | 0.26 |

### What the lab found (read this before trusting any row)

- **Plain ML does not work here.** Unconstrained tree, boosting, ridge, MLP, GRU and RL models all *underperformed* even
  equal-weight and bought nothing statistically distinguishable from noise (IC t-stats < 2). The data is ~20 years x 12
  assets; a flexible model has far too little signal to learn from. Plain momentum alone has IC 0.068 (t 2.1) on this
  panel, better than every plain ML model.
- **ML added value only when constrained by prior knowledge.** Keeping the momentum risk-off rule and training on
  rank targets raised IC to 0.08-0.11 and lifted Sharpe from ~0.4 to ~0.7. That is the useful, honest finding: ML is a
  better *ranker* than the simple 1-3-6-12 average, but it still did not beat the hand-built rule as a *portfolio*
  (P(beat) 0.2-0.3, deeper drawdowns).
- **Nothing beat SPY in this sample's Sharpe** (a strong 2012-2026 bull market); the lower-drawdown strategies
  are the GA (-11.9%) and the production rotation.
- **Selection caveats.** The hybrids were designed *after* seeing round-1 results (declared as extra trials; the DSR
  counts 13 strategies, not my design iterations). The production rotation was designed with knowledge of this period, so
  it is an optimistic baseline that the ML models are *not* given. Treat `mlp_h`'s IC as promising, not proven.
- GA and RL training-window fitness was far higher than out-of-sample results (in a single-block check the RL agent's training-window Sharpe was ~1.9 vs 0.3 out-of-sample over the full walk-forward): the
  textbook overfitting signature, and why every result above is walk-forward.
- **Leak detector**: `tests/test_lab.py` runs the models on driftless random-walk prices and requires |IC| < 0.06;
  a leaking harness would show a large IC.

## Building your own

```python
import numpy as np
from lab.models import register

@register("my_strategy")
class MyStrategy:
    name = "my_strategy"
    def fit(self, mkt, train_rows):          # train_rows: week indices whose labels are fully observed
        ...
    def weights(self, mkt, rows):            # -> [len(rows), mkt.N] long-only weights, sum <= 1 (rest = cash)
        ...
# python -m lab.run --strategies my_strategy,ens:my_strategy+mlp_h
```

`mkt` (`lab.data.Market`) exposes features `X[t, asset, feature]`, forward excess returns `y`, next-week returns
`r1`, volatility, momentum, trend flags and raw return sequences, all indexed by weekly decision date.
Anything registered gets the same walk-forward, costs, significance tests and report. Note the multiple-testing
warning: every extra strategy you try raises the bar the deflated Sharpe applies.

## Not wired to live trading

None of these models is connected to `live/trader.py`. By the lab's own evidence, none has earned a place in front of
the production rotation yet; `mlp_h`/`ens:rf_h+gbm_h+mlp_h` are the candidates to re-test on genuinely new data
(for example by running the lab again in six months with `--oos-start` unchanged and checking whether IC holds).
