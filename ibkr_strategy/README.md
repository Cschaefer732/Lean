# IBKR Momentum Rotation

A rules-based, long-only, multi-asset ETF strategy for Interactive Brokers: weekly-smoothed momentum rotation with
absolute-momentum crash protection. The strategy logic is one small, stateless Python package
(`momentum_rotation/`) that is shared by three front-ends, so the research, the backtest and the live trader cannot
drift apart:

| Front-end | File | Purpose |
|---|---|---|
| Research harness | `research/` | Vectorised backtests, parameter sweeps, robustness checks |
| LEAN algorithm | `lean/IBKRMomentumRotation.py` | Event-driven backtest with the IB fee/margin model; deployable with LEAN's IB brokerage (`live-interactive`) |
| IBKR API runner | `live/trader.py` | Direct TWS / IB Gateway API trading via `ib_async`, paper-first |

> **Not financial advice. Not verified against a live IBKR session.** This environment had no IBKR account, Gateway
> or credentials. The IBKR adapter is tested against a faked `ib_async.IB`, and every order-generation path is
> unit-tested, but the first run against a real paper account is the real integration test. Do that before anything else.

## The strategy

Universe (13 liquid US ETFs): `SPY QQQ IWM EFA EEM VNQ TLT IEF LQD TIP GLD DBC` + `SHY` as cash.

1. **Score** every risk asset by the mean of its 1/3/6/12-month total returns.
2. **Select** the 5 highest scores that are positive (absolute momentum: no positive trend, no position).
3. **Weight** picks by inverse 63-day volatility; each pick owns at most 1/5 of the book.
4. **Idle slots** (fewer than 5 positive assets) go to the best positive-momentum asset of `IEF/TLT/GLD`, otherwise
   to `SHY`. In a broad sell-off the book moves to bonds, gold and cash by itself.
5. **Smooth**: the target is the average of the signal on the decision day and on the last trading day of the three
   previous weeks. Rebalance weekly (last session of the week, ~15:45 ET, market-on-close).

No leverage by default, no shorting, no options, about 7x annual turnover.

## Research summary

Data: Yahoo Finance daily total-return series, 2006-02 to 2026-10-09 (limited by DBC's inception); 5 bps cost per unit
of turnover. Design period 2007-2015, confirmation period 2016-2026, plus the full period. Reproduce with
`research/fetch_data.py` then `research/run_*.py`.

| 2007-01 to 2026-10 | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| **Strategy** (trade at the decision close, MOC) | 9.4% | 11.5% | 0.67 | -17.5% | 0.54 |
| **Strategy** (trade one day later) | 9.5% | 11.6% | 0.68 | -20.6% | 0.46 |
| SPY buy & hold | 11.0% | 19.5% | 0.54 | -55.2% | 0.20 |
| 60/40 SPY/IEF | 8.1% | 11.0% | 0.60 | -32.6% | 0.25 |

By sub-period the strategy (same-close) returned Sharpe 0.52 / CAGR 7.8% / MaxDD -14.4% in 2007-2015 and
0.82 / 10.7% / -17.5% in 2016-2026. SPY returned 0.29 / 6.3% / -55.2% and 0.79 / 15.1% / -33.7%. Calendar year 2008:
strategy +15.7%, SPY -36.8%. 2022 (stocks and bonds fell together): strategy -14.9%, SPY -18.2%, 60/40 -16.7%.

**What this does and does not show**

- The robust result is **drawdown control**: 17.5% max drawdown vs SPY's 55.2%. Across the 210-config grid, 91% of
  configs kept max drawdown under 30% in both periods (the worst, a 2-asset weekly variant, hit 41%).
  **It will lag SPY in strong equity bull markets** (it did, 2016-2026: 10.7% vs 15.1%).
- The **Sharpe advantage is a point estimate, not proof**. A 21-day block bootstrap gives the Sharpe difference vs
  SPY a 90% interval of [-0.16, +0.46] (P(>0) about 0.79). Do not expect a reliably higher Sharpe than SPY.
- **Parameters were chosen from a plateau, not an optimum.** In a 210-config grid, in-sample and out-of-sample Sharpe
  correlated only 0.40, so the in-sample winner is mostly noise. `top_k` 4-6 and any multi-horizon lookback blend were
  statistically indistinguishable; the structure matters more than the numbers.
- **Rebalance-day timing luck is large**: moving a monthly rebalance by a few days alone moved Sharpe from 0.51 to
  0.75. That is why the strategy averages four weekly signals instead of trading one monthly snapshot.
- Tried and rejected: weekly un-smoothed rotation (turnover 16x, worse), rank hysteresis (worse), 5 extra tickers
  (worse out-of-sample), a Faber SMA blend (same Sharpe, lower return), vol-targeting/leverage (no Sharpe gain; return
  rises only by taking proportionally more drawdown, `--leverage` exists but defaults to 1.0).
- Robust to costs: for the monthly base configuration Sharpe fell from 0.75 to 0.64 at 25 bps per unit of turnover.
  Robust to execution delay: final strategy Sharpe 0.67 / 0.68 / 0.72 for same-close / +1 day / +2 days.
- Survivorship caveat: the ETFs were chosen with hindsight knowing they are the surviving, liquid core asset classes.

**Correction made during the work:** an early version of the research engine let `lag=0` earn the decision day's own
return (look-ahead). The engine now rejects `lag < 1`. All numbers above come from the corrected engine.

## LEAN cross-check (independent engine)

`lean/run_backtest.sh` runs `IBKRMomentumRotation` on the same data through LEAN with
`BrokerageName.INTERACTIVE_BROKERS_BROKERAGE` (IB tiered fees, margin account), $100k start, 2007-02 to 2026-10:
**9.75% CAGR, 18.1% max drawdown, 3,595 orders, $4.4k total fees**. Against the research engine replaying LEAN's exact
timing: target weights identical on 96% of weeks, weekly-return correlation 0.94, volatility 11.5% vs 11.4%, CAGR
within 0.35%/yr (fees and idle cash). Two real bugs were found and fixed this way (single-row factor files made LEAN
silently drop GLD; float dust weights). LEAN's own Sharpe (0.51) uses its risk-free curve and is not comparable to the
tables above. Note: LEAN's daily-resolution scheduler fills at the previous close, so the backtest decides and fills on
Thursday's close; the live runner decides on the day's own prices around 15:30 and uses MOC orders.

LEAN's bundled data only covers a few of these ETFs, so `lean/build_lean_data.py` builds a gitignored overlay folder
from the Yahoo CSVs (the repo's `Data/` is not modified).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest tests -q                                   # 56 tests, no network or IBKR needed

# research (needs network once)
python research/fetch_data.py && python research/run_final_eval.py

# LEAN backtest (needs .NET 9 SDK, `dotnet build Launcher -c Release`, Python 3.11 + PYTHONNET_PYDLL)
lean/run_backtest.sh
```

### Trading on IBKR through the API

1. Run IB Gateway (or TWS) logged into a **paper** account; enable *Settings > API > Enable ActiveX and Socket
   Clients*; leave *Read-Only API* off only when you want to place orders. Ports: Gateway paper 4002, TWS paper 7497,
   Gateway live 4001, TWS live 7496. A margin account is recommended (MOC buys are placed alongside sells).
2. Dry run (read-only connection, prints the plan, sends nothing):
   `python -m live.trader --port 4002`
3. Paper trade: `python -m live.trader --port 4002 --execute`
4. Schedule daily at 15:30 America/New_York, for example
   `30 15 * * 1-5  cd /path/to/ibkr_strategy && .venv/bin/python -m live.trader --port 4002 --execute >> state/trader.log 2>&1`
   (use the container's timezone accordingly). The script exits quietly unless today is the final trading session of
   the week, determined from IBKR's own trading calendar, so holidays are handled.

Safeguards built into `live/trader.py`: read-only API connection unless `--execute`; non-paper accounts refused
without `--live`; `state/HALT` file kill switch; once-per-day idempotence; MOC cutoff check (15:50 ET); stale-data,
missing-history and >25% one-day-move checks; max-turnover cap (`--max-turnover`, default 1.25x equity); the strategy
only ever touches its own 13 symbols, and `--capital` limits the equity it manages in a shared account. Every decision
is written to `state/decision-YYYY-MM-DD.json`.

Known IBKR-side assumptions to confirm on first paper run: historical bars use `ADJUSTED_LAST` (IBKR's split- and
dividend-adjusted series; use `--what-to-show TRADES` for split-only), you need market-data entitlements for US ETFs
(or pass `--delayed-data`), and contracts are resolved via `reqContractDetails` preferring ARCA/NASDAQ listings.

### Live trading with LEAN instead

The LEAN algorithm is ready for `live-interactive`. The IB brokerage plugin is a separate repository
(`QuantConnect/Lean.Brokerages.InteractiveBrokers`) that is not part of this checkout; install it via the LEAN CLI
(`lean live deploy --brokerage "Interactive Brokers"`) or place its build in `Launcher/Plugins`, and set the `ib-*`
keys in `Launcher/config.json`.

## Strategy lab (ML / GA / RL)

`lab/` is a leak-free walk-forward tester and builder for random forests, gradient boosting, MLP/GRU deep nets, a genetic
algorithm, a policy-gradient RL agent and ensembles, with deflated-Sharpe and bootstrap significance tests. Headline
finding: unconstrained ML did **not** beat the hand-built rotation out-of-sample; ML only helped as a constrained ranker.
See [`lab/README.md`](lab/README.md) for results and how to add your own strategy. `python -m lab.run`.

## HFT / market-microstructure system

`hft/` is a market-making research bench and execution skeleton built on **real SPY/BAC/AIG/IBM tick data** (the AlgoSeek
sample shipped with LEAN): NBBO consolidation, vectorised and online L1 features, an event-driven replay venue with
latency, queue-position and cancel/fill races, fee models, a pre-trade risk gate with kill switch, and an IBKR adapter
(dry-run by default, paper-port guard, tick recorder, latency stats). The same `Strategy` objects run in replay and live.

Honest headline: the only viable niche is passive market making in SPY-like names, protected by an order-book-imbalance
alpha. The alpha is real out of sample but small, the edge lives at single-digit **milliseconds** and is gone by about
**50 ms** one-way latency, and at **IBKR Pro tiered fees it loses money** (+$1,050 vs -$3,306 on the two test days). IBKR's
API path is not an HFT venue. Taking liquidity is unprofitable by a wide margin. Full design, results, caveats and the
roadmap are in [`hft/docs/DESIGN.md`](hft/docs/DESIGN.md); raw outputs are in `hft/reports/`.

```bash
python -m hft.research.signal_study spy bac aig ibm   # do L1 features predict the next move? (OOS by day)
python -m hft.research.mm_study                       # latency / fee / queue sweeps with shuffled-signal control (~30 min)
python -m hft.research.bench                          # replay throughput and tick-to-decision time
python -m hft.run_live --symbol SPY --record          # record your own ticks (never sends orders)
python -m hft.run_live --symbol SPY --seconds 600     # dry-run against IB Gateway
```

## Layout

```
momentum_rotation/   config, stateless signals, order sizing (pure functions, fully tested)
research/            data download, engine, sweeps, final evaluation, LEAN cross-check
lean/                LEAN algorithm, data builder, backtest runner
live/                IBKR broker adapter and weekly rebalancer
lab/                 walk-forward ML/GA/RL strategy tester and builder (see lab/README.md)
hft/                 tick-level market-making bench: NBBO, replay venue, risk, IBKR adapter (see hft/docs/DESIGN.md)
tests/               signal parity, no-lookahead, order planning, safety guards, IBKR adapter, lab leak detector, HFT fill/latency/risk
```
