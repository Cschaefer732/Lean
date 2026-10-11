# HFT system design

This document is the design for a high-frequency market-making stack, what was built and measured here, and a blunt
account of what an IBKR deployment can and cannot do. Numbers marked **(measured)** come from `hft/reports/` and can be
regenerated; numbers marked **(typical)** are industry orders of magnitude I did not measure; items marked **(verify)**
are external facts about IBKR/regulation that must be checked against current documentation before relying on them.

## 1. Bottom line

* The only structurally viable retail-accessible niche is **passive market making in a tight-spread, large-tick name
  (SPY-like)**, protected from adverse selection by a short-horizon order-book alpha. Taking liquidity is dead on arrival.
* On real SPY ticks the alpha is real but small, the edge lives at **single-digit milliseconds**, and it is **gone by about
  50 ms one-way latency**. IBKR's API path is realistically in that regime or worse (10s-100s of ms), so a profitable
  deployment *through the IBKR API* is not supported by this evidence.
* At **IBKR Pro tiered fees the strategy loses money** even at 1 ms latency; it only breaks even with rebate-tier pricing
  that is available to exchange-connected firms, not to a retail account.
* Two levers tried to rescue it under IBKR constraints both failed: a rebate-free fee schedule is the binding problem
  (section 4.2), and a wider-spread name does not help because adverse selection grows with the spread (section 4.2b).
* The queue-position model decides whether the edge exists at all (results range from clearly positive to zero depending
  on assumptions), so the replay result is a hypothesis to be tested with small live size, not a forecast.

What the stack is good for: a leak-checked research bench and a production-shaped execution skeleton you can point at
better data or a better venue.

## 2. Reality check: three latency tiers

| Tier | Typical tick-to-trade | What it takes | Reachable via IBKR API? |
|---|---|---|---|
| A. True HFT | 0.1-10 us (FPGA) / 1-50 us (kernel-bypass C++) **(typical)** | Exchange colocation, direct binary feeds (ITCH/PITCH/MDP), market-participant membership or DMA, FPGA/NIC timestamps | No |
| B. Low-latency | 100 us - 10 ms **(typical)** | Cloud/colo VM near the matching engine, FIX/OUCH via a DMA broker, consolidated or direct feed | No |
| C. IBKR API | tens to hundreds of ms **(typical, verify by recording)** | TWS/Gateway (Java) -> TCP socket -> `ib_async`, SMART routing, retail fee schedule | Yes: this is the only tier IBKR serves |

IBKR constraints that shape the design **(verify against current docs)**: roughly 50 API messages per second per
client; tick-by-tick subscriptions are a capped, scarce resource; no post-only / add-liquidity-only order type on the
standard API (emulated locally and therefore racy); no queue-position or full-depth data on SMART for NBBO strategies;
message-to-trade-ratio fees/policies; paper-account fills are simulated by IBKR so **paper trading cannot calibrate a
queue model**.

## 3. Architecture

```
            +--------------------+     +-------------------+
 venue /    |  Market data in    |     |   Replay venue    |   <- same Strategy objects
 recorded   |  (ticks -> Quote / | --> |   (research)      |      run on both
 ticks      |   Trade events)    |     +-------------------+
            +---------+----------+     +-------------------+
                      |                |   IBKR adapter    |
                      v                |   (live / dry-run)|
            +--------------------+     +---------+---------+
            | OnlineFeatures     |               ^
            |  imbalance, signed |               |  place / cancel
            |  flow (100ms, 1s)  |     +---------+---------+
            +---------+----------+     |  RiskGate         |
                      v                |  qty, notional,   |
            +--------------------+     |  position, band,  |
            | LinearAlpha        |     |  msg-rate, kill   |
            |  predicted move    |     +---------+---------+
            +---------+----------+               ^
                      v                          |
            +--------------------+               |
            | PassiveMM          |---------------+
            |  quote / pull /    |
            |  inventory control |
            +--------------------+
```

| Component | File | Notes |
|---|---|---|
| Event types, `Strategy`/`Context` contract | `hft/core.py` | Strategies see only `Context`; nothing venue-specific leaks in |
| Real tick loading, NBBO consolidation | `hft/data.py` | numba kernel; per-venue staleness timeout, self-crossed venue quotes dropped |
| Vectorised features/targets (research) | `hft/features.py` | grid resampling to avoid burst inflation |
| Online features + tiny alpha | `hft/signals.py` | parity with offline path is unit-tested |
| Strategies | `hft/strategies.py` | `PassiveMM` with alpha pull + inventory cap; baselines and control |
| Replay venue | `hft/replay.py` | latency, queue position, cancel/fill races, fees, kill switch |
| Fees | `hft/fees.py` | free / IBKR Pro tiered / prop rebate tier (approximations, parameters) |
| Risk | `hft/risk.py` | pre-trade checks, split order/cancel message budgets, loss kill switch |
| IBKR adapter | `hft/ibkr_venue.py`, `hft/run_live.py` | dry-run default, paper-port guard, latency stats, tick recorder |
| Research | `hft/research/` | `signal_study`, `mm_study`, `wide_spread_study`, `bench` |

### 3.1 Replay timeline semantics

```
market event at t  -> venue state & fills updated at t -> strategy sees it at t + md_latency
strategy order     -> reaches the venue at t_decision + order_latency  (cancels and fills RACE)
venue fill at t    -> strategy learns of it at t + order_latency
```

Same-millisecond ordering: trades are processed before quotes (a trade stamped at ms *t* causes the quote update stamped
at *t*). Trade side is signed against the quote **before** the trade, in the offline features, the replay and the live
path alike (an offline/online mismatch here was found and fixed during development).

### 3.2 Fill model (conservative L1 queue model)

* A passive order joins the **back** of the visible aggregate queue (`queue_ahead = size at touch * queue_pessimism`).
* Trades at our price consume `queue_ahead` first; the remainder fills us. Trades strictly through our price fill us fully.
* Visible-size drops at our level not explained by trades are cancellations: `queue_model='prob'` shrinks `queue_ahead`
  proportionally, `'none'` never does (only capped by the visible size).
* Orders resting behind a market that moved away cannot fill until the market trades through them.
* Post-only orders that would cross are rejected; marketable orders fill at the touch as taker.

Known approximations: consolidated NBBO treated as one venue (consolidated trades + aggregated queue roughly offset),
our orders do not move the market, no hidden/iceberg liquidity, no odd-lot or ISO effects, marketable orders fill fully
at the touch, IBKR's per-order minimum commission applied per fill (slightly conservative), SEC/TAF fees at a current-ish
rate even though the sample is from 2013.

## 4. Data and evidence

Real AlgoSeek top-of-book tick data shipped with LEAN: SPY, BAC, AIG, IBM, 2013-10-07..11 (5 sessions). Quote rows are
per-exchange updates; a naive consolidation produced crossed markets from stale quotes on quiet venues, so each venue
side expires after a staleness timeout (200 ms for SPY, 3 s for sparsely quoted names) and self-crossed venue quotes are
ignored. The timeout was chosen by the fraction of trades printing inside the NBBO (about 87% for SPY) and by the
crossed-market rate (about 3-4%, those events are skipped).

### 4.1 Signal study (`hft/reports/signal_study.txt`, OOS = last two days, 1 s horizon, 100 ms grid)

| Symbol | imbalance IC | GBM OOS IC | mean 1 s mid move in model top 5% / bottom 5% (ticks) |
|---|---|---|---|
| SPY | 0.185 | 0.196 | +0.172 / -0.161 |
| AIG | 0.140 | 0.153 | +0.095 / -0.082 |
| IBM | 0.079 | 0.112 | +0.072 / -0.054 |
| BAC | 0.057 | 0.061 | +0.010 / -0.011 |

Traps found and handled:
* Event-time sampling inflates imbalance IC (SPY 0.295 vs 0.185 on a time grid) because quote bursts are over-weighted.
  All economics use the grid.
* A trade-price target is polluted by bid/ask bounce a taker cannot capture, so targets are mid-based.
* SPY imbalance IC by horizon is 0.08 (100 ms), 0.15 (500 ms), 0.19 (1 s), 0.19 (5 s), 0.11 (30 s). The less liquid
  names peak at 5-30 s (e.g. BAC 0.06 at 1 s but 0.18 at 30 s), which is a slower and more IBKR-compatible but also a
  fee-dominated regime that was not explored.

Economics: the best bucket moves **+0.17 ticks** in a second; a taker pays at least half the spread (0.5 tick) per side
plus fees (IBKR Pro tiered is about 0.67 tick per side when taking; about 0.17 tick when making, before the SEC/TAF
charges on sells, which add roughly 0.5 tick on a $167 stock). **Every taking strategy is unprofitable by a wide margin**,
regardless of model quality.

### 4.2 Market-making study (`hft/reports/mm_study.txt`)

Alpha (3-feature ridge) and the pull threshold were fitted/selected on Oct 7-9 only; Oct 10-11 are the test days.

**Does the alpha add value?** (rebate-tier fees, 100-share quotes, 26 thirty-minute blocks, paired bootstrap). Note that
by 20 ms the alpha still beats a no-information control but no longer beats simply quoting both sides.

| One-way latency | alpha pull vs naive | P(diff <= 0) | alpha pull vs shuffled-alpha control | P(diff <= 0) |
|---|---|---|---|---|
| 0 ms | +$33/block | 0.021 | +$105/block | 0.000 |
| 1 ms | +$37/block | 0.004 | +$84/block | 0.000 |
| 20 ms | +$2/block | 0.471 | +$49/block | 0.000 |

**Latency decay** (two test days, rebate-tier fees, net PnL after liquidation cost):

| One-way latency | naive | alpha pull | shuffled control |
|---|---|---|---|
| 0 | +$576 | **+$1,436** | -$1,284 |
| 1 ms | +$82 | **+$1,050** | -$1,129 |
| 5 ms | -$46 | +$647 | -$1,047 |
| 20 ms | +$243 | +$295 | -$990 |
| 50 ms | +$38 | -$31 | -$895 |
| 100 ms | -$36 | -$180 | -$852 |
| 250 ms | -$364 | -$118 | -$867 |
| 1 s | -$553 | -$413 | -$515 |

Mechanism (from the fill diagnostics): spread capture stays near +0.4-0.5 ticks while the 1 s markout (adverse selection)
is -0.26 ticks at zero latency and -0.37 to -0.43 ticks at 50 ms-1 s. Latency erodes the *protection*, not the capture.

**Fees** (alpha pull, 1 ms): free +$808, rebate-tier +$1,050, **IBKR Pro tiered -$3,306**.
**Queue assumptions** (alpha pull, 1 ms, rebate tier): joining pessimism 0.5x +$2,319; 1x +$1,050; 2x +$640; no-cancel
model +$238; no-cancel with 2x pessimism -$20. The sign survives most assumptions but the magnitude varies by more than
10x, and these assumptions cannot be calibrated without real fills.

**Statistical power is low**: two test days, one symbol, 2013, per-share edge of roughly 0.1 cent. The shuffled control
and the block bootstrap are what make the *alpha-versus-baseline* comparison credible; the *absolute* PnL is not a forecast.

### 4.2b Does a wider spread rescue it? No (`hft/reports/wide_spread_ibm.txt`)

Hypothesis: on a 3-tick-spread name (IBM in this sample), quoting one tick inside the touch captures far more per fill
(+1.2 ticks vs +0.5 on SPY) and might survive IBKR fees and latency. Same protocol (alpha and threshold on Oct 7-9,
test on Oct 10-11, IBKR's 50 msg/s cap enforced, 250 ms minimum requote interval when latency > 0).

| Fees | Latency | touch MM | inside MM | inside MM + alpha pull |
|---|---|---|---|---|
| free | 0 | -$1,306 | -$1,285 | +$33 |
| free | 50 ms | -$989 | -$775 | -$282 |
| free | 250 ms | -$849 | -$613 | -$283 |
| IBKR Pro tiered | 0 | -$3,569 | -$3,379 | -$1,614 |
| IBKR Pro tiered | 50 ms | -$2,549 | -$2,332 | -$1,632 |
| IBKR Pro tiered | 250 ms | -$2,228 | -$1,980 | -$1,504 |

The extra spread capture is more than offset by adverse selection (1 s markout about -0.5 tick, 5 s markout about -1 tick),
so **every IBKR configuration loses money and even the zero-fee rows are at best break-even**. Caveat: the IBM sample is
sparse (about 22k NBBO events per day), so fills are less trustworthy than for SPY; this is a negative result on weaker data.

### 4.3 Measured speed of this implementation (`hft/reports/bench.txt`)

* Replay: about 185k market events/s including simulator, risk gate and strategy (1.19M events in 6.4 s, one SPY day).
* `PassiveMM.on_quote` (features + alpha + quote logic), CPython 3.11: **p50 2.4 us, p99 4.0 us, p99.9 5.7 us**; inside the
  simulator the worst callback was 1.8 ms (garbage collection / scheduler jitter), which is the real argument for a
  non-GC language on a colocated path.

The strategy compute is thousands of times smaller than IBKR's network latency, so **rewriting in C++ gains nothing on the
IBKR path**. It only matters for tiers A/B.

## 5. Latency budget

| Stage | Tier A (typical) | Tier B (typical) | IBKR path |
|---|---|---|---|
| Exchange -> NIC | 1-5 us | 20-200 us | WAN + IBKR server + TWS: ms to tens of ms |
| Decode / book update | 0.1-1 us | 1-5 us | Java TWS + socket + `ib_async` decode: ~ms |
| Features + alpha + decision | 0.1-1 us | 1-10 us | **2.4 us (measured, Python)** |
| Pre-trade risk | 0.05-0.5 us | 0.5-2 us | ~1 us (this `RiskGate`) |
| Encode + send + exchange ack | 1-10 us | 50-500 us | ms to hundreds of ms |

`LatencyStats` records `feed_age_us` (exchange timestamp to arrival; includes clock skew, so only trust it with NTP/PTP
discipline), `tick_to_decision_us` and `place_call_us` live. **Record these on your own connection first**; the
replay curve in 4.2 tells you what each millisecond is worth.

## 6. Risk design

Layered, and identical in replay and live:

1. **Pre-trade gate** (`RiskGate`): max order size, max notional, price band versus reference mid (fat finger), position
   limit counting working orders, message-rate token bucket. Mirrors the pre-trade controls broker-dealers must apply to
   market access **(verify the current SEC 15c3-5 expectations)**.
2. **Reserved cancel budget**: orders and cancels draw on separate halves of the message budget so a burst of new orders
   cannot starve cancels. Cancels always pass when halted.
3. **Loss kill switch**: mark-to-mid PnL after fees below the limit halts trading and cancels everything.
4. **Session window**: no trading in the first/last 5 minutes (auction noise); end-of-session cancel-all.
5. **Live-specific**: dry-run by default, read-only connection unless `--send`, refusal of live ports without an explicit
   flag, refusal to combine recording with order sending.
6. **Connection and state safety** (built, unit-tested against a fake IB, never run against a real gateway):
   * stale-feed halt: no quote for `--stale-feed-s` (default 2 s) during the session pulls all orders and blocks quoting
     until fresh data arrives;
   * disconnect handling: quoting stops immediately; bounded reconnect attempts (default 5, every 5 s) then a hard halt;
     after reconnecting, tick subscriptions are re-created and state is reconciled;
   * reconciliation: broker position wins (mismatches are counted), cash is rebuilt from broker executions, and every
     order still resting at the broker for the symbol is cancelled rather than adopted;
   * execution reports are de-duplicated by `execId`, and a fill for an order we already forgot still updates position
     and cash;
   * the loss kill switch measures PnL on session trades only (a position carried in at startup is excluded; an earlier
     version counted its full market value as profit).
7. **Still missing**: a broker-side dead-man switch (the above cannot cancel orders while the link is down; consider
   short-lived GTD orders **(verify)**), self-trade prevention across strategies, LULD/halt awareness, commission-report
   updates arriving after the fill, and partial-fill/ack-latency instrumentation beyond `place_call_us`.

## 7. How to use it

```bash
# research (data ships with the repo)
python -m hft.research.signal_study spy bac aig ibm
python -m hft.research.mm_study          # ~30 min, writes hft/reports/mm_study.txt
python -m hft.research.bench

# 1) record your own current ticks (no orders are ever sent in --record mode)
python -m hft.run_live --symbol SPY --seconds 3600 --record
# 2) refit the alpha on the recording, re-run the latency/fee/queue sweeps on it
# 3) dry-run: logs the orders it WOULD place, measures tick-to-decision on your connection
python -m hft.run_live --symbol SPY --seconds 600 --alpha alpha.json
# 4) only then, tiny size on a paper port, then tiny size live
python -m hft.run_live --symbol SPY --send --qty 100 --max-pos 200
```

## 8. What I would do next, in order

1. **Record current data** from the intended connection and rerun every study; the 2013 sample is a different regime
   (SEC fee and rebate levels, queue dynamics, tick sizes, venue mix have all changed).
2. **Measure real latency** with `run_live` in dry-run: if tick-to-decision plus order round trip is above about 20-50 ms,
   stop; the replay says the edge is gone.
3. **Decide the venue question**: with retail fees and API latency the answer from this evidence is no. The options are
   a DMA/FIX broker with rebate pricing and a nearby host, or colocation, which changes the cost structure entirely.
4. **Calibrate the queue model with small live fills** (paper fills are synthetic). Compare simulated versus actual fill
   rate and markout per order; tune `queue_pessimism` until they match.
5. Add the missing production risk items in section 6.7 before any non-trivial size.
6. Only then consider richer alpha (full depth, cross-asset lead-lag with ES futures/QQQ, trade-through/ISO detection),
   larger universe, and a compiled hot path for tier A/B.
