"""Strategy lab CLI: walk-forward test any set of strategies on identical, leak-free out-of-sample data.

    python -m lab.run                                   # all built-in strategies
    python -m lab.run --strategies rf,gbm,ga,rl,ens:rf+gbm+ga --oos-start 2012-01-01
    python -m lab.run --list
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CASH, ROOT, Market, build_market
from .metrics import (backtest_weights, block_bootstrap_sharpe_diff, daily_sharpe, deflated_sharpe,
                      information_coefficient, summarize)
from .models import REGISTRY, get_factory
from .models.ga import genome_weights, N_GENES
from .walkforward import schedule, walk_forward

DEFAULT = "rf,extra_trees,gbm,ridge,mlp,gru,ga,rl,ens:rf+gbm+mlp+ga+rl"
OUT = Path(__file__).resolve().parent / "reports"


def baseline_weights(mkt: Market, dates: pd.DatetimeIndex) -> dict[str, tuple[pd.DataFrame, int]]:
    """name -> (weekly weights, smoothing weeks). The production strategy already smooths internally."""
    from momentum_rotation import StrategyConfig, target_weights

    cfg = StrategyConfig()
    prod = pd.DataFrame({d: target_weights(mkt.px.loc[:d], cfg) for d in dates}).T.reindex(columns=mkt.assets).fillna(0.0)
    zero = pd.DataFrame(0.0, index=dates, columns=mkt.assets)
    spy, sixty, ew = zero.copy(), zero.copy(), zero.copy()
    spy["SPY"] = 1.0
    sixty["SPY"], sixty["IEF"] = 0.6, 0.4
    ew[:] = 1 / len(mkt.assets)
    return {
        "BENCH production rotation": (prod, 1),
        "BENCH SPY buy&hold": (spy, 1),
        "BENCH 60/40 SPY/IEF": (sixty, 1),
        "BENCH equal-weight 12": (ew, 1),
    }


def random_rule_sets(mkt: Market, rows: np.ndarray, n: int, seed: int = 123) -> np.ndarray:
    """Sharpe of n *unfitted* random rotation rule sets: the null distribution for 'skill from rule choice'."""
    rng = np.random.default_rng(seed)
    dates = mkt.dates[rows]
    out = []
    for _ in range(n):
        g = rng.random(N_GENES)
        W = pd.DataFrame(genome_weights(mkt, g, rows), index=dates, columns=mkt.assets)
        bt = backtest_weights(W, mkt)
        out.append(summarize(bt["ret"].loc[dates[0]:], mkt.rets[CASH])["Sharpe"])
    return np.array(out)


def run(args) -> pd.DataFrame:
    t0 = time.time()
    mkt = build_market(horizon=args.horizon)
    blocks = schedule(mkt, args.oos_start, args.retrain_weeks)
    oos_rows = np.concatenate([b[1] for b in blocks])
    dates = mkt.dates[oos_rows]
    rf = mkt.rets[CASH]
    print(f"OOS {dates[0].date()} -> {dates[-1].date()}  ({len(dates)} weeks, {len(blocks)} retrains, "
          f"first model trains on {len(blocks[0][0])} weeks)\n", flush=True)

    weights: dict[str, tuple[pd.DataFrame, int]] = {}
    scores: dict[str, pd.DataFrame] = {}
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    plain = [n for n in names if not n.startswith("ens:")]
    for spec in dict.fromkeys(plain + [m for n in names if n.startswith("ens:") for m in n[4:].split("+")]):
        t = time.time()
        key = OUT.parent / "cache" / f"{spec}_{args.oos_start}_{args.retrain_weeks}_{args.horizon}.pkl"
        if args.cache and key.exists():
            W, S = pd.read_pickle(key)
        else:
            W, S = walk_forward(get_factory(spec), mkt, args.oos_start, args.retrain_weeks, return_scores=True)
            if args.cache:
                key.parent.mkdir(exist_ok=True)
                pd.to_pickle((W, S), key)
        weights[spec] = (W, args.smooth_weeks)
        if S is not None:
            scores[spec] = S
        print(f"  {spec:12s} walk-forward done in {time.time() - t:5.1f}s", flush=True)
    for n in names:
        if n.startswith("ens:"):
            members = n[4:].split("+")
            weights[n] = (sum(weights[m][0] for m in members) / len(members), args.smooth_weeks)
    candidates = [n for n in names if n in weights]
    shown = [n for n in weights if n in names]

    allw = {**baseline_weights(mkt, dates), **{n: weights[n] for n in shown}}
    runs = {}
    for name, (W, sm) in allw.items():
        bt = backtest_weights(W, mkt, smooth_weeks=sm, cost_bps=args.cost_bps)
        runs[name] = bt.loc[dates[0]:]

    base = runs["BENCH production rotation"]["ret"]
    spy = runs["BENCH SPY buy&hold"]["ret"]
    trial_sr = np.array([daily_sharpe(runs[n]["ret"], rf) for n in candidates])
    rows = []
    for name, bt in runs.items():
        s = summarize(bt["ret"], rf, bt["turnover"])
        row = {"strategy": name, **s}
        if name in candidates:
            row["DSR"] = deflated_sharpe(bt["ret"], rf, trial_sr)
            row["P(>prod)"] = block_bootstrap_sharpe_diff(bt["ret"], base, rf)["p_gt0"]
            row["P(>SPY)"] = block_bootstrap_sharpe_diff(bt["ret"], spy, rf)["p_gt0"]
            if name in scores:
                row.update(information_coefficient(scores[name], mkt))
        rows.append(row)
    board = pd.DataFrame(rows).set_index("strategy")

    null = random_rule_sets(mkt, oos_rows, args.null_runs)
    board["pct_vs_random"] = [
        float((null < board.loc[n, "Sharpe"]).mean()) if n in candidates else np.nan for n in board.index
    ]
    write_report(board, runs, mkt, null, dates, args, len(candidates), time.time() - t0)
    return board


def write_report(board, runs, mkt, null, dates, args, n_trials, secs) -> None:
    OUT.mkdir(exist_ok=True)
    board.to_csv(OUT / "leaderboard.csv")
    fmt = board.copy()
    for c in ("CAGR", "Vol", "MaxDD"):
        fmt[c] = (fmt[c] * 100).map("{:.1f}%".format)
    for c in ("Sharpe", "Sortino", "Calmar", "TO/yr", "IC", "IC_t"):
        if c in fmt:
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for c in ("DSR", "P(>prod)", "P(>SPY)", "pct_vs_random", "IC_hit"):
        if c in fmt:
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    cols = [c for c in ("CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "TO/yr", "IC", "IC_t", "DSR", "P(>prod)",
                        "P(>SPY)", "pct_vs_random") if c in fmt]
    years = pd.DataFrame({n: b["ret"].groupby(b.index.year).apply(lambda r: (1 + r).prod() - 1) * 100 for n, b in runs.items()}).round(1)
    md = [
        f"# Strategy lab leaderboard",
        "",
        f"Out-of-sample {dates[0].date()} to {dates[-1].date()} ({len(dates)} weekly decisions), retrained every "
        f"{args.retrain_weeks} weeks on an expanding window with a {args.horizon + 1}-week purge. Costs {args.cost_bps} bps/turnover, "
        f"execution at the decision close, {args.smooth_weeks}-week weight smoothing for all candidates. Runtime {secs:.0f}s.",
        "",
        board_to_md(fmt[cols]),
        "",
        "Column guide: `DSR` = deflated Sharpe (probability true Sharpe > 0 after best-of-"
        f"{n_trials} selection, >0.95 is convincing); `P(>prod)`/`P(>SPY)` = block-bootstrap probability that the Sharpe beats the "
        "production rotation / SPY (0.5 = coin flip, >0.95 = strong); `IC` = mean rank correlation of predictions with realised "
        f"{args.horizon}-week excess returns (`IC_t` uses the overlap-adjusted sample); `pct_vs_random` = share of "
        f"{len(null)} unfitted random rotation rule sets with a lower Sharpe "
        f"(random sets: median {np.median(null):.2f}, 90th pct {np.percentile(null, 90):.2f}).",
        "",
        "## Calendar-year returns (%)",
        "",
        board_to_md(years.T.astype(str)),
    ]
    (OUT / "leaderboard.md").write_text("\n".join(md) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 6))
        for n, b in runs.items():
            (1 + b["ret"]).cumprod().plot(ax=ax, label=n, lw=2.2 if n.startswith("BENCH") else 1.2,
                                           ls="--" if n.startswith("BENCH") else "-")
        ax.set_yscale("log")
        ax.set_title("Walk-forward out-of-sample equity (log scale)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "equity.png", dpi=110)
    except Exception as e:  # plotting must never break a run
        print("plot skipped:", e)
    (OUT / "run_config.json").write_text(json.dumps(vars(args), indent=2))


def board_to_md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep, *body])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default=DEFAULT)
    ap.add_argument("--oos-start", default="2012-01-01")
    ap.add_argument("--retrain-weeks", type=int, default=52)
    ap.add_argument("--horizon", type=int, default=4, help="prediction horizon in weeks")
    ap.add_argument("--smooth-weeks", type=int, default=4)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--null-runs", type=int, default=150)
    ap.add_argument("--cache", action="store_true", help="reuse/store walk-forward weights in lab/cache (delete after code changes)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        get_factory("rf")
        print("strategies:", ", ".join(sorted(REGISTRY)), "| ensembles: ens:a+b+c")
        return
    board = run(args)
    pd.set_option("display.width", 250)
    print("\n" + (OUT / "leaderboard.md").read_text())


if __name__ == "__main__":
    main()
