"""Build a LEAN data folder overlay (daily equity bars + factor/map files) from research/data/*.csv.

The overlay symlinks everything in the repo's Data/ and replaces only the tickers we generate, so the repo's own
data is never modified. Prices are stored split-adjusted but not dividend-adjusted; dividends are expressed through
factor files so LEAN's default `Adjusted` normalisation yields the same total-return series as the research code.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from momentum_rotation import StrategyConfig  # noqa: E402

NASDAQ = {"QQQ"}


def write_symbol(sym: str, df: pd.DataFrame, usa: Path) -> None:
    t = sym.lower()
    lines = [
        f"{d:%Y%m%d} 00:00,{round(r.open * 1e4)},{round(r.high * 1e4)},{round(r.low * 1e4)},"
        f"{round(r.close * 1e4)},{int(r.volume)}"
        for d, r in df.iterrows()
    ]
    with zipfile.ZipFile(usa / "daily" / f"{t}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{t}.csv", "\n".join(lines) + "\n")

    ratio = df["adjclose"] / df["close"]
    factor = (ratio / ratio.iloc[-1]).round(7)
    rows = [f"{df.index[0]:%Y%m%d},{factor.iloc[0]:.7f},1,{df['close'].iloc[0]:.2f}"]
    for i in range(len(df) - 1):
        if abs(factor.iloc[i + 1] / factor.iloc[i] - 1) > 1e-4:
            rows.append(f"{df.index[i]:%Y%m%d},{factor.iloc[i]:.7f},1,{df['close'].iloc[i]:.2f}")
    rows.append("20501231,1,1,0")
    (usa / "factor_files" / f"{t}.csv").write_text("\n".join(rows) + "\n")

    exch = "Q" if sym in NASDAQ else "P"
    (usa / "map_files" / f"{t}.csv").write_text(f"{df.index[0]:%Y%m%d},{t},{exch}\n20501231,{t},{exch}\n")


def link_dir_except(src: Path, dst: Path, skip: set[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name.lower() in skip:
            continue
        link = dst / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean-data", default=str(ROOT.parent / "Data"))
    ap.add_argument("--csv-dir", default=str(ROOT / "research" / "data"))
    ap.add_argument("--out", default=str(ROOT / "lean" / "data"))
    args = ap.parse_args()

    src, out, csv_dir = Path(args.lean_data), Path(args.out), Path(args.csv_dir)
    symbols = StrategyConfig().symbols
    tickers = {s.lower() for s in symbols}

    out.mkdir(parents=True, exist_ok=True)
    link_dir_except(src, out, {"equity"})
    link_dir_except(src / "equity", out / "equity", {"usa"})
    usa = out / "equity" / "usa"
    link_dir_except(src / "equity" / "usa", usa, {"daily", "factor_files", "map_files"})
    for sub in ("daily", "factor_files", "map_files"):
        stale = usa / sub
        if stale.exists():
            for f in stale.iterdir():
                f.unlink()
        link_dir_except(src / "equity" / "usa" / sub, stale, {f"{t}.zip" for t in tickers} | {f"{t}.csv" for t in tickers})

    for sym in symbols:
        df = pd.read_csv(csv_dir / f"{sym}.csv", index_col=0, parse_dates=True).dropna()
        write_symbol(sym, df, usa)
        print(f"{sym:4s} {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} bars)")
    print(f"LEAN data overlay written to {out}")


if __name__ == "__main__":
    main()
