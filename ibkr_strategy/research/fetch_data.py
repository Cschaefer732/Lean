"""Download daily OHLCV (split-adjusted + total-return adjusted close) for the research universe from Yahoo Finance."""
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
    "TLT", "IEF", "SHY", "LQD", "TIP",
    "GLD", "DBC", "VGK", "EWJ", "XLU", "XLP", "XLV",
]
OUT = Path(__file__).parent / "data"


def fetch(symbol: str) -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        "?period1=631152000&period2=1791676800&interval=1d&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)["chart"]["result"][0]
    quote = payload["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
            "adjclose": payload["indicators"]["adjclose"][0]["adjclose"],
        },
        index=pd.to_datetime(payload["timestamp"], unit="s").normalize(),
    )
    df.index.name = "date"
    return df.dropna(subset=["close"]).loc[~df.index.duplicated(keep="last")]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for sym in sys.argv[1:] or UNIVERSE:
        df = fetch(sym)
        df.to_csv(OUT / f"{sym}.csv")
        print(f"{sym:5s} {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} rows)")


if __name__ == "__main__":
    main()
