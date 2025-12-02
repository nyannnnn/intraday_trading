import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import UNIVERSE, DATA_DIR, BAR_INTERVAL_MIN


def get_yf_interval(bar_minutes: int) -> str:
    """Map a bar interval in minutes to a yfinance intraday interval string."""
    mapping = {
        1: "1m",
        2: "2m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "60m",
    }
    return mapping.get(bar_minutes, f"{bar_minutes}m")


def download_symbol_intraday(
    symbol: str,
    period: str = "60d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download intraday OHLCV for a symbol from Yahoo Finance via yfinance."""
    interval = get_yf_interval(BAR_INTERVAL_MIN)
    print(f"[{symbol}] Downloading {period} of data at {interval} bars...")

    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df.empty:
        print(f"[{symbol}] WARNING: received empty DataFrame from yfinance.")
        return df

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "datetime"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    return df


def main():
    """Download intraday data for the universe and persist raw CSVs to disk."""
    raw_dir: Path = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving raw data to: {raw_dir.resolve()}")

    for sym in UNIVERSE:
        try:
            df = download_symbol_intraday(sym, period="60d")
            if df.empty:
                continue

            out_path = raw_dir / f"{sym}.csv"
            df.to_csv(out_path, index=False)
            print(f"[{sym}] Saved {len(df)} rows to {out_path.name}")
        except Exception as e:
            print(f"[{sym}] ERROR while downloading: {e}")

        time.sleep(1.0)

    print("Done.")


if __name__ == "__main__":
    main()
