"""Market data quickstart fetching data from multiple public APIs."""
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


def positive_int(value: str) -> int:
    """Validate that ``value`` represents a positive integer for CLI arguments."""

    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - handled by argparse at runtime
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")

    return parsed


def _preview_dataframe(
    dataframe: pd.DataFrame, *, rows: Optional[int], shuffle: bool
) -> pd.DataFrame:
    """Return a preview of ``dataframe`` respecting row and shuffle options."""

    if dataframe.empty:
        return dataframe

    if rows is None or rows >= len(dataframe):
        if shuffle:
            return dataframe.sample(frac=1)
        return dataframe

    if shuffle:
        return dataframe.sample(n=rows)

    return dataframe.tail(rows)


def _preview_descriptor(*, limited: bool, shuffle: bool) -> str:
    """Human-friendly text describing the preview that will be displayed."""

    if shuffle:
        return "shuffled"
    if limited:
        return "most recent"
    return "all available"


def get_yfinance_data(
    ticker: str, *, rows: Optional[int] = 5, shuffle: bool = False
) -> Optional[pd.DataFrame]:
    """Fetch recent hourly data for ``ticker`` from Yahoo Finance.

    Returns the dataframe of historical prices when successful, otherwise ``None``.
    """

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1h")
        if hist.empty:
            print(f"[Yahoo] ❌ No data returned for {ticker}.")
            return None

        preview = _preview_dataframe(hist, rows=rows, shuffle=shuffle)
        descriptor = _preview_descriptor(
            limited=rows is not None and rows < len(hist), shuffle=shuffle and not preview.empty
        )
        print(
            f"[Yahoo] ✅ Data for {ticker} (showing {descriptor} {len(preview)} rows):\n",
            preview,
        )
        return hist
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Yahoo] ❌ Error: {exc}")
        return None


def get_alpha_vantage_sample(
    ticker: str, *, rows: Optional[int] = 5, shuffle: bool = False
) -> Optional[pd.DataFrame]:
    """Fetch hourly data using Alpha Vantage's demo API key.

    The response is limited and primarily intended for demonstration purposes.
    """

    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}&interval=60min&datatype=json&apikey=demo"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Alpha Vantage] ❌ Error: {exc}")
        return None

    series_key = "Time Series (60min)"
    if series_key not in data:
        print(f"[Alpha Vantage] ❌ No data: {data}")
        return None

    df = pd.DataFrame.from_dict(data[series_key], orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    preview = _preview_dataframe(df, rows=rows, shuffle=shuffle)
    descriptor = _preview_descriptor(
        limited=rows is not None and rows < len(df), shuffle=shuffle and not preview.empty
    )
    print(
        f"[Alpha Vantage DEMO] ✅ Data for {ticker} (showing {descriptor} {len(preview)} rows):\n",
        preview,
    )
    return df


def show_polygon_demo_info() -> None:
    """Print information about Polygon.io demo access."""

    print("[Polygon.io] ❌ Requires API Key for real-time data.")
    print(
        "🔗 But you can explore demo access here: https://polygon.io/docs/stocks/getting-started"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch sample market data for a ticker symbol using Yahoo Finance and "
            "Alpha Vantage's demo endpoint."
        )
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Ticker symbol to request (default: AAPL)",
    )
    parser.add_argument(
        "--rows",
        type=positive_int,
        default=5,
        help="Number of rows to display from each data source (default: 5).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the preview rows before displaying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🚀 Starting Multi-Source Market Data Fetch...\n")

    get_yfinance_data(args.ticker, rows=args.rows, shuffle=args.shuffle)
    get_alpha_vantage_sample(args.ticker, rows=args.rows, shuffle=args.shuffle)
    show_polygon_demo_info()


if __name__ == "__main__":
    main()
