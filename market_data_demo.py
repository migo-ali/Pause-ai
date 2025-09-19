"""Market data quickstart fetching data from multiple public APIs."""
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


def get_yfinance_data(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch recent hourly data for ``ticker`` from Yahoo Finance.

    Returns the dataframe of historical prices when successful, otherwise ``None``.
    """

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1h")
        if hist.empty:
            print(f"[Yahoo] ❌ No data returned for {ticker}.")
            return None

        print(f"[Yahoo] ✅ Data for {ticker}:\n", hist.tail())
        return hist
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Yahoo] ❌ Error: {exc}")
        return None


def get_alpha_vantage_sample(ticker: str) -> Optional[pd.DataFrame]:
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
    print(f"[Alpha Vantage DEMO] ✅ Data for {ticker}:\n", df.head())
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🚀 Starting Multi-Source Market Data Fetch...\n")

    get_yfinance_data(args.ticker)
    get_alpha_vantage_sample(args.ticker)
    show_polygon_demo_info()


if __name__ == "__main__":
    main()
