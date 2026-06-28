"""
FinSight AI — Data Loader
Handles stock data retrieval via yfinance with Streamlit caching.
"""

from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf


# ── Stock Universe ─────────────────────────────────────────────────
# Maps display names to yfinance ticker symbols.
# Indian stocks automatically include the ".NS" (NSE) suffix.
STOCK_MAP: Dict[str, str] = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOGL)": "GOOGL",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Reliance (RELIANCE)": "RELIANCE.NS",
    "TCS (TCS)": "TCS.NS",
    "Infosys (INFY)": "INFY.NS",
}


def get_stock_list() -> list:
    """Return the list of supported stock display names."""
    return list(STOCK_MAP.keys())


def get_ticker(display_name: str) -> str:
    """Convert a display name to its yfinance ticker symbol."""
    return STOCK_MAP.get(display_name, display_name)


@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(
    ticker: str,
    start_date: date,
    end_date: date,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        The yfinance ticker symbol (e.g., "AAPL" or "RELIANCE.NS").
    start_date : date
        Start of the date range.
    end_date : date
        End of the date range.

    Returns
    -------
    tuple
        (DataFrame of OHLCV data, error_message or None)
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )

        if data is None or data.empty:
            return None, (
                f"No market data found for **{ticker}** in the selected "
                f"date range. The market may have been closed, or the "
                f"ticker symbol may be invalid."
            )

        # Flatten MultiIndex columns if present (yfinance >= 0.2.36)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ensure standard column names
        data.columns = [col.strip().title() for col in data.columns]

        # Guarantee the index is a DatetimeIndex named "Date"
        data.index.name = "Date"

        return data, None

    except Exception as exc:
        return None, (
            f"Failed to fetch data for **{ticker}**. "
            f"Please check your internet connection and try again.\n\n"
            f"*Error detail:* `{exc}`"
        )


@st.cache_data(ttl=600, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """
    Retrieve basic company information for the given ticker.

    Returns a dictionary with keys like 'shortName', 'sector',
    'marketCap', etc.  Returns an empty dict on failure.
    """
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception:
        return {}


def get_default_date_range() -> Tuple[date, date]:
    """Return a sensible default date range (1 year ending today)."""
    end = date.today()
    start = end - timedelta(days=365)
    return start, end
