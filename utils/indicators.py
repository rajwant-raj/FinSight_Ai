"""
FinSight AI — Technical Indicators
Pure calculation functions for common financial indicators.
"""

import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average.

    The SMA smooths price data by averaging the closing price over the
    last *window* trading days.
    """
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponential Moving Average.

    The EMA gives more weight to recent prices, making it more
    responsive to new information than the SMA.
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes to
    evaluate overbought (>70) or oversold (<30) conditions.

    Uses the classic Wilder smoothing method.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Subsequent values use Wilder smoothing
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (
            avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]
        ) / period
        avg_loss.iloc[i] = (
            avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]
        ) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_daily_returns(series: pd.Series) -> pd.Series:
    """
    Daily percentage returns.

    Computed as the percentage change in closing price from one day
    to the next.
    """
    return series.pct_change() * 100


def calculate_volatility(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Rolling Volatility (annualized standard deviation of daily returns).

    Volatility measures the degree of variation in price over time.
    Higher values indicate more uncertainty and risk.
    """
    daily_returns = series.pct_change()
    rolling_std = daily_returns.rolling(window=window).std()
    # Annualize (approx. 252 trading days)
    annualized = rolling_std * np.sqrt(252) * 100
    return annualized
