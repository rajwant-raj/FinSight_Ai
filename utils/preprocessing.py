"""
FinSight AI — Preprocessing
Feature engineering and data preparation for ML models.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def add_moving_averages(
    df: pd.DataFrame,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Add Simple Moving Average columns for specified window sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "Close" column.
    windows : list of int
        Moving average window sizes (default: [20, 50, 100]).

    Returns
    -------
    pd.DataFrame
        Copy of the input with new SMA columns appended.
    """
    if windows is None:
        windows = [20, 50, 100]

    result = df.copy()
    for w in windows:
        result[f"SMA_{w}"] = result["Close"].rolling(window=w).mean()
    return result


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Daily_Return' column (percentage change of Close).

    Returns
    -------
    pd.DataFrame
        Copy with the new column appended.
    """
    result = df.copy()
    result["Daily_Return"] = result["Close"].pct_change() * 100
    return result


def prepare_ml_features(
    df: pd.DataFrame,
    lags: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features for the stock price prediction model.

    Creates:
      - Lag features (Close price for the previous *lags* days)
      - Rolling statistics (mean and std for 5, 10, 20 windows)
      - Day-of-week and month as cyclical features
      - Daily return and volatility proxies

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data.
    lags : int
        Number of lag days (default: 5).

    Returns
    -------
    tuple
        (X features DataFrame, y target Series)
        Rows containing NaN are dropped.
    """
    data = df.copy()

    # ── Lag features ──
    for i in range(1, lags + 1):
        data[f"Close_Lag_{i}"] = data["Close"].shift(i)

    # ── Rolling statistics ──
    for window in [5, 10, 20]:
        data[f"Roll_Mean_{window}"] = (
            data["Close"].rolling(window=window).mean()
        )
        data[f"Roll_Std_{window}"] = (
            data["Close"].rolling(window=window).std()
        )

    # ── Daily return ──
    data["Daily_Return"] = data["Close"].pct_change()

    # ── Price-based ratios ──
    data["High_Low_Ratio"] = data["High"] / data["Low"]
    data["Close_Open_Ratio"] = data["Close"] / data["Open"]

    # ── Volume change ──
    data["Volume_Change"] = data["Volume"].pct_change()

    # ── Calendar features (cyclical encoding) ──
    data["Day_of_Week"] = data.index.dayofweek
    data["Month"] = data.index.month
    data["Day_Sin"] = np.sin(2 * np.pi * data["Day_of_Week"] / 5)
    data["Day_Cos"] = np.cos(2 * np.pi * data["Day_of_Week"] / 5)
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)

    # ── Target: next day's closing price ──
    data["Target"] = data["Close"].shift(-1)

    # Drop rows with NaN (from lags / rolling / target shift)
    data.dropna(inplace=True)

    # ── Feature / target split ──
    feature_cols = [
        col
        for col in data.columns
        if col not in ["Target", "Open", "High", "Low", "Close", "Volume"]
    ]
    X = data[feature_cols]
    y = data["Target"]

    return X, y
