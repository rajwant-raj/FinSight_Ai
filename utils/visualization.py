"""
FinSight AI — Visualization
All Plotly chart functions for the dashboard.
Charts automatically adapt to light/dark theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# Accent palette
_COLORS = {
    "blue": "#3b82f6",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "red": "#ef4444",
    "amber": "#f59e0b",
    "purple": "#8b5cf6",
    "pink": "#ec4899",
    "indigo": "#6366f1",
}


def _is_light_theme() -> bool:
    """Check whether the app is in light mode."""
    return st.session_state.get("theme", "dark") == "light"


def _get_layout_defaults() -> dict:
    """Return Plotly layout defaults based on the active theme."""
    light = _is_light_theme()
    return dict(
        template="plotly_white" if light else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#1e293b" if light else "#e2e8f0",
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#334155" if light else "#cbd5e1"),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.06)" if light else "rgba(255,255,255,0.04)",
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="#475569" if light else "#94a3b8"),
            title_font=dict(color="#334155" if light else "#cbd5e1"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)" if light else "rgba(255,255,255,0.04)",
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="#475569" if light else "#94a3b8"),
            title_font=dict(color="#334155" if light else "#cbd5e1"),
        ),
    )


def _apply_layout(fig: go.Figure, title: str, yaxis_title: str = "") -> go.Figure:
    """Apply shared layout defaults and a title to a figure."""
    light = _is_light_theme()
    fig.update_layout(
        **_get_layout_defaults(),
        title=dict(
            text=title,
            font=dict(size=16, color="#1e293b" if light else "#f1f5f9"),
            x=0,
            xanchor="left",
        ),
        yaxis_title=yaxis_title,
        height=440,
    )
    return fig


# ── Chart Functions ──────────────────────────────────────────────────


def plot_closing_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Line chart of daily closing prices with gradient fill."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color=_COLORS["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    return _apply_layout(fig, f"{ticker} — Closing Price", "Price (USD)")


def plot_volume(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Bar chart of daily trading volume."""
    # Color bars green/red based on price direction
    colors = [
        _COLORS["green"] if row["Close"] >= row["Open"] else _COLORS["red"]
        for _, row in df.iterrows()
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=colors,
            marker_line_width=0,
            opacity=0.7,
            name="Volume",
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )
    return _apply_layout(fig, f"{ticker} — Trading Volume", "Volume")


def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """OHLC candlestick chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=_COLORS["green"],
            decreasing_line_color=_COLORS["red"],
            increasing_fillcolor=_COLORS["green"],
            decreasing_fillcolor=_COLORS["red"],
            name="OHLC",
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_layout(fig, f"{ticker} — Candlestick Chart", "Price (USD)")


def plot_moving_averages(
    df: pd.DataFrame,
    ticker: str,
    windows: list = None,
) -> go.Figure:
    """Overlay selected moving averages on the closing price."""
    if windows is None:
        windows = [20, 50, 100]

    palette = [_COLORS["cyan"], _COLORS["amber"], _COLORS["purple"]]
    fig = go.Figure()

    # Base closing price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color=_COLORS["blue"], width=1.5),
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )

    for i, w in enumerate(windows):
        col = f"SMA_{w}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=f"SMA {w}",
                    line=dict(color=palette[i % len(palette)], width=1.5, dash="dot"),
                    hovertemplate="$%{y:.2f}<extra></extra>",
                )
            )

    return _apply_layout(fig, f"{ticker} — Moving Averages", "Price (USD)")


def plot_daily_returns(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Histogram + line of daily percentage returns."""
    returns = df["Close"].pct_change() * 100

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Histogram(
            x=returns.dropna(),
            nbinsx=50,
            marker_color=_COLORS["indigo"],
            opacity=0.7,
            name="Distribution",
            hovertemplate="%{x:.2f}%<br>Count: %{y}<extra></extra>",
        )
    )
    return _apply_layout(fig, f"{ticker} — Daily Returns Distribution", "Frequency")


def plot_sma_ema(
    df: pd.DataFrame,
    ticker: str,
    period: int = 20,
) -> go.Figure:
    """Compare SMA and EMA for a given period."""
    from utils.indicators import calculate_ema, calculate_sma

    sma = calculate_sma(df["Close"], period)
    ema = calculate_ema(df["Close"], period)

    close_bg = "rgba(0,0,0,0.15)" if _is_light_theme() else "rgba(255,255,255,0.3)"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color=close_bg, width=1),
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sma,
            mode="lines",
            name=f"SMA {period}",
            line=dict(color=_COLORS["cyan"], width=2),
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema,
            mode="lines",
            name=f"EMA {period}",
            line=dict(color=_COLORS["amber"], width=2),
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    return _apply_layout(
        fig, f"{ticker} — SMA vs EMA ({period}-Day)", "Price (USD)"
    )


def plot_rsi(df: pd.DataFrame, ticker: str, period: int = 14) -> go.Figure:
    """RSI chart with overbought/oversold zones."""
    from utils.indicators import calculate_rsi

    rsi = calculate_rsi(df["Close"], period)

    fig = go.Figure()

    # Overbought / oversold bands
    fig.add_hline(y=70, line_dash="dash", line_color=_COLORS["red"], opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=_COLORS["green"], opacity=0.5)
    fig.add_hrect(y0=70, y1=100, fillcolor=_COLORS["red"], opacity=0.05)
    fig.add_hrect(y0=0, y1=30, fillcolor=_COLORS["green"], opacity=0.05)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi,
            mode="lines",
            name=f"RSI ({period})",
            line=dict(color=_COLORS["purple"], width=2),
            hovertemplate="%{y:.1f}<extra></extra>",
        )
    )

    fig.update_yaxes(range=[0, 100])
    return _apply_layout(fig, f"{ticker} — Relative Strength Index (RSI)", "RSI")


def plot_volatility(df: pd.DataFrame, ticker: str, window: int = 20) -> go.Figure:
    """Rolling annualized volatility."""
    from utils.indicators import calculate_volatility

    vol = calculate_volatility(df["Close"], window)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vol,
            mode="lines",
            name=f"Volatility ({window}d)",
            line=dict(color=_COLORS["pink"], width=2),
            fill="tozeroy",
            fillcolor="rgba(236,72,153,0.08)",
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
    )
    return _apply_layout(
        fig, f"{ticker} — Annualized Volatility ({window}-Day)", "Volatility (%)"
    )


def plot_prediction_results(
    y_actual: pd.Series,
    y_predicted: np.ndarray,
) -> go.Figure:
    """Actual vs Predicted scatter and line plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_actual.index,
            y=y_actual.values,
            mode="lines",
            name="Actual",
            line=dict(color=_COLORS["blue"], width=2),
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_actual.index,
            y=y_predicted,
            mode="lines",
            name="Predicted",
            line=dict(color=_COLORS["green"], width=2, dash="dot"),
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    return _apply_layout(fig, "Actual vs Predicted Closing Price", "Price (USD)")


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 10,
) -> go.Figure:
    """Horizontal bar chart of the top-N feature importances."""
    # Sort and take top N
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, _COLORS["blue"]], [1, _COLORS["cyan"]]],
            ),
            hovertemplate="%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(height=380)
    return _apply_layout(fig, "Top Feature Importances", "")
