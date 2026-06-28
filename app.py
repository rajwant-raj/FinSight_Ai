"""
FinSight AI — Financial Market Analytics & Stock Prediction Dashboard
Main Streamlit application.

Run with:
    streamlit run app.py
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    get_default_date_range,
    get_stock_info,
    get_stock_list,
    get_ticker,
    load_stock_data,
)
from utils.indicators import (
    calculate_daily_returns,
    calculate_ema,
    calculate_rsi,
    calculate_sma,
    calculate_volatility,
)
from utils.prediction import train_model
from utils.preprocessing import add_daily_returns, add_moving_averages
from utils.visualization import (
    plot_candlestick,
    plot_closing_price,
    plot_daily_returns,
    plot_feature_importance,
    plot_moving_averages,
    plot_rsi,
    plot_sma_ema,
    plot_volatility,
    plot_volume,
)


# ════════════════════════════════════════════════════════════════════
#  Page Configuration
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FinSight AI — Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ════════════════════════════════════════════════════════════════════

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Load custom CSS (base styles)
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject theme-specific overrides
is_light = st.session_state.theme == "light"
if is_light:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%) !important;
            border-right: 1px solid #cbd5e1 !important;
        }
        h1 {
            background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        h2 { color: #1e293b !important; border-bottom-color: #e2e8f0 !important; }
        h3 { color: #1e293b !important; }
        p, li { color: #475569 !important; }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.8) !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        }
        [data-testid="stMetric"]:hover {
            box-shadow: 0 4px 20px rgba(59,130,246,0.12) !important;
        }
        [data-testid="stMetric"] label {
            color: #64748b !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(241,245,249,0.9) !important;
            border: 1px solid #e2e8f0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #475569 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #3b82f6 !important;
            color: white !important;
        }
        [data-baseweb="select"] > div {
            background: white !important;
            border: 1px solid #cbd5e1 !important;
            color: #1e293b !important;
        }
        .stDateInput > div > div {
            background: white !important;
            border: 1px solid #cbd5e1 !important;
        }
        .stMultiSelect > div > div {
            background: white !important;
            border: 1px solid #cbd5e1 !important;
        }
        [data-testid="stExpander"] {
            background: rgba(241,245,249,0.7) !important;
            border: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] h2 {
            color: #475569 !important;
        }
        [data-testid="stSidebar"] .stCaption p {
            color: #64748b !important;
        }
        hr { border-color: #e2e8f0 !important; }
        ::-webkit-scrollbar-thumb {
            background: rgba(0,0,0,0.15) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 FinSight AI")
    st.caption("Financial Market Analytics & Prediction")
    st.markdown("---")

    # ── Theme Toggle ──
    st.markdown("## Appearance")
    theme_label = "🌙 Dark Mode" if is_light else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True, key="theme_toggle"):
        st.session_state.theme = "dark" if is_light else "light"
        st.rerun()
    st.markdown("---")

    # ── Stock Selector ──
    st.markdown("## Stock Selection")
    selected_stock = st.selectbox(
        "Choose a company",
        get_stock_list(),
        index=0,
        help="Select a stock to analyze.",
    )
    ticker = get_ticker(selected_stock)

    st.markdown("---")

    # ── Date Range ──
    st.markdown("## Date Range")
    default_start, default_end = get_default_date_range()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start",
            value=default_start,
            max_value=date.today(),
        )
    with col_d2:
        end_date = st.date_input(
            "End",
            value=default_end,
            max_value=date.today(),
        )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.markdown("---")

    # ── Moving Average Selection ──
    st.markdown("## Moving Averages")
    ma_options = st.multiselect(
        "Select MA periods",
        options=[20, 50, 100],
        default=[20, 50],
        help="Choose which moving average periods to display.",
    )

    st.markdown("---")

    # ── Prediction ──
    st.markdown("## ML Prediction")
    if st.button("🤖  Run Prediction", use_container_width=True):
        st.session_state.run_prediction = True

    st.markdown("---")

    # ── Reset ──
    if st.button("🔄  Reset Dashboard", use_container_width=True):
        st.session_state.prediction_result = None
        st.session_state.run_prediction = False
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Plotly")


# ════════════════════════════════════════════════════════════════════
#  Data Loading
# ════════════════════════════════════════════════════════════════════

with st.spinner("Fetching market data …"):
    data, error = load_stock_data(ticker, start_date, end_date)

if error:
    st.error(error, icon="⚠️")
    st.stop()

# Enrich data with moving averages and daily returns
data = add_moving_averages(data, windows=ma_options)
data = add_daily_returns(data)

# Fetch company info (non-blocking — may be empty)
stock_info = get_stock_info(ticker)
company_name = stock_info.get("shortName", selected_stock)


# ════════════════════════════════════════════════════════════════════
#  Header
# ════════════════════════════════════════════════════════════════════

st.markdown(f"# {company_name}")
sub_color = "#94a3b8" if not is_light else "#64748b"
st.markdown(
    f"<span style='color:{sub_color};font-size:0.95rem;'>"
    f"Ticker: <b>{ticker}</b> &nbsp;·&nbsp; "
    f"Period: {start_date.strftime('%b %d, %Y')} — {end_date.strftime('%b %d, %Y')} "
    f"&nbsp;·&nbsp; {len(data)} trading days"
    f"</span>",
    unsafe_allow_html=True,
)
st.markdown("")


# ════════════════════════════════════════════════════════════════════
#  Tab Layout
# ════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "📈 Historical Analysis",
        "📉 Technical Indicators",
        "🤖 ML Prediction",
        "💡 Insights",
    ]
)


# ────────────────────────────────────────────────────────────────────
#  TAB 1 — Dashboard Overview
# ────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## Market Overview")
    st.markdown("")

    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    current_price = float(latest["Close"])
    open_price = float(latest["Open"])
    high_price = float(latest["High"])
    low_price = float(latest["Low"])
    volume = int(latest["Volume"])
    daily_change = current_price - float(prev["Close"])
    pct_change = (daily_change / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0.0

    # KPI row 1
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Price", f"${current_price:,.2f}", f"{daily_change:+,.2f}")
    k2.metric("Open", f"${open_price:,.2f}")
    k3.metric("Day High", f"${high_price:,.2f}")
    k4.metric("Day Low", f"${low_price:,.2f}")

    # KPI row 2
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Volume", f"{volume:,}")
    k6.metric("Daily Change", f"${daily_change:+,.2f}")
    k7.metric("% Change", f"{pct_change:+.2f}%")

    # Market status badge
    trend_color = "#10b981" if daily_change >= 0 else "#ef4444"
    trend_label = "▲ Bullish" if daily_change >= 0 else "▼ Bearish"
    badge_bg = "rgba(255,255,255,0.03)" if not is_light else "rgba(255,255,255,0.8)"
    k8.markdown(
        f"<div style='padding:0.6rem;text-align:center;background:{badge_bg};"
        f"border-radius:12px;'>"
        f"<span style='color:{trend_color};font-size:1.5rem;font-weight:700;'>"
        f"{trend_label}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("### Quick Price Chart")
    st.plotly_chart(
        plot_closing_price(data, ticker),
        use_container_width=True,
        config={"displayModeBar": True},
        theme=None,
        key="dash_closing_price",
    )

    # Download historical data
    st.markdown("")
    csv_data = data.reset_index().to_csv(index=False)
    st.download_button(
        label="📥  Download Historical Data (CSV)",
        data=csv_data,
        file_name=f"{ticker}_historical_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_historical",
    )


# ────────────────────────────────────────────────────────────────────
#  TAB 2 — Historical Analysis
# ────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("## Historical Analysis")
    st.markdown("")

    # Closing price
    st.plotly_chart(
        plot_closing_price(data, ticker),
        use_container_width=True,
        theme=None,
        key="hist_closing_price",
    )

    # Volume
    st.plotly_chart(
        plot_volume(data, ticker),
        use_container_width=True,
        theme=None,
        key="hist_volume",
    )

    # Candlestick
    st.plotly_chart(
        plot_candlestick(data, ticker),
        use_container_width=True,
        theme=None,
        key="hist_candlestick",
    )

    # Moving Averages
    if ma_options:
        st.plotly_chart(
            plot_moving_averages(data, ticker, ma_options),
            use_container_width=True,
            theme=None,
            key="hist_moving_averages",
        )
    else:
        st.info("Select moving average periods from the sidebar to display them.")

    # Daily Returns
    st.plotly_chart(
        plot_daily_returns(data, ticker),
        use_container_width=True,
        theme=None,
        key="hist_daily_returns",
    )


# ────────────────────────────────────────────────────────────────────
#  TAB 3 — Technical Indicators
# ────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## Technical Indicators")
    st.markdown("")

    # ── SMA vs EMA ──
    st.markdown("### Simple Moving Average (SMA) vs Exponential Moving Average (EMA)")
    indicator_period = st.slider(
        "Select period", min_value=5, max_value=100, value=20, step=5, key="ind_period"
    )
    st.plotly_chart(
        plot_sma_ema(data, ticker, indicator_period),
        use_container_width=True,
        theme=None,
        key="ind_sma_ema",
    )
    with st.expander("ℹ️ About SMA & EMA"):
        st.markdown(
            "**SMA** (Simple Moving Average) calculates the average closing price "
            "over a specified number of days. It treats all data points equally.\n\n"
            "**EMA** (Exponential Moving Average) places greater weight on recent "
            "prices, making it more responsive to new information. Traders often "
            "compare SMA and EMA to identify trend changes — when the EMA crosses "
            "above the SMA, it may signal a bullish trend."
        )

    st.markdown("---")

    # ── RSI ──
    st.markdown("### Relative Strength Index (RSI)")
    st.plotly_chart(
        plot_rsi(data, ticker),
        use_container_width=True,
        theme=None,
        key="ind_rsi",
    )
    with st.expander("ℹ️ About RSI"):
        st.markdown(
            "**RSI** measures the speed and magnitude of recent price changes on "
            "a scale of 0 to 100.\n\n"
            "- **Above 70**: The stock may be *overbought* (potentially overvalued).\n"
            "- **Below 30**: The stock may be *oversold* (potentially undervalued).\n"
            "- **Between 30–70**: Considered neutral territory.\n\n"
            "RSI is most useful when combined with other indicators rather than "
            "used in isolation."
        )

    st.markdown("---")

    # ── Daily Returns ──
    st.markdown("### Daily Returns")
    returns = calculate_daily_returns(data["Close"])
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Mean Return", f"{returns.mean():.3f}%")
    col_r2.metric("Max Return", f"{returns.max():.2f}%")
    col_r3.metric("Min Return", f"{returns.min():.2f}%")
    st.plotly_chart(
        plot_daily_returns(data, ticker),
        use_container_width=True,
        theme=None,
        key="ind_daily_returns",
    )
    with st.expander("ℹ️ About Daily Returns"):
        st.markdown(
            "**Daily Returns** represent the percentage change in closing price "
            "from one trading day to the next. Analyzing the distribution helps "
            "assess the stock's risk profile — a wider distribution indicates "
            "higher volatility."
        )

    st.markdown("---")

    # ── Volatility ──
    st.markdown("### Volatility")
    vol_window = st.slider(
        "Rolling window (days)", min_value=5, max_value=60, value=20, step=5, key="vol_window"
    )
    st.plotly_chart(
        plot_volatility(data, ticker, vol_window),
        use_container_width=True,
        theme=None,
        key="ind_volatility",
    )
    with st.expander("ℹ️ About Volatility"):
        st.markdown(
            "**Volatility** measures the degree of variation in a stock's price "
            "over time, expressed as annualized standard deviation of daily "
            "returns.\n\n"
            "- **High volatility** → greater uncertainty and risk, but also "
            "potential for higher returns.\n"
            "- **Low volatility** → more stable and predictable price behaviour."
        )


# ────────────────────────────────────────────────────────────────────
#  TAB 4 — Machine Learning Prediction
# ────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("## ML-Based Price Prediction")
    st.markdown(
        "Use a **Random Forest Regressor** trained on engineered features to "
        "predict the next trading day's closing price."
    )
    st.markdown("")

    # Disclaimer
    st.warning(
        "⚠️ **Disclaimer:** This prediction is for *educational purposes only* "
        "and does **not** constitute financial advice. Stock markets are "
        "inherently unpredictable. Never make investment decisions based solely "
        "on model outputs.",
        icon="⚠️",
    )
    st.markdown("")

    # Run prediction if button was clicked
    if st.session_state.run_prediction:
        with st.spinner("Training model — this may take a moment …"):
            result, pred_error = train_model(data)

        if pred_error:
            st.error(pred_error, icon="❌")
            st.session_state.run_prediction = False
        else:
            st.session_state.prediction_result = result
            st.session_state.run_prediction = False

    # Display results
    result = st.session_state.prediction_result

    if result is not None:
        st.markdown("### 🎯 Prediction Result")
        st.markdown("")

        pc1, pc2 = st.columns([1, 2])

        with pc1:
            # Predicted price card
            current = float(data.iloc[-1]["Close"])
            predicted = result.predicted_price
            change = predicted - current
            change_pct = (change / current) * 100

            pred_bg = "rgba(59,130,246,0.08)" if not is_light else "rgba(59,130,246,0.06)"
            pred_border = "rgba(59,130,246,0.3)" if not is_light else "rgba(59,130,246,0.2)"
            pred_label_color = "#94a3b8" if not is_light else "#64748b"
            pred_value_color = "#f1f5f9" if not is_light else "#0f172a"
            st.markdown(
                f"<div style='"
                f"background:{pred_bg};border:1px solid {pred_border};"
                f"border-radius:16px;padding:2rem;text-align:center;'>"
                f"<p style='color:{pred_label_color};font-size:0.85rem;margin-bottom:0.25rem;"
                f"text-transform:uppercase;letter-spacing:0.08em;'>Predicted Next Close</p>"
                f"<p style='color:{pred_value_color};font-size:2.5rem;font-weight:800;"
                f"margin:0.5rem 0;'>${predicted:,.2f}</p>"
                f"<p style='color:{'#10b981' if change >= 0 else '#ef4444'};"
                f"font-size:1.1rem;font-weight:600;'>"
                f"{'▲' if change >= 0 else '▼'} ${abs(change):,.2f} "
                f"({change_pct:+.2f}%)</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with pc2:
            st.markdown("#### Model Performance Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"${result.mae:,.2f}")
            m2.metric("RMSE", f"${result.rmse:,.2f}")
            m3.metric("R² Score", f"{result.r2:.4f}")

            m4, m5 = st.columns(2)
            m4.metric("Training Samples", f"{result.train_size:,}")
            m5.metric("Test Samples", f"{result.test_size:,}")

        st.markdown("")

        # Feature importance chart
        st.markdown("### Feature Importance")
        st.plotly_chart(
            plot_feature_importance(
                result.feature_names, result.feature_importances
            ),
            use_container_width=True,
            theme=None,
            key="ml_feature_importance",
        )

        # Download prediction results
        pred_df = pd.DataFrame(
            {
                "Metric": ["Predicted Price", "Current Price", "MAE", "RMSE", "R² Score"],
                "Value": [
                    f"${predicted:,.2f}",
                    f"${current:,.2f}",
                    f"${result.mae:,.2f}",
                    f"${result.rmse:,.2f}",
                    f"{result.r2:.4f}",
                ],
            }
        )
        csv_pred = pred_df.to_csv(index=False)
        st.download_button(
            label="📥  Download Prediction Results (CSV)",
            data=csv_pred,
            file_name=f"{ticker}_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_prediction",
        )
    else:
        # Prompt to run prediction
        st.markdown("")
        st.info(
            "Click **🤖 Run Prediction** in the sidebar to train the model "
            "and generate a price prediction.",
            icon="💡",
        )


# ────────────────────────────────────────────────────────────────────
#  TAB 5 — Insights
# ────────────────────────────────────────────────────────────────────

with tab5:
    st.markdown("## Market Insights")
    st.markdown(
        "Automatically generated insights based on the latest market data and "
        "technical indicators."
    )
    st.markdown("")

    latest_close = float(data.iloc[-1]["Close"])
    prev_close = float(data.iloc[-2]["Close"]) if len(data) > 1 else latest_close

    # ── Helper: styled insight card ──
    def insight_card(
        icon: str,
        title: str,
        value: str,
        description: str,
        color: str,
    ):
        card_bg = "rgba(255,255,255,0.03)" if not is_light else "rgba(255,255,255,0.85)"
        card_border = "rgba(255,255,255,0.08)" if not is_light else "#e2e8f0"
        card_label = "#94a3b8" if not is_light else "#64748b"
        card_value = "#f1f5f9" if not is_light else "#0f172a"
        card_desc = "#94a3b8" if not is_light else "#475569"
        card_shadow = "0 2px 8px rgba(0,0,0,0.04)" if is_light else "none"
        st.markdown(
            f"""
            <div style="
                background: {card_bg};
                border: 1px solid {card_border};
                border-left: 4px solid {color};
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: {card_shadow};
            ">
                <p style="color:{card_label};font-size:0.78rem;text-transform:uppercase;
                   letter-spacing:0.06em;margin-bottom:0.25rem;">
                    {icon} {title}
                </p>
                <p style="color:{card_value};font-size:1.35rem;font-weight:700;
                   margin:0.25rem 0;">
                    {value}
                </p>
                <p style="color:{card_desc};font-size:0.88rem;line-height:1.6;
                   margin:0;">
                    {description}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ic1, ic2 = st.columns(2)

    with ic1:
        # ── 1. Current Trend ──
        daily_chg = latest_close - prev_close
        pct_chg = (daily_chg / prev_close) * 100 if prev_close != 0 else 0
        if daily_chg >= 0:
            trend_val = f"▲ Upward ({pct_chg:+.2f}%)"
            trend_desc = (
                "The stock closed higher than the previous trading day, "
                "indicating positive momentum."
            )
            trend_color = "#10b981"
        else:
            trend_val = f"▼ Downward ({pct_chg:+.2f}%)"
            trend_desc = (
                "The stock closed lower than the previous trading day, "
                "suggesting selling pressure."
            )
            trend_color = "#ef4444"
        insight_card("📈", "Current Trend", trend_val, trend_desc, trend_color)

        # ── 2. Bullish / Bearish ──
        sma_20 = calculate_sma(data["Close"], 20)
        sma_50 = calculate_sma(data["Close"], 50)
        sma20_val = sma_20.iloc[-1] if not sma_20.empty and pd.notna(sma_20.iloc[-1]) else None
        sma50_val = sma_50.iloc[-1] if not sma_50.empty and pd.notna(sma_50.iloc[-1]) else None

        if sma20_val is not None and sma50_val is not None:
            if sma20_val > sma50_val and latest_close > sma20_val:
                signal = "🟢 Bullish"
                signal_desc = (
                    "The 20-day SMA is above the 50-day SMA and the price is "
                    "above both — a classic bullish signal."
                )
                signal_color = "#10b981"
            elif sma20_val < sma50_val and latest_close < sma20_val:
                signal = "🔴 Bearish"
                signal_desc = (
                    "The 20-day SMA is below the 50-day SMA and the price is "
                    "below both — a bearish signal."
                )
                signal_color = "#ef4444"
            else:
                signal = "🟡 Neutral"
                signal_desc = (
                    "Mixed signals — the price and moving averages are not "
                    "clearly aligned. Consider waiting for confirmation."
                )
                signal_color = "#f59e0b"
        else:
            signal = "⚪ Insufficient Data"
            signal_desc = "Not enough data to determine a signal. Expand the date range."
            signal_color = "#64748b"

        insight_card("🎯", "Market Signal", signal, signal_desc, signal_color)

        # ── 3. Moving Average Signal ──
        if sma20_val is not None:
            if latest_close > sma20_val:
                ma_sig = "Price above SMA-20 ✅"
                ma_desc = (
                    "The current price is trading above the 20-day moving average, "
                    "which is often interpreted as short-term strength."
                )
                ma_color = "#10b981"
            else:
                ma_sig = "Price below SMA-20 ⚠️"
                ma_desc = (
                    "The current price is below the 20-day moving average, "
                    "which may indicate short-term weakness."
                )
                ma_color = "#ef4444"
        else:
            ma_sig = "N/A"
            ma_desc = "Not enough data for the 20-day moving average."
            ma_color = "#64748b"
        insight_card("📊", "Moving Average Signal", ma_sig, ma_desc, ma_color)

    with ic2:
        # ── 4. Volatility ──
        vol_series = calculate_volatility(data["Close"], 20)
        latest_vol = vol_series.iloc[-1] if not vol_series.empty and pd.notna(vol_series.iloc[-1]) else None

        if latest_vol is not None:
            if latest_vol > 40:
                vol_label = f"🔴 High ({latest_vol:.1f}%)"
                vol_desc = (
                    "Annualized volatility is elevated, indicating significant "
                    "price swings. Higher risk, but also potential for larger returns."
                )
                vol_color = "#ef4444"
            elif latest_vol > 20:
                vol_label = f"🟡 Moderate ({latest_vol:.1f}%)"
                vol_desc = (
                    "Volatility is within a normal range. The stock is "
                    "experiencing typical market fluctuations."
                )
                vol_color = "#f59e0b"
            else:
                vol_label = f"🟢 Low ({latest_vol:.1f}%)"
                vol_desc = (
                    "The stock is showing low volatility, suggesting relatively "
                    "stable and predictable price behaviour."
                )
                vol_color = "#10b981"
        else:
            vol_label = "N/A"
            vol_desc = "Not enough data to calculate volatility."
            vol_color = "#64748b"
        insight_card("📉", "Volatility Level", vol_label, vol_desc, vol_color)

        # ── 5. Investment Risk Level ──
        rsi_series = calculate_rsi(data["Close"], 14)
        latest_rsi = rsi_series.iloc[-1] if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]) else None

        if latest_vol is not None and latest_rsi is not None:
            risk_score = 0
            if latest_vol > 40:
                risk_score += 2
            elif latest_vol > 20:
                risk_score += 1
            if latest_rsi > 70 or latest_rsi < 30:
                risk_score += 1
            if daily_chg < 0:
                risk_score += 1

            if risk_score >= 3:
                risk_label = "🔴 High Risk"
                risk_desc = (
                    "Multiple risk indicators are elevated. Exercise caution — "
                    "high volatility, extreme RSI, or recent negative momentum."
                )
                risk_color = "#ef4444"
            elif risk_score >= 1:
                risk_label = "🟡 Moderate Risk"
                risk_desc = (
                    "Some risk factors are present but not alarming. "
                    "Standard market conditions with moderate caution advised."
                )
                risk_color = "#f59e0b"
            else:
                risk_label = "🟢 Low Risk"
                risk_desc = (
                    "Risk indicators are favorable — low volatility, "
                    "balanced RSI, and positive momentum."
                )
                risk_color = "#10b981"
        else:
            risk_label = "N/A"
            risk_desc = "Not enough data to assess risk."
            risk_color = "#64748b"
        insight_card("🛡️", "Investment Risk Level", risk_label, risk_desc, risk_color)

        # ── 6. RSI Status ──
        if latest_rsi is not None:
            if latest_rsi > 70:
                rsi_label = f"Overbought ({latest_rsi:.1f})"
                rsi_desc = (
                    "RSI is above 70, suggesting the stock may be overbought. "
                    "A price correction could follow."
                )
                rsi_color = "#ef4444"
            elif latest_rsi < 30:
                rsi_label = f"Oversold ({latest_rsi:.1f})"
                rsi_desc = (
                    "RSI is below 30, indicating the stock may be oversold. "
                    "A price rebound could be possible."
                )
                rsi_color = "#10b981"
            else:
                rsi_label = f"Neutral ({latest_rsi:.1f})"
                rsi_desc = (
                    "RSI is in the neutral range (30–70), indicating balanced "
                    "buying and selling pressure."
                )
                rsi_color = "#3b82f6"
        else:
            rsi_label = "N/A"
            rsi_desc = "Not enough data to calculate RSI."
            rsi_color = "#64748b"
        insight_card("⚡", "RSI Status", rsi_label, rsi_desc, rsi_color)

    st.markdown("")
    st.warning(
        "These insights are generated automatically based on technical "
        "indicators and should not be treated as financial advice.",
        icon="⚠️",
    )
