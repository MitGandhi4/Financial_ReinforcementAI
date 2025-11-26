"""Streamlit dashboard for AlphaRL: performance metrics, equity curve,
trade log, and LLM reasoning for each trade.

Run with: streamlit run dashboard/app.py
Expects the FastAPI backend to be running (default http://localhost:8000).
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.logging_config import get_logger

logger = get_logger(__name__)

API_BASE_URL = os.environ.get("ALPHARL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AlphaRL Dashboard", layout="wide")
st.title("AlphaRL — RL Trading Agent Dashboard")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    fetch_clicked = st.button("Load backtest results")
    st.caption(f"API: {API_BASE_URL}")


def _get(path: str) -> dict | list | None:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("API request to %s failed: %s", path, exc)
        st.error(f"Could not reach AlphaRL API at {API_BASE_URL}{path}: {exc}")
        return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("API request to %s failed: %s", path, exc)
        st.error(f"Could not reach AlphaRL API at {API_BASE_URL}{path}: {exc}")
        return None


if fetch_clicked or ticker:
    data = _get(f"/backtest/{ticker}")

    if data:
        col1, col2, col3, col4 = st.columns(4)
        strat = data["strategy"]
        bh = data["buy_and_hold"]
        col1.metric("Strategy Total Return", f"{strat['total_return_pct']:.2f}%",
                    delta=f"{strat['total_return_pct'] - bh['total_return_pct']:.2f}% vs B&H")
        col2.metric("Sharpe Ratio", f"{strat['sharpe_ratio']:.2f}")
        col3.metric("Max Drawdown", f"{strat['max_drawdown_pct']:.2f}%")
        col4.metric("Calmar Ratio", f"{strat['calmar_ratio']:.2f}")

        st.subheader("Equity Curve: Strategy vs. Buy & Hold")
        equity_df = pd.DataFrame(data["equity_curve"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["net_worth"].astype(float),
                                  name="PPO Agent", mode="lines"))
        buy_hold = pd.DataFrame(data["buy_and_hold_curve"])
        if not buy_hold.empty:
            fig.add_trace(go.Scatter(
                x=buy_hold["date"],
                y=buy_hold["net_worth"].astype(float),
                name="Buy & Hold",
                mode="lines",
            ))
        fig.update_layout(
            template="plotly_dark",
            height=460,
            yaxis_title="Portfolio value ($)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(tickprefix="$", separatethousands=True)
        st.plotly_chart(fig, use_container_width=True)
        if data.get("data_source") == "synthetic demonstration":
            st.info("Demonstration uses deterministic synthetic data, not historical market prices.")

        st.subheader("Strategy vs. Buy-and-Hold Metrics")
        metrics_df = pd.DataFrame([strat, bh]).set_index("label")
        st.dataframe(metrics_df)

        st.subheader("Trade Log")
        trade_log = pd.DataFrame(data.get("trade_log", []))
        if trade_log.empty:
            st.info("No trades recorded in this backtest.")
        else:
            st.dataframe(trade_log)

            st.subheader("LLM Trade Reasoning")
            trade_idx = st.number_input(
                "Trade index", min_value=0, max_value=max(len(trade_log) - 1, 0), value=0
            )
            if st.button("Explain this trade"):
                reasoning = _post("/reasoning", {"ticker": ticker, "trade_index": int(trade_idx)})
                if reasoning:
                    st.write(reasoning["explanation"])
    else:
        st.warning(
            f"No backtest results found for {ticker}. Run training + backtest first, "
            "e.g. `python -m src.agent.train --ticker AAPL --timesteps 200000` then "
            "`python -m src.backtest.backtester --ticker AAPL`."
        )
