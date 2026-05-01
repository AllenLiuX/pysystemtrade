"""
Instruments view — per-instrument performance and price charts.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render(metrics: dict):
    st.header("Instrument Details")

    instr_metrics = metrics.get("instruments", {})
    if not instr_metrics:
        st.warning("No instrument data available.")
        return

    _instrument_table(instr_metrics)
    _price_chart(metrics)


def _instrument_table(instr_metrics: dict):
    st.subheader("Per-Instrument Performance")
    rows = []
    for instr, m in instr_metrics.items():
        row = {
            "Instrument": instr,
            "Total Return (%)": m.get("total_return", 0),
            "Ann. Return (%)": m.get("annualized_return", 0),
            "Ann. Vol (%)": m.get("annualized_vol", 0),
            "Sharpe": m.get("sharpe", 0),
            "Max Drawdown (%)": m.get("max_drawdown", 0),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include="number").columns
    st.dataframe(df.style.format({c: "{:.2f}" for c in numeric_cols}), use_container_width=True, hide_index=True)


def _price_chart(metrics: dict):
    st.subheader("Per-Instrument Price Chart")

    raw_prices = metrics.get("raw_prices", {})
    volume_data = metrics.get("volume_data", {})

    if not raw_prices:
        equity_curve = metrics.get("equity_curve")
        if equity_curve is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Portfolio Value", line=dict(color="#1f77b4", width=2)))
            fig.update_layout(title="Portfolio Value Over Time", yaxis_title="Value (CNY)", xaxis_title="Date", height=400)
            st.plotly_chart(fig, use_container_width=True)
        return

    instrument = st.selectbox("Select instrument", list(raw_prices.keys()))

    prices = raw_prices.get(instrument)
    vol = volume_data.get(instrument)

    if prices is not None and not prices.empty:
        if vol is not None and not vol.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name=instrument, line=dict(color="#1f77b4", width=2)), row=1, col=1)
            fig.add_trace(go.Bar(x=vol.index, y=vol.values, name="Volume", marker_color="#636efa", opacity=0.6), row=2, col=1)
            fig.update_layout(title=f"{instrument} Price & Volume", yaxis_title="Price (CNY)", height=500)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name=instrument, line=dict(color="#1f77b4", width=2)))
            fig.update_layout(title=f"{instrument} Price", yaxis_title="Price (CNY)", xaxis_title="Date", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No price data for {instrument}")
