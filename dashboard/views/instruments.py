"""
Instruments view — per-instrument performance and price charts.
"""

import streamlit as st
import plotly.graph_objects as go
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
    st.dataframe(df.style.format({c: "{:.2f}" for c in numeric_cols}), use_container_width=True)


def _price_chart(metrics: dict):
    st.subheader("Price Chart")
    equity_curve = metrics.get("equity_curve")
    if equity_curve is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Portfolio Value", line=dict(color="#1f77b4", width=2)))
    fig.update_layout(title="Portfolio Value Over Time", yaxis_title="Value (CNY)", xaxis_title="Date", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: Per-instrument price charts can be added in a future update.")
