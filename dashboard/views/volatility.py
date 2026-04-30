"""
Volatility view — portfolio and instrument rolling volatility.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render(metrics: dict):
    st.header("Volatility Analysis")

    vol_data = metrics.get("volatility")
    if vol_data is None or vol_data.empty:
        st.warning("No volatility data available.")
        return

    _plot_volatility(vol_data)

    st.subheader("Annualized Volatility Summary")
    summary = vol_data.dropna().tail(1).T
    summary.columns = ["Current Vol (%)"]
    st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)


def _plot_volatility(vol_data: pd.DataFrame):
    fig = go.Figure()
    colors = {"portfolio": ("black", 3)}
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_idx = 0

    for col in vol_data.columns:
        if col == "portfolio":
            color, width = colors["portfolio"]
        else:
            color = default_colors[color_idx % len(default_colors)]
            width = 1.5
            color_idx += 1
        fig.add_trace(go.Scatter(x=vol_data.index, y=vol_data[col], name=col, line=dict(color=color, width=width)))

    fig.update_layout(title="Portfolio and Instrument Annualized Volatility (21-day rolling)", yaxis_title="Annualized Volatility (%)", xaxis_title="Date", showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=500)
    st.plotly_chart(fig, use_container_width=True)
