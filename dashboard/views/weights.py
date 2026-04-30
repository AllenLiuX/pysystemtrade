"""
Weights view — rolling risk parity weights and vol-scaled positions.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render(metrics: dict):
    st.header("Portfolio Weights")

    weights = metrics.get("weights")
    if weights is None or weights.empty:
        st.warning("No weight data available.")
        return

    _plot_rolling_weights(weights)
    _current_weights_bar(weights)

    positions = metrics.get("positions")
    if positions is not None and not positions.empty:
        st.subheader("Vol-Scaled Positions (Latest)")
        st.dataframe(positions.tail().style.format("{:.4f}"), use_container_width=True)


def _plot_rolling_weights(weights: pd.DataFrame):
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], name=col, line=dict(color=colors[i % len(colors)], width=2), stackgroup="one"))

    fig.update_layout(title="Rolling Risk Parity Weights Over Time", yaxis_title="Weight", xaxis_title="Date", yaxis=dict(range=[0, 1.05]), showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=400)
    st.plotly_chart(fig, use_container_width=True)


def _current_weights_bar(weights: pd.DataFrame):
    st.subheader("Current Allocation")
    latest = weights.iloc[-1]
    fig = go.Figure(go.Bar(x=latest.index, y=latest.values, text=[f"{v:.1%}" for v in latest.values], textposition="auto", marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
    fig.update_layout(title="Latest Risk Parity Weights", yaxis_title="Weight", yaxis=dict(range=[0, 1.1]), height=350)
    st.plotly_chart(fig, use_container_width=True)
