"""
Weights view — rolling risk parity weights, equal-weight comparison, and vol-scaled positions.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render(metrics: dict):
    st.header("Portfolio Weights")

    weights = metrics.get("weights")
    if weights is None or weights.empty:
        st.warning("No weight data available.")
        return

    ew_weights = metrics.get("equal_weight_weights")

    _plot_rolling_weights(weights, ew_weights)
    _current_weights_bar(weights, ew_weights)

    positions = metrics.get("positions")
    if positions is not None and not positions.empty:
        _positions_table(positions)


def _plot_rolling_weights(weights: pd.DataFrame, ew_weights: pd.DataFrame = None):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Risk Parity Weights", "Equal-Weight Weights"), vertical_spacing=0.12, row_heights=[0.6, 0.4])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], name=col, line=dict(color=colors[i % len(colors)], width=2), stackgroup="one", legendgroup="rp"), row=1, col=1)

    if ew_weights is not None and not ew_weights.empty:
        for i, col in enumerate(ew_weights.columns):
            fig.add_trace(go.Scatter(x=ew_weights.index, y=ew_weights[col], name=col, line=dict(color=colors[i % len(colors)], width=2, dash="dash"), stackgroup="one", showlegend=False, legendgroup="ew"), row=2, col=1)

    fig.update_layout(height=600, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _current_weights_bar(weights: pd.DataFrame, ew_weights: pd.DataFrame = None):
    st.subheader("Current Allocation")

    latest_rp = weights.iloc[-1]
    if ew_weights is not None and not ew_weights.empty:
        latest_ew = ew_weights.iloc[-1]
        all_instruments = list(set(list(latest_rp.index) + list(latest_ew.index)))
        rp_vals = [latest_rp.get(instr, 0) for instr in all_instruments]
        ew_vals = [latest_ew.get(instr, 0) for instr in all_instruments]
        colors_rp = ["rgba(31,119,180,1)", "rgba(255,127,14,1)", "rgba(44,160,44,1)", "rgba(214,39,40,1)", "rgba(148,103,189,1)"]
        colors_ew = ["rgba(31,119,180,0.5)", "rgba(255,127,14,0.5)", "rgba(44,160,44,0.5)", "rgba(214,39,40,0.5)", "rgba(148,103,189,0.5)"]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=all_instruments, y=rp_vals, name="Risk Parity", marker_color=colors_rp[:len(all_instruments)], text=[f"{v:.1%}" for v in rp_vals], textposition="auto"))
        fig.add_trace(go.Bar(x=all_instruments, y=ew_vals, name="Equal Weight", marker_color=colors_ew[:len(all_instruments)], text=[f"{v:.1%}" for v in ew_vals], textposition="auto"))
        fig.update_layout(yaxis_title="Weight", yaxis=dict(range=[0, 1.1]), barmode="group", height=350)
    else:
        fig = go.Figure(go.Bar(x=latest_rp.index, y=latest_rp.values, text=[f"{v:.1%}" for v in latest_rp.values], textposition="auto", marker_color=["rgba(31,119,180,1)", "rgba(255,127,14,1)", "rgba(44,160,44,1)", "rgba(214,39,40,1)", "rgba(148,103,189,1)"]))
        fig.update_layout(yaxis_title="Weight", yaxis=dict(range=[0, 1.1]), height=350)

    st.plotly_chart(fig, use_container_width=True)


def _positions_table(positions: pd.DataFrame):
    st.subheader("Vol-Scaled Positions")

    df = positions.dropna(how="all").copy()
    if df.empty:
        st.warning("No position data available.")
        return

    df.index = df.index.date

    abs_positions = df.abs()
    total_abs = abs_positions.sum(axis=1)
    weight_pct = df.div(total_abs, axis=0) * 100

    for col in df.columns:
        df[f"{col} Weight (%)"] = weight_pct[col].round(2)

    df["Total Position"] = df[df.columns[:len(positions.columns)]].sum(axis=1).round(4)
    df["Total Weight (%)"] = weight_pct.sum(axis=1).round(2)

    fmt = {c: "{:.4f}" for c in df.columns[:len(positions.columns)]}
    fmt.update({c: "{:.2f}" for c in df.columns[len(positions.columns):]})

    st.dataframe(df.style.format(fmt), use_container_width=True)
