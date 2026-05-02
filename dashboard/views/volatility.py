"""
Volatility view — multi-lookback volatility, target vs actual, and volatility cones.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

TARGET_VOL = 25.0  # Annualized volatility target (%)


def render(metrics: dict):
    st.header("Volatility Analysis")

    vol_data = metrics.get("volatility")
    if vol_data is None or not vol_data:
        st.warning("No volatility data available.")
        return

    vol_lookbacks = metrics.get("vol_lookbacks", [21])
    vol_cones = metrics.get("volatility_cones", {})

    selected_lookback = st.selectbox("Volatility Lookback (days)", vol_lookbacks, index=vol_lookbacks.index(21) if 21 in vol_lookbacks else 0)

    selected_vol = vol_data.get(selected_lookback)
    if selected_vol is not None and not selected_vol.empty:
        _plot_volatility(selected_vol, selected_lookback)
        _volatility_summary(selected_vol)

    if vol_cones:
        _plot_volatility_cones(vol_cones)


def _plot_volatility(vol_data: pd.DataFrame, lookback: int):
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

    # Target volatility line
    fig.add_hline(y=TARGET_VOL, line_dash="dash", line_color="red", opacity=0.7, annotation_text=f"Target ({TARGET_VOL}%)", annotation_position="top right")

    fig.update_layout(title=f"Portfolio and Instrument Annualized Volatility ({lookback}-day rolling)", yaxis_title="Annualized Volatility (%)", xaxis_title="Date", showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=500)
    st.plotly_chart(fig, use_container_width=True)


def _volatility_summary(vol_data: pd.DataFrame):
    st.subheader("Annualized Volatility Summary")
    summary = vol_data.dropna().tail(1).T
    summary.columns = ["Current Vol (%)"]
    summary["Target Vol (%)"] = TARGET_VOL
    summary["Deviation (%)"] = summary["Current Vol (%)"] - TARGET_VOL
    st.dataframe(summary.style.format({"Current Vol (%)": "{:.2f}", "Target Vol (%)": "{:.1f}", "Deviation (%)": "{:+.2f}"}), use_container_width=True)


def _plot_volatility_cones(vol_cones: dict):
    st.subheader("Realized Volatility Cones")

    horizons = sorted(vol_cones.keys())
    all_instruments = set()
    for h_data in vol_cones.values():
        all_instruments.update(h_data.keys())
    all_instruments = sorted(all_instruments)

    selected_instr = st.selectbox("Select Instrument", all_instruments, index=0 if "portfolio" not in all_instruments else all_instruments.index("portfolio"))

    fig = go.Figure()
    color_map = {"portfolio": "#000000"}
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color = color_map.get(selected_instr, default_colors[all_instruments.index(selected_instr) % len(default_colors)])

    x_labels = [str(h) for h in horizons]
    p5_vals, p25_vals, p50_vals, p75_vals, p95_vals, current_vals = [], [], [], [], [], []
    x_current = []

    for h in horizons:
        data = vol_cones[h].get(selected_instr)
        if data:
            p5_vals.append(data.get("p5", 0))
            p25_vals.append(data.get("p25", 0))
            p50_vals.append(data.get("p50", 0))
            p75_vals.append(data.get("p75", 0))
            p95_vals.append(data.get("p95", 0))
            if data.get("current") is not None:
                current_vals.append(data["current"])
                x_current.append(str(h))

    def hex_to_rgba(hex_color, alpha):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Outer band (p5-p95)
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=p5_vals + p95_vals[::-1],
        fill="toself", fillcolor=hex_to_rgba(color, 0.15),
        line=dict(color=color, width=0),
        name=f"{selected_instr} (p5-p95)", showlegend=True,
        hoverinfo="skip"
    ))

    # Inner band (p25-p75)
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=p25_vals + p75_vals[::-1],
        fill="toself", fillcolor=hex_to_rgba(color, 0.3),
        line=dict(color=color, width=0),
        name=f"{selected_instr} (p25-p75)", showlegend=False,
        hoverinfo="skip"
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=x_labels, y=p50_vals,
        line=dict(color=color, width=2),
        name=f"{selected_instr} Median", showlegend=True
    ))

    # Current vol markers
    if current_vals:
        fig.add_trace(go.Scatter(
            x=x_current, y=current_vals,
            mode="markers", marker=dict(color=color, size=8, symbol="diamond"),
            name=f"{selected_instr} (current)", showlegend=True
        ))

    fig.update_layout(
        title=f"Volatility Cone: {selected_instr}",
        xaxis_title="Lookback Horizon (days)",
        yaxis_title="Annualized Volatility (%)",
        height=450,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
