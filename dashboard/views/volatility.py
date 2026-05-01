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
    selected_horizon = st.selectbox("Cone Horizon (days)", horizons, index=len(horizons) - 1)

    cone = vol_cones[selected_horizon]
    if not cone:
        st.warning("No cone data for this horizon.")
        return

    instruments = [k for k in cone.keys() if k != "portfolio"]
    all_names = ["portfolio"] + instruments

    fig = go.Figure()
    colors = {"portfolio": "black"}
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, name in enumerate(all_names):
        data = cone.get(name)
        if not data:
            continue
        color = colors.get(name, default_colors[idx % len(default_colors)])

        # Fan chart: p5-p95 as shaded area, p25-p75 as darker area, p50 as line
        x_labels = ["p5", "p25", "p50", "p75", "p95"]
        y_values = [data.get(x, 0) for x in x_labels]

        # Outer band (p5-p95)
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_values,
            fill="toself", fillcolor=f"{color}20",
            line=dict(color=color, width=0),
            name=f"{name} (p5-p95)", showlegend=(idx == 0),
            hoverinfo="skip"
        ))

        # Inner band (p25-p75)
        fig.add_trace(go.Scatter(
            x=["p25", "p50", "p75"], y=[data.get("p25", 0), data.get("p50", 0), data.get("p75", 0)],
            fill="toself", fillcolor=f"{color}40",
            line=dict(color=color, width=0),
            name=f"{name} (p25-p75)", showlegend=False,
            hoverinfo="skip"
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_values,
            line=dict(color=color, width=2),
            name=name, showlegend=True
        ))

        # Current vol marker
        current = data.get("current")
        if current is not None:
            fig.add_trace(go.Scatter(
                x=["current"], y=[current],
                mode="markers", marker=dict(color=color, size=10, symbol="diamond"),
                name=f"{name} (current)", showlegend=False
            ))

    fig.update_layout(
        title=f"Volatility Cone ({selected_horizon}-day realized vol percentiles)",
        xaxis_title="Percentile",
        yaxis_title="Annualized Volatility (%)",
        height=450,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
