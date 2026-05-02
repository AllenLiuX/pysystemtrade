"""
Overview view — KPI cards, equity curve, drawdown, strategy comparison.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def render(metrics: dict):
    st.header("Strategy Overview")

    equity_curve = metrics.get("equity_curve")
    if equity_curve is None or equity_curve.empty:
        st.warning("No equity curve data available.")
        return

    perf = metrics.get("performance", {})
    equal_perf = metrics.get("equal_performance", {})
    equal_equity = metrics.get("equal_weight_equity")
    returns_data = metrics.get("returns", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Value", f"{equity_curve.iloc[-1]:,.0f} CNY")
    col2.metric("Sharpe Ratio", f"{perf.get('sharpe', 0):.2f}")
    col3.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.2f}%")
    col4.metric("Annualized Return", f"{perf.get('annualized_return', 0):.2f}%")

    _plot_equity_and_drawdown(equity_curve, equal_equity)
    _comparison_table(perf, equal_perf)

    if returns_data:
        _return_distributions(returns_data)


def _plot_equity_and_drawdown(equity_curve, equal_equity=None):
    returns = equity_curve.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Risk Parity", line=dict(color="#1f77b4", width=2)), row=1, col=1)

    capital = equity_curve.iloc[0]
    fig.add_hline(y=capital, line_dash="dot", line_color="gray", opacity=0.5, annotation_text="Initial Capital", row=1, col=1)

    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy", fillcolor="rgba(255,0,0,0.2)", line=dict(color="#1f77b4", width=1), name="RP Drawdown", legendgroup="rp"), row=2, col=1)

    if equal_equity is not None:
        fig.add_trace(go.Scatter(x=equal_equity.index, y=equal_equity.values, name="Equal Weight", line=dict(color="#2ca02c", width=2, dash="dash")), row=1, col=1)

        ew_returns = equal_equity.pct_change().dropna()
        ew_cumulative = (1 + ew_returns).cumprod()
        ew_running_max = ew_cumulative.cummax()
        ew_drawdown = (ew_cumulative - ew_running_max) / ew_running_max * 100
        fig.add_trace(go.Scatter(x=ew_drawdown.index, y=ew_drawdown.values, fill="tozeroy", fillcolor="rgba(0,255,0,0.2)", line=dict(color="#2ca02c", width=1), name="EW Drawdown", legendgroup="ew"), row=2, col=1)

    fig.update_layout(height=600, title="Equity Curve & Drawdown", showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_yaxes(title_text="Portfolio Value (CNY)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _comparison_table(perf: dict, equal_perf: dict):
    st.subheader("Strategy Comparison")
    comparison = pd.DataFrame({
        "Risk Parity": [perf.get("total_return", 0), perf.get("annualized_return", 0), perf.get("annualized_vol", 0), perf.get("sharpe", 0), perf.get("max_drawdown", 0)],
        "Equal Weight": [equal_perf.get("total_return", 0), equal_perf.get("annualized_return", 0), equal_perf.get("annualized_vol", 0), equal_perf.get("sharpe", 0), equal_perf.get("max_drawdown", 0)],
    }, index=["Total Return (%)", "Ann. Return (%)", "Ann. Vol (%)", "Sharpe", "Max Drawdown (%)"])
    st.dataframe(comparison.style.format("{:.2f}"), use_container_width=True)


def _return_distributions(returns_data: dict):
    st.subheader("Return Distributions")

    selected = st.selectbox("Select series", list(returns_data.keys()), index=0)
    returns = returns_data.get(selected)
    if returns is None or returns.empty:
        st.warning("No return data available.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        _plot_return_histogram(returns, selected)

    with col2:
        _return_stats(returns)


def _plot_return_histogram(returns: pd.Series, name: str):
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=returns.values, nbinsx=50, name=name, marker_color="#1f77b4", opacity=0.7, histnorm="probability density"))

    sorted_vals = np.sort(returns.values)
    kde_x = np.linspace(sorted_vals.min(), sorted_vals.max(), 200)
    bandwidth = returns.std() * 1.06 * len(returns) ** (-0.2)
    kde_y = np.zeros_like(kde_x)
    for x in sorted_vals:
        kde_y += np.exp(-0.5 * ((kde_x - x) / bandwidth) ** 2)
    kde_y /= len(sorted_vals) * bandwidth * np.sqrt(2 * np.pi)

    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines", name="KDE", line=dict(color="#ff7f0e", width=2)))

    mean = returns.mean()
    std = returns.std()
    normal_y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((kde_x - mean) / std) ** 2)
    fig.add_trace(go.Scatter(x=kde_x, y=normal_y, mode="lines", name="Normal", line=dict(color="red", width=1.5, dash="dash")))

    fig.add_vline(x=mean, line_dash="dot", line_color="red", annotation_text="Mean", annotation_position="top right")
    fig.add_vline(x=returns.median(), line_dash="solid", line_color="black", annotation_text="Median", annotation_position="top left")

    fig.update_layout(
        title=f"Return Distribution: {name}",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        height=350,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)


def _return_stats(returns: pd.Series):
    st.markdown("**Distribution Statistics**")

    mean = returns.mean() * 100
    std = returns.std() * 100
    skew = returns.skew()
    kurt = returns.kurtosis()
    min_ret = returns.min() * 100
    max_ret = returns.max() * 100
    var_95 = returns.quantile(0.05) * 100
    cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100

    stats = pd.DataFrame({
        "Value": [
            f"{mean:.4f}%",
            f"{std:.4f}%",
            f"{skew:.3f}",
            f"{kurt:.3f}",
            f"{min_ret:.4f}%",
            f"{max_ret:.4f}%",
            f"{var_95:.4f}%",
            f"{cvar_95:.4f}%",
        ]
    }, index=["Mean", "Std Dev", "Skewness", "Excess Kurtosis", "Min", "Max", "VaR (95%)", "CVaR (95%)"])

    st.dataframe(stats, use_container_width=True)
