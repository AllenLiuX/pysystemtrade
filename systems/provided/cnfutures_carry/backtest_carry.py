# %% [markdown]
# # Chinese Futures Carry Backtest
#
# Backtests the pysystemtrade carry trading rule on all 82 Chinese futures
# continuous contracts. Uses the full System pipeline with data from Supabase.
#
# Carry = annualised roll (front month − next month) / volatility, smoothed
# with a 90-day EWMA. Positive forecast = backwardation (go long), negative
# = contango (go short).

# %% [markdown]
# ## 1. Imports and Setup

# %%
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass

from sysdata.config.configdata import Config

from systems.basesystem import System
from systems.rawdata import RawData
from systems.trading_rules import TradingRule
from systems.forecasting import Rules
from systems.forecast_scale_cap import ForecastScaleCap
from systems.forecast_combine import ForecastCombine
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.accounts.accounts_stage import Account

from systems.provided.cnfutures_carry.cnfutures_data import (
    CnFuturesSimData,
    ALL_INSTRUMENTS,
    _CNFUTURES_CONFIG,
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %% [markdown]
# ## 2. Load Data

# %%
data = CnFuturesSimData()
instruments = ALL_INSTRUMENTS
print(f"Loaded {len(instruments)} instruments")
print(f"Examples: {instruments[:5]} ... {instruments[-3:]}")

# %% [markdown]
# ## 3. Configure System

# %%
config = Config()
config.instruments = instruments
config.notional_trading_capital = 10_000_000
config.percentage_vol_target = 25.0
config.base_currency = "CNY"

# Equal weight all instruments (use those with sufficient data)
config.instrument_weights = {sym: 1.0 / len(instruments) for sym in instruments}

# Carry rule with 90-day smooth
config.trading_rules = {
    "carry90": {
        "function": "systems.provided.rules.carry.carry",
        "data": ["rawdata.raw_carry"],
        "other_args": {"smooth_days": 90},
    },
}

# Default forecast scalars (will be estimated if not set)
config.forecast_weights = {"carry90": 1.0}
config.use_forecast_scale_estimates = True
config.forecast_div_multiplier = 1.0

print(f"Config: {len(instruments)} instruments, carry90 rule, {config.notional_trading_capital:,.0f} capital")

# %% [markdown]
# ## 4. Build System

# %%
# Create rules stage explicitly
rules_stage = Rules()

stages = [
    Account(),
    Portfolios(),
    PositionSizing(),
    ForecastCombine(),
    ForecastScaleCap(),
    rules_stage,
    RawData(),
]

system = System(stages, data=data, config=config)
print(f"System built: {system}")

# %% [markdown]
# ## 5. Backtest Results — Equity Curve

# %%
# Portfolio-level equity
portfolio = system.accounts.portfolio()
equity = portfolio.curve() + config.notional_trading_capital
equity_pct = (equity / equity.iloc[0] - 1) * 100  # percentage returns

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(equity.index, equity.values, linewidth=1, color="#2c3e50")
ax1.set_title("Chinese Futures Carry Strategy — Portfolio Equity", fontsize=14, fontweight="bold")
ax1.set_ylabel("Equity (CNY)", fontsize=11)
ax1.grid(True, alpha=0.3)

# Drawdown
running_max = equity.cummax()
drawdown = (equity / running_max - 1) * 100
ax2.fill_between(drawdown.index, 0, drawdown.values, color="#e74c3c", alpha=0.5)
ax2.set_ylabel("Drawdown %", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.grid(True, alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Performance Metrics

# %%
returns = equity.pct_change().dropna()
years = (equity.index[-1] - equity.index[0]).days / 365.25

total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
ann_vol = returns.std() * np.sqrt(252) * 100
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

downside = returns[returns < 0].std() * np.sqrt(252) * 100
sortino = ann_return / downside if downside > 0 else 0

max_dd = drawdown.min()

# Calmar
calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

print(f"{'Metric':30s} {'Value':>10s}")
print("-" * 42)
print(f"{'Total Return':30s} {total_return:10.2f} %")
print(f"{'Annual Return':30s} {ann_return:10.2f} %")
print(f"{'Annual Volatility':30s} {ann_vol:10.2f} %")
print(f"{'Sharpe Ratio':30s} {sharpe:10.3f}")
print(f"{'Sortino Ratio':30s} {sortino:10.3f}")
print(f"{'Calmar Ratio':30s} {calmar:10.3f}")
print(f"{'Max Drawdown':30s} {max_dd:10.2f} %")
print(f"{'Backtest Years':30s} {years:10.2f}")

# %% [markdown]
# ## 7. Per-Instrument Carry Analysis

# %%
# Show the latest raw carry values for all instruments
latest_carry = {}
for sym in instruments:
    try:
        rc = system.rawdata.raw_carry(sym)
        if not rc.empty:
            latest_carry[sym] = rc.iloc[-1]
    except Exception:
        pass

carry_df = pd.Series(latest_carry).sort_values()
top_n = 15

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Top 15 by carry
top = carry_df.tail(top_n)
ax = axes[0]
colors = ["#27ae60" if v > 0 else "#e74c3c" for v in top.values]
ax.barh(range(len(top)), top.values, color=colors, edgecolor="white")
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_title(f"Top {top_n} Carry (Backwardation → Long)", fontsize=12, fontweight="bold")
ax.set_xlabel("Raw Carry (annualised roll / vol)")

# Bottom 15 by carry
bottom = carry_df.head(top_n)
ax = axes[1]
colors = ["#27ae60" if v > 0 else "#e74c3c" for v in bottom.values]
ax.barh(range(len(bottom)), bottom.values, color=colors, edgecolor="white")
ax.set_yticks(range(len(bottom)))
ax.set_yticklabels(bottom.index)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_title(f"Bottom {top_n} Carry (Contango → Short)", fontsize=12, fontweight="bold")
ax.set_xlabel("Raw Carry (annualised roll / vol)")

plt.tight_layout()
plt.show()

# Summary stats
n_positive = (carry_df > 0).sum()
n_negative = (carry_df < 0).sum()
print(f"\nCarry Distribution: {n_positive} positive (backwardation), "
      f"{n_negative} negative (contango), {len(carry_df)} total")
print(f"Mean carry: {carry_df.mean():.4f}, Median: {carry_df.median():.4f}, "
      f"Std: {carry_df.std():.4f}")

# %% [markdown]
# ## 8. Forecast Distribution

# %%
all_forecasts = []
for sym in instruments:
    try:
        fc = system.forecastScaleCap.get_capped_forecast(sym, "carry90")
        if not fc.empty:
            all_forecasts.append(fc)
    except Exception:
        continue

if all_forecasts:
    combined = pd.concat(all_forecasts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(combined.values, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Carry Forecast Distribution (Capped at ±20)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Forecast stats: mean={combined.mean():.3f}, std={combined.std():.3f}, "
          f"skew={combined.skew():.3f}, kurt={combined.kurtosis():.3f}")

# %% [markdown]
# ## 9. Monthly Returns Heatmap

# %%
    monthly_returns = equity.resample("M").last().pct_change() * 100
monthly_returns.index = pd.PeriodIndex(monthly_returns.index, freq="ME")

# Pivot into years × months for heatmap
returns_pivot = monthly_returns.groupby([
    monthly_returns.index.year,
    monthly_returns.index.month
]).sum().unstack()

if not returns_pivot.empty and len(returns_pivot) > 1:
    fig, ax = plt.subplots(figsize=(14, max(4, len(returns_pivot) * 0.4)))
    im = ax.imshow(returns_pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(returns_pivot)))
    ax.set_yticklabels(returns_pivot.index)

    for i in range(len(returns_pivot)):
        for j in range(12):
            val = returns_pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < 5 else "white")

    ax.set_title("Monthly Returns Heatmap (%)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Return %")
    plt.tight_layout()
    plt.show()

# %%
print("\nBacktest complete.")
