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
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
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
print("Preloading all contract data...")
count = data.preload_all_data()
print(f"Preloaded {count}/{len(ALL_INSTRUMENTS)} instruments")

# Filter out delisted instruments: last price date > 180 days ago
# or carry data has no roll spread (PRICE == CARRY throughout)
import datetime as dt
cutoff_date = pd.Timestamp(dt.datetime.now() - dt.timedelta(days=180))
active_instruments = []
delisted = []
for sym in ALL_INSTRUMENTS:
    try:
        prices = data.daily_prices(sym)
        if prices is None or prices.empty:
            delisted.append(sym)
            continue
        last_date = prices.index[-1]
        if last_date < cutoff_date:
            delisted.append(sym)
            continue
        # Check if carry data has any roll spread
        carry_df = data._get_raw_carry_data_df(sym)
        if carry_df is not None and not carry_df.empty:
            spread = (carry_df["CARRY"] - carry_df["PRICE"]).dropna()
            if spread.abs().max() < 0.01:
                delisted.append(sym)
                continue
        active_instruments.append(sym)
    except Exception:
        delisted.append(sym)

instruments = active_instruments
print(f"Active instruments: {len(instruments)}/{len(ALL_INSTRUMENTS)}")
print(f"Delisted/insufficient data: {len(delisted)} -> {delisted}")

# %% [markdown]
# ## 2b. Volume Distribution Across Instruments
#
# Shows average daily volume per instrument to understand liquidity profile.
# Mean and median lines indicate the center of the distribution.
# Volume is summed across all individual contracts per product per day.

# %%
# Query volume from PostgreSQL (where individual contract data lives)
from sysdata.cnfutures.db_config import _get_engine
from sqlalchemy import text

volume_stats = []
try:
    engine = _get_engine()
    with engine.connect() as conn:
        stmt = text("""
            WITH individual AS (
                SELECT
                    UPPER(SUBSTRING(symbol FROM '^([A-Z]{1,2})[0-9]')) as product,
                    dt,
                    SUM(volume) as daily_volume
                FROM cnfutures_daily_prices
                WHERE symbol ~ '^[A-Z]{1,2}[0-9]{4}$'
                  AND volume IS NOT NULL AND volume > 0
                GROUP BY product, dt
            )
            SELECT product || '0' as cont_sym,
                   AVG(daily_volume) as avg_volume,
                   COUNT(*) as n_days,
                   MAX(dt) as last_date
            FROM individual
            GROUP BY product
            ORDER BY avg_volume DESC
        """)
        result = conn.execute(stmt)
        for row in result:
            volume_stats.append({
                "sym": row.cont_sym,
                "avg_volume": row.avg_volume,
                "n_days": row.n_days,
                "last_date": row.last_date,
            })
except Exception as e:
    print(f"Warning: Could not query volume from PG: {e}")

vol_df = pd.DataFrame(volume_stats)
# Filter to only active instruments in our universe
if not vol_df.empty:
    vol_df = vol_df[vol_df["sym"].isin(instruments)]
    vol_df = vol_df.sort_values("avg_volume", ascending=False)

if not vol_df.empty:
    mean_vol = vol_df["avg_volume"].mean()
    median_vol = vol_df["avg_volume"].median()
    max_days = vol_df["n_days"].max()

    fig, ax = plt.subplots(figsize=(14, max(8, len(vol_df) * 0.25)))
    ax2 = ax.twiny()

    colors = ["#2980b9" if v >= median_vol else "#95a5a6" for v in vol_df["avg_volume"]]
    ax.barh(range(len(vol_df)), vol_df["avg_volume"].values, color=colors, edgecolor="white")
    ax.axvline(mean_vol, color="#e74c3c", linewidth=1.5, linestyle="--", label=f"Mean ({mean_vol:,.0f})")
    ax.axvline(median_vol, color="#27ae60", linewidth=1.5, linestyle="-.", label=f"Median ({median_vol:,.0f})")

    ax2.scatter(vol_df["n_days"].values, range(len(vol_df)), color="#e67e22", s=30, zorder=5, label=f"Days of data (max={max_days})")

    ax.set_yticks(range(len(vol_df)))
    labels = [f"{r.sym} {_CNFUTURES_CONFIG.get(r.sym, ('',))[0]}" for r in vol_df.itertuples()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Average Daily Volume (summed across contracts)")
    ax2.set_xlabel("Number of Trading Days")
    ax.set_title("Average Daily Volume by Instrument", fontsize=12, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    ax.grid(True, alpha=0.3, axis="x")
    ax2.grid(False)
    plt.tight_layout()
    plt.show()

    print(f"\nVolume Stats: Mean = {mean_vol:,.0f}, Median = {median_vol:,.0f}")
    print(f"Top 5 by volume:")
    for _, r in vol_df.head(5).iterrows():
        print(f"  {r.sym}: {r.avg_volume:>12,.0f} ({r.n_days} days)")
    print(f"Bottom 5 by volume:")
    for _, r in vol_df.tail(5).iterrows():
        print(f"  {r.sym}: {r.avg_volume:>12,.0f} ({r.n_days} days)")

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
monthly_returns = equity.resample("ME").last().pct_change() * 100
monthly_returns.index = pd.PeriodIndex(monthly_returns.index, freq="M")

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

# %% [markdown]
# ## 10. Per-Instrument Decomposition
#
# Rank each instrument by carry Sharpe ratio. Shows which contracts perform
# best and worst using the carry rule, plus current carry signal for context.

# %%
# Rebuild system with all instruments (if running this cell standalone)
if "system" not in dir() or len(data.get_instrument_list()) < 80:
    data = CnFuturesSimData()
    data.preload_all_data()
    instruments = ALL_INSTRUMENTS
    config = Config()
    config.instruments = instruments
    config.notional_trading_capital = 100_000_000
    config.percentage_vol_target = 25.0
    config.base_currency = "CNY"
    config.instrument_weights = {s: 1 / len(instruments) for s in instruments}
    config.trading_rules = {
        "carry90": {
            "function": "systems.provided.rules.carry.carry",
            "data": ["rawdata.raw_carry"],
            "other_args": {"smooth_days": 90},
        }
    }
    config.forecast_weights = {"carry90": 1.0}
    config.use_forecast_scale_estimates = True
    config.forecast_div_multiplier = 1.0
    system = System(
        [Account(), Portfolios(), PositionSizing(), ForecastCombine(),
         ForecastScaleCap(), Rules(), RawData()],
        data=data, config=config,
    )

results = []
for sym in instruments:
    try:
        pos = system.portfolio.get_notional_position(sym)
        price = data.daily_prices(sym)
        common = pos.index.intersection(price.index)
        if len(common) < 20:
            continue
        pos = pos[common]
        price = price[common]
        price_diff = price.diff().shift(-1)
        daily_pnl = pos * price_diff
        notional = abs(pos) * price
        notional = notional.replace(0, np.nan)
        pnl_pct = daily_pnl / notional * 100
        rets = pnl_pct.dropna() / 100
        if len(rets) < 20:
            continue
        ann_ret = ((1 + rets.mean()) ** 252 - 1) * 100
        ann_vol = rets.std() * (252**0.5) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        eq = (pnl_pct.dropna() / 100 + 1).cumprod()
        max_dd = (eq / eq.cummax() - 1).min() * 100
        fc = system.forecastScaleCap.get_capped_forecast(sym, "carry90")
        rc = system.rawdata.raw_carry(sym)
        cfg = _CNFUTURES_CONFIG.get(sym, ("", 1, "", ""))
        results.append({
            "sym": sym, "name": cfg[0], "asset_class": cfg[2],
            "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "years": len(rets) / 252,
            "avg_forecast": fc.mean(), "latest_carry": rc.iloc[-1] if not rc.empty else np.nan,
        })
    except Exception:
        pass

perf = pd.DataFrame(results).sort_values("sharpe", ascending=False)

# Top and bottom 15
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
for ax, title, data_slice in [
    (axes[0], f"Top 15 by Carry Sharpe", perf.head(15)),
    (axes[1], f"Bottom 15 by Carry Sharpe", perf.tail(15)),
]:
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in data_slice.sharpe.values]
    ax.barh(range(len(data_slice)), data_slice.sharpe.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(data_slice)))
    labels = [f"{r.sym} {r.name}" for r in data_slice.itertuples()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
plt.tight_layout()
plt.show()

# Asset class summary
class_df = perf.groupby("asset_class").agg(
    {"sharpe": "mean", "ann_return": "mean", "sym": "count"}
).sort_values("sharpe", ascending=False)
print("\nBy Asset Class:")
for row in class_df.itertuples():
    print(f"  {row.Index:16s}: Sharpe {row.sharpe:.2f} | Ret {row.ann_return:.1f}% | n={row.sym}")

n_positive = (perf.sharpe > 0).sum()
print(f"\nSharpe > 0: {n_positive}/{len(perf)}")
print(f"Sharpe > 0.5: {(perf.sharpe > 0.5).sum()}/{len(perf)}")
print(f"Best:  {perf.iloc[0].sym} ({perf.iloc[0]['name']}) Sharpe={perf.iloc[0].sharpe:.2f}")
print(f"Worst: {perf.iloc[-1].sym} ({perf.iloc[-1]['name']}) Sharpe={perf.iloc[-1].sharpe:.2f}")

# %% [markdown]
# ## 11. Case Studies — Deep Dive into Selected Instruments
#
# Selects 4 instruments for detailed analysis:
# - **Best Sharpe**: highest risk-adjusted returns
# - **Worst Sharpe**: lowest risk-adjusted returns
# - **RB0 (Rebar)**: most liquid industrial futures on DCE
# - **IF0 (CSI 300)**: flagship equity index futures
#
# For each, examines: raw carry vs smoothed vs forecast, position dynamics,
# equity curve, return attribution, and signal persistence.

# %%
# Select case study instruments
best_sym = perf.iloc[0]["sym"]
worst_sym = perf.iloc[-1]["sym"]

# Pick RB0 and IF0 if available, otherwise fallback
candidate_representatives = ["RB0", "IF0", "CU0", "M0"]
representatives = [s for s in candidate_representatives if s in perf["sym"].values]

case_studies = list(dict.fromkeys([best_sym, worst_sym] + representatives))
case_studies = case_studies[:4]  # max 4

CASE_STUDY_NAMES = {
    best_sym: f"Best Sharpe ({perf.loc[perf['sym'] == best_sym].iloc[0]['name']})",
    worst_sym: f"Worst Sharpe ({perf.loc[perf['sym'] == worst_sym].iloc[0]['name']})",
}
for sym in representatives:
    if sym in case_studies and sym not in CASE_STUDY_NAMES:
        cfg = _CNFUTURES_CONFIG.get(sym, ("", 1, "", ""))
        CASE_STUDY_NAMES[sym] = f"{sym} {cfg[0]} ({cfg[2]})"

print(f"Case studies: {case_studies}")
for sym in case_studies:
    print(f"  {sym}: {CASE_STUDY_NAMES[sym]}")

# %%
def plot_case_study(system, sym, title, perf_df):
    """Deep-dive analysis for a single instrument."""
    row = perf_df.loc[perf_df["sym"] == sym]
    if row.empty:
        print(f"No perf data for {sym}")
        return

    row = row.iloc[0]

    # Gather data
    price = data.daily_prices(sym)
    raw_carry = system.rawdata.raw_carry(sym)
    smoothed_carry = system.rawdata.smoothed_carry(sym, smooth_days=90)
    forecast = system.forecastScaleCap.get_capped_forecast(sym, "carry90")
    position = system.portfolio.get_notional_position(sym)

    # Compute per-instrument equity
    common_idx = position.index.intersection(price.index)
    if len(common_idx) < 20:
        print(f"Insufficient data alignment for {sym}")
        return

    pos = position[common_idx]
    px = price[common_idx]
    px_diff = px.diff().shift(-1)
    daily_pnl = pos * px_diff
    notional = abs(pos) * px
    notional = notional.replace(0, np.nan)
    pnl_pct = daily_pnl / notional * 100
    equity = (pnl_pct.dropna() / 100 + 1).cumprod() * 100
    drawdown = (equity / equity.cummax() - 1) * 100

    # Signal analysis
    forecast_sign_changes = (forecast * forecast.shift(1) < 0).sum()
    forecast_zero_crossings = ((forecast.shift(1) <= 0) & (forecast > 0)).sum() + \
                              ((forecast.shift(1) >= 0) & (forecast < 0)).sum()
    pct_time_long = (forecast > 0).sum() / len(forecast) * 100
    pct_time_short = (forecast < 0).sum() / len(forecast) * 100
    pct_time_flat = (forecast.abs() < 0.5).sum() / len(forecast) * 100

    # Carry regime analysis
    raw_carry_mean = raw_carry.mean()
    raw_carry_std = raw_carry.std()
    carry_sharpe = raw_carry.mean() / raw_carry.std() if raw_carry.std() > 0 else 0

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Case Study: {title}", fontsize=14, fontweight="bold")

    # 1. Equity curve + drawdown
    ax = axes[0, 0]
    ax.plot(equity.index, equity.values, linewidth=1, color="#2c3e50")
    ax.set_title("Equity Curve (Indexed to 100)", fontsize=11)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1, 0]
    ax2.fill_between(drawdown.index, 0, drawdown.values, color="#e74c3c", alpha=0.5)
    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)

    # 2. Raw carry vs smoothed carry
    ax = axes[0, 1]
    ax.plot(raw_carry.index, raw_carry.values, linewidth=0.5, color="#95a5a6", label="Raw Carry", alpha=0.7)
    ax.plot(smoothed_carry.index, smoothed_carry.values, linewidth=1.5, color="#2980b9", label="Smoothed (90d EWMA)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Raw vs Smoothed Carry", fontsize=11)
    ax.set_ylabel("Carry (ann. roll / vol)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Capped forecast
    ax = axes[1, 1]
    ax.plot(forecast.index, forecast.values, linewidth=1, color="#8e44ad")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(20, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-20, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("Capped Forecast (±20 cap)", fontsize=11)
    ax.set_ylabel("Forecast")
    ax.grid(True, alpha=0.3)

    # 4. Position over time
    ax = axes[2, 0]
    ax.bar(position.index, position.values, width=1.5, color=["#27ae60" if v > 0 else "#e74c3c" for v in position.values], alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Notional Position Over Time", fontsize=11)
    ax.set_ylabel("Position (notional)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    # 5. Forecast distribution
    ax = axes[2, 1]
    ax.hist(forecast.values, bins=40, color="#8e44ad", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Forecast Distribution", fontsize=11)
    ax.set_xlabel("Forecast Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"  Asset Class:       {row['asset_class']}")
    print(f"  Backtest Period:   {row['years']:.1f} years")
    print(f"  Ann. Return:       {row['ann_return']:.2f}%")
    print(f"  Ann. Volatility:   {row['ann_vol']:.2f}%")
    print(f"  Sharpe Ratio:      {row['sharpe']:.2f}")
    print(f"  Max Drawdown:      {row['max_dd']:.2f}%")
    print(f"{'─' * 60}")
    print(f"  Carry Signal Stats:")
    print(f"    Mean Raw Carry:      {raw_carry_mean:.4f}")
    print(f"    Std Raw Carry:       {raw_carry_std:.4f}")
    print(f"    Carry Sharpe:        {carry_sharpe:.4f}")
    print(f"    Mean Smoothed Carry: {smoothed_carry.mean():.4f}")
    print(f"  Forecast Stats:")
    print(f"    Mean Forecast:       {forecast.mean():.3f}")
    print(f"    Std Forecast:        {forecast.std():.3f}")
    print(f"    Sign Changes:        {forecast_sign_changes}")
    print(f"    Zero Crossings:      {forecast_zero_crossings}")
    print(f"  Position Regime:")
    print(f"    % Time Long:         {pct_time_long:.1f}%")
    print(f"    % Time Short:        {pct_time_short:.1f}%")
    print(f"    % Time Flat (<0.5):  {pct_time_flat:.1f}%")
    print(f"{'─' * 60}\n")


# %%
# Run all case studies
for sym in case_studies:
    plot_case_study(system, sym, CASE_STUDY_NAMES[sym], perf)

# %% [markdown]
# ## 12. Case Study Comparison — Signal Persistence & Forecast Quality
#
# Compares how the carry rule behaves across instruments:
# - Signal persistence (how long forecasts stay in one direction)
# - Forecast-to-realized correlation (does the forecast predict returns?)
# - Turnover (how often positions change)

# %%
signal_analysis = []

for sym in case_studies:
    try:
        forecast = system.forecastScaleCap.get_capped_forecast(sym, "carry90")
        raw_carry = system.rawdata.raw_carry(sym)
        position = system.portfolio.get_notional_position(sym)
        price = data.daily_prices(sym)

        # Signal persistence: average run length of same-sign forecast
        forecast_sign = np.sign(forecast.values)
        sign_changes = np.diff(forecast_sign)
        runs = []
        current_run = 1
        for i in range(1, len(forecast_sign)):
            if sign_changes[i-1] == 0:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        avg_run_length = np.mean(runs) if runs else 0

        # Position turnover
        pos_changes = position.diff().abs()
        avg_turnover = pos_changes.mean()
        total_turnover = pos_changes.sum()

        # Forecast vs realized return correlation
        common_idx = forecast.index.intersection(price.index)
        if len(common_idx) > 30:
            fc_aligned = forecast.loc[common_idx]
            px_aligned = price.loc[common_idx]
            fwd_ret = px_aligned.shift(-1).pct_change().shift(1)  # 1-day forward return
            aligned = pd.DataFrame({"fc": fc_aligned, "ret": fwd_ret}).dropna()
            fc_ret_corr = aligned["fc"].corr(aligned["ret"]) if len(aligned) > 30 else np.nan
        else:
            fc_ret_corr = np.nan

        # Forecast autocorrelation (persistence of signal)
        fc_autocorr_1d = forecast.autocorr(lag=1)
        fc_autocorr_5d = forecast.autocorr(lag=5)
        fc_autocorr_20d = forecast.autocorr(lag=20)

        cfg = _CNFUTURES_CONFIG.get(sym, ("", 1, "", ""))
        signal_analysis.append({
            "sym": sym,
            "name": cfg[0],
            "asset_class": cfg[2],
            "avg_run_length": avg_run_length,
            "avg_daily_turnover": avg_turnover,
            "total_turnover": total_turnover,
            "fc_ret_correlation": fc_ret_corr,
            "fc_autocorr_1d": fc_autocorr_1d,
            "fc_autocorr_5d": fc_autocorr_5d,
            "fc_autocorr_20d": fc_autocorr_20d,
            "n_observations": len(forecast),
        })
    except Exception as e:
        print(f"Error analyzing {sym}: {e}")

signal_df = pd.DataFrame(signal_analysis)

# Display comparison table
print("\n" + "=" * 80)
print("SIGNAL PERSISTENCE & FORECAST QUALITY COMPARISON")
print("=" * 80)

for _, row in signal_df.iterrows():
    print(f"\n{row['sym']} ({row['name']}) — {row['asset_class']}")
    print(f"  Observations:        {row['n_observations']}")
    print(f"  Avg Run Length:      {row['avg_run_length']:.1f} days")
    print(f"  Avg Daily Turnover:  {row['avg_daily_turnover']:,.0f}")
    print(f"  Total Turnover:      {row['total_turnover']:,.0f}")
    print(f"  FC↔Return Corr:      {row['fc_ret_correlation']:.4f}")
    print(f"  FC Autocorr (1d):    {row['fc_autocorr_1d']:.4f}")
    print(f"  FC Autocorr (5d):    {row['fc_autocorr_5d']:.4f}")
    print(f"  FC Autocorr (20d):   {row['fc_autocorr_20d']:.4f}")

print("\n" + "=" * 80)

# %%
# Visual comparison of signal dynamics across case studies
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, sym in enumerate(case_studies):
    if idx >= 4:
        break
    try:
        forecast = system.forecastScaleCap.get_capped_forecast(sym, "carry90")
        raw_carry = system.rawdata.raw_carry(sym)

        ax = axes[idx]
        ax.plot(raw_carry.index, raw_carry.values, linewidth=0.5, color="#95a5a6", alpha=0.5, label="Raw")
        ax.plot(forecast.index, forecast.values, linewidth=1.2, color="#2980b9", label="Forecast")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{sym} — {CASE_STUDY_NAMES[sym]}", fontsize=11)
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except Exception:
        axes[idx].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[idx].transAxes)

plt.suptitle("Signal Dynamics Comparison: Raw Carry vs Capped Forecast", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# Bar chart comparison of key signal metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

labels = [f"{s}\n({CASE_STUDY_NAMES[s].split('(')[0].strip()})" for s in case_studies[:4]]

# Autocorrelation comparison
ax = axes[0]
x = np.arange(len(case_studies[:4]))
width = 0.25
ax.bar(x - width, [signal_df.loc[signal_df['sym'] == s, 'fc_autocorr_1d'].values[0] for s in case_studies[:4]], width, label="1-day", color="#3498db")
ax.bar(x, [signal_df.loc[signal_df['sym'] == s, 'fc_autocorr_5d'].values[0] for s in case_studies[:4]], width, label="5-day", color="#2ecc71")
ax.bar(x + width, [signal_df.loc[signal_df['sym'] == s, 'fc_autocorr_20d'].values[0] for s in case_studies[:4]], width, label="20-day", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Forecast Autocorrelation", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Run length comparison
ax = axes[1]
run_lengths = [signal_df.loc[signal_df['sym'] == s, 'avg_run_length'].values[0] for s in case_studies[:4]]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"][:len(run_lengths)]
ax.bar(range(len(run_lengths)), run_lengths, color=colors, edgecolor="white")
ax.set_xticks(range(len(run_lengths)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Avg Signal Run Length (days)", fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

# Forecast-return correlation
ax = axes[2]
fc_ret_corrs = [signal_df.loc[signal_df['sym'] == s, 'fc_ret_correlation'].values[0] for s in case_studies[:4]]
bar_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in fc_ret_corrs]
ax.bar(range(len(fc_ret_corrs)), fc_ret_corrs, color=bar_colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(range(len(fc_ret_corrs)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Forecast ↔ Forward Return Correlation", fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %%
print("\nCase study analysis complete.")
