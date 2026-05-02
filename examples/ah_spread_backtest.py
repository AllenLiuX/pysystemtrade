# %% [markdown]
# # A+H Spread Trading Rule Backtest
#
# This notebook backtests the `ah_spread` mean-reversion trading rule on A+H dual-listed stocks.
#
# **Strategy:** When A-share outperforms H-share, sell A / buy H. When A-share underperforms H-share, buy A / sell H.
#
# **Signal:** Rolling z-score of cumulative log return spread between paired A and H shares.
#
# **Universe:** 188 A+H dual-listed pairs (376 instruments total).
#
# **Analysis includes:**
# - Full universe vs Top 20 by liquidity comparison
# - Case studies on representative pairs
# - Forecast distribution and turnover analysis
# - Predictive power: IC (Information Coefficient) vs forward returns
#
# ---
#
# ## Assumptions
#
# 1. **Instrument treatment:** A and H shares are paired instruments in a dollar-neutral pair trade.
#    Each pair receives opposite-signed forecasts: when A gets a positive forecast, H gets negative.
#
# 2. **Signal construction:** The spread is computed as `cumsum(log_ret_A - log_ret_H)`.
#    A-shares receive the raw z-score; H-shares receive the negated z-score.
#    After the `ah_spread` rule negates the signal, A and H have opposite-signed forecasts.
#    When z-score is positive (A outperformed): A gets short signal, H gets long signal.
#    When z-score is negative (A underperformed): A gets long signal, H gets short signal.
#
# 3. **Dollar-neutral positions:** Raw subsystem positions are normalized per pair so that
#    `|position_A| == |position_H|` and `position_A = -position_H`. The magnitude is the
#    average of the absolute raw positions. A-share's sign determines the pair direction.
#
# 4. **Initial capital:** 10M CNY, split equally across all pairs (not instruments).
#    Each pair receives equal notional allocation.
#
# 5. **Forecast scaling:** Default pysystemtrade scaling — forecasts are scaled to a target
#    magnitude of 10 and capped at ±20.
#
# 6. **Short selling:** No constraints. Both long and short positions are allowed on A and H shares.
#    In practice, A-share short selling may be restricted (margin trading only, limited stock borrow).
#
# 7. **Transaction costs:** Default pysystemtrade A-stock costs (~0.1% per trade, including stamp duty
#    and commission for A-shares; HK brokerage fees for H-shares are not modeled separately).
#
# 8. **Data source:** Daily adjusted (forward-adjusted, qfq) prices from Supabase/parquet.
#    H-share prices are NOT converted to CNY for the spread calculation — log returns are unitless.
#
# 9. **Common trading days:** Spread is computed only on days when both A and H markets are open.
#
# 10. **Lookback:** 20-day rolling window for z-score calculation (configurable).
#
# 11. **Liquidity proxy:** Top 20 selection is based on average daily trading volume (in shares).
#
# 12. **No regime filter:** The strategy runs continuously regardless of market regime.
#
# 13. **Survivorship bias:** The A+H universe is based on currently dual-listed stocks.
#
# 14. **FX risk:** H-share positions are denominated in HKD. Portfolio P&L is reported in CNY.

# %%
# Setup: Change to project root directory
import os
from pathlib import Path


project_root = Path(os.getcwd()).parent if "examples" in os.getcwd() else Path(os.getcwd())
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

# %%
# Standard library imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# pysystemtrade imports
from sysdata.sim.astock_sim_data import AStockSimData
from sysdata.config.configdata import Config
from systems.basesystem import System
from systems.rawdata import RawData
from systems.ah_data import AHData
from systems.trading_rules import TradingRule
from systems.forecasting import Rules
from systems.provided.rules.ah_spread import ah_spread
from systems.portfolio import Portfolios
from systems.forecast_scale_cap import ForecastScaleCap
from systems.forecast_combine import ForecastCombine
from systems.positionsizing import PositionSizing
from systems.accounts.accounts_stage import Account

# A+H infrastructure
from sysdata.astock.akshare_client import AkshareClient
from sysdata.astock.universe import AStockUniverse

# %%
# Universe definition: Get all A+H pairs
print("=" * 60)
print("A+H SPREAD BACKTEST — UNIVERSE DEFINITION")
print("=" * 60)

# Load A+H mapping
pairs = AkshareClient.get_ah_pairs()
print(f"Total A+H pairs in mapping: {len(pairs)}")

# Build instrument list: both A and H shares
all_instruments = []
for a_code, h_code in pairs:
    all_instruments.append(a_code)
    all_instruments.append(h_code + ".HK")

print(f"Total instruments (A + H): {len(all_instruments)}")

# %%
# Data verification: Check which instruments have data
print("\nChecking data availability...")

data = AStockSimData()
available = set(data.get_instrument_list())

# Filter to instruments with data
instruments_with_data = [i for i in all_instruments if i in available]
print(f"Instruments with data: {len(instruments_with_data)} / {len(all_instruments)}")

# Build pair-level availability
pairs_with_data = []
for a_code, h_code in pairs:
    h_full = h_code + ".HK"
    if a_code in available and h_full in available:
        pairs_with_data.append((a_code, h_full))

print(f"Complete pairs (both legs have data): {len(pairs_with_data)}")

# %%
# Top 20 selection by liquidity (average daily volume)
print("\nSelecting Top 20 pairs by average daily volume...")

# Only preload A-share prices — H-shares not needed for volume proxy
a_codes_only = [a for a, _ in pairs_with_data]
print(f"Preloading prices for {len(a_codes_only)} A-share instruments...")
data.preload_prices(a_codes_only)
print(f"Preloaded {len(data._price_cache)} instruments")

# Vectorized volume calculation: last 60-day mean for all A-shares at once
volume_data = {}
for a_code in a_codes_only:
    a_prices = data._price_cache.get(a_code)
    if a_prices is not None and len(a_prices) > 60:
        volume_data[a_code] = a_prices.iloc[-60:].mean()

# Sort by volume and take top 20
top_20_a = sorted(volume_data.keys(), key=lambda x: volume_data[x], reverse=True)[:20]

# Reuse pairs mapping from earlier (line 112) — no need to refetch
a_to_h = {a: h + ".HK" for a, h in pairs}

top_20_pairs_fixed = [(a, a_to_h[a]) for a in top_20_a if a in a_to_h]
top_20_instruments = [code for pair in top_20_pairs_fixed for code in pair]

print(f"Top 20 pairs selected:")
for a, h in top_20_pairs_fixed:
    vol = volume_data.get(a, 0)
    print(f"  {a} / {h}: avg vol = {vol:,.0f}")

# %%
# System builder function
def build_ah_backtest_system(
    instruments: list,
    lookback: int = 20,
    capital: float = 10_000_000,
):
    """
    Build a pysystemtrade system for A+H spread trading.

    Args:
        instruments: List of instrument codes (A and H shares)
        lookback: Z-score rolling window in days
        capital: Notional trading capital in CNY

    Returns:
        System instance
    """
    config = Config()
    config.instruments = instruments
    config.notional_trading_capital = capital

    # Equal instrument weights
    n = len(instruments)
    config.instrument_weights = {instr: 1.0 / n for instr in instruments}

    config.trading_rules = {
        "ah_spread": {
            "function": "systems.provided.rules.ah_spread.ah_spread",
            "data": ["ah_data.get_ah_spread_zscore"],
            "other_args": {"_lookback": lookback},
            "forecast_scalar": 10.0,
        },
    }

    rules = Rules(config.trading_rules)

    stages = [
        Account(),
        ForecastScaleCap(),
        rules,
        ForecastCombine(),
        PositionSizing(),
        Portfolios(),
        RawData(),
        AHData(),
    ]

    system = System(stages, data=data, config=config)
    return system


# %%
# Performance metrics calculator
def calc_performance_metrics(equity_curve, trading_days_per_year=252):
    """Calculate standard performance metrics from an equity curve."""
    returns = equity_curve.pct_change().dropna()

    if len(returns) < 2:
        return {}

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    years = len(equity_curve) / trading_days_per_year
    ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(trading_days_per_year) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_dd = drawdown.min()

    negative_returns = returns[returns < 0]
    downside_dev = negative_returns.std() * np.sqrt(trading_days_per_year) * 100 if len(negative_returns) > 0 else 0
    sortino = ann_return / downside_dev if downside_dev > 0 else 0

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    return {
        "Total Return (%)": round(total_return, 2),
        "Annualized Return (%)": round(ann_return, 2),
        "Annualized Vol (%)": round(ann_vol, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Calmar Ratio": round(calmar, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Trading Days": len(equity_curve),
        "Years": round(years, 2),
    }


def calc_instrument_pnl_decomposition(system, instruments, pairs_list):
    """
    Decompose portfolio P&L into 4 buckets:
    - Long A-shares
    - Short A-shares
    - Long H-shares
    - Short H-shares

    Each bucket accumulates daily P&L independently — no common-index intersection.

    Returns a dict with equity curves for each bucket and a summary DataFrame.
    """
    a_codes = {a for a, _ in pairs_list}

    # Collect daily P&L per bucket as lists of series
    bucket_daily_pnl = {"Long A": [], "Short A": [], "Long H": [], "Short H": []}

    for instr in instruments:
        try:
            pos = system.accounts.get_buffered_position(instr)
            pnl = system.accounts.pandl_for_instrument(instr).curve()
            if len(pos) == 0 or len(pnl) == 0:
                continue

            # Reindex position to match P&L dates
            pos_aligned = pos.reindex(pnl.index, method="ffill").fillna(0.0)

            is_a = instr in a_codes
            bucket_prefix = "Long A" if is_a else "Long H"
            short_prefix = "Short A" if is_a else "Short H"

            # Compute daily P&L from cumulative P&L
            daily_pnl = pnl.diff().fillna(pnl.iloc[0] if len(pnl) > 0 else 0.0)

            long_mask = pos_aligned > 0
            short_mask = pos_aligned < 0

            bucket_daily_pnl[bucket_prefix].append(daily_pnl.where(long_mask, 0.0))
            bucket_daily_pnl[short_prefix].append(daily_pnl.where(short_mask, 0.0))
        except Exception:
            pass

    # Build cumulative curves per bucket
    result = {}
    for bucket, daily_list in bucket_daily_pnl.items():
        if daily_list:
            combined = pd.DataFrame(daily_list).sum()
            result[bucket] = combined.cumsum()
        else:
            result[bucket] = pd.Series(0.0)

    return result


def calc_pair_normalized_pnl(system, pairs_list, data):
    """
    Compute pair-level P&L using dollar-normalized positions.

    For each pair:
    1. Get raw positions for both legs
    2. Normalize positions: avg_abs = (|pos_A| + |pos_H|) / 2
    3. Dollar P&L = capital_per_pair * (norm_pos_A * ret_A + norm_pos_H * ret_H)
    4. Cumulative P&L = cumsum of daily dollar P&L

    Returns:
        Dict with per-pair equity curves and aggregate pair P&L
    """
    if not pairs_list:
        return {}, pd.Series(dtype=float)

    capital_per_pair = system.config.notional_trading_capital / len(pairs_list)

    pair_pnl_curves = {}
    daily_pnl_series = []

    for a_code, h_code in pairs_list:
        try:
            # Get raw positions
            pos_a = system.accounts.get_buffered_position(a_code)
            pos_h = system.accounts.get_buffered_position(h_code)

            if len(pos_a) == 0 or len(pos_h) == 0:
                continue

            # Normalize positions
            common_dates = pos_a.index.intersection(pos_h.index)
            if len(common_dates) < 2:
                continue

            pos_a_aligned = pos_a.reindex(common_dates, method="ffill").fillna(0.0)
            pos_h_aligned = pos_h.reindex(common_dates, method="ffill").fillna(0.0)

            avg_abs = (pos_a_aligned.abs() + pos_h_aligned.abs()) / 2.0
            sign_a = pos_a_aligned.apply(lambda x: 1 if x >= 0 else -1)

            norm_pos_a = avg_abs * sign_a
            norm_pos_h = -norm_pos_a

            # Get returns
            prices_a = data.get_raw_price(a_code)
            prices_h = data.get_raw_price(h_code)

            if prices_a is None or prices_h is None:
                continue

            ret_a = prices_a.loc[common_dates].pct_change().dropna()
            ret_h = prices_h.loc[common_dates].pct_change().dropna()

            # Compute normalized P&L
            common_ret = ret_a.index.intersection(ret_h.index)
            if len(common_ret) == 0:
                continue

            norm_pos_a_ret = norm_pos_a.reindex(common_ret, method="ffill").fillna(0.0)
            norm_pos_h_ret = norm_pos_h.reindex(common_ret, method="ffill").fillna(0.0)

            # Dollar P&L = capital * position * return
            pair_daily_dollar_pnl = capital_per_pair * (
                norm_pos_a_ret.loc[common_ret] * ret_a.loc[common_ret] +
                norm_pos_h_ret.loc[common_ret] * ret_h.loc[common_ret]
            )

            daily_pnl_series.append(pair_daily_dollar_pnl)

            # Per-pair cumulative P&L as equity curve
            pair_cum_pnl = pair_daily_dollar_pnl.cumsum() + capital_per_pair
            pair_pnl_curves[f"{a_code}/{h_code}"] = pair_cum_pnl

        except Exception as e:
            continue

    # Aggregate all daily P&L series by summing
    if daily_pnl_series:
        combined_daily = pd.DataFrame(daily_pnl_series).sum()
        all_pair_pnl = combined_daily.cumsum() + system.config.notional_trading_capital
    else:
        all_pair_pnl = pd.Series(dtype=float)

    return pair_pnl_curves, all_pair_pnl


# %% [markdown]
# ## Backtest: Top 20 Pairs by Liquidity

# %%
print("=" * 60)
print("BACKTEST: TOP 20 PAIRS BY LIQUIDITY")
print("=" * 60)

top_20_system = build_ah_backtest_system(top_20_instruments, lookback=20)
print(f"System created with {len(top_20_instruments)} instruments")
print(f"Capital: {top_20_system.config.notional_trading_capital:,.0f} CNY")

# Get equity curve
top_20_equity = top_20_system.accounts.portfolio().curve() + top_20_system.config.notional_trading_capital
top_20_metrics = calc_performance_metrics(top_20_equity)

print("\nPerformance Metrics:")
for k, v in top_20_metrics.items():
    print(f"  {k}: {v}")

# %%
# Plot Top 20 equity curve and drawdown
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(top_20_equity.index, top_20_equity.values, 'b-', linewidth=1.5)
axes[0].set_ylabel('Portfolio Value (CNY)')
axes[0].set_title('Top 20 A+H Pairs — Equity Curve')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=top_20_system.config.notional_trading_capital, color='gray', linestyle='--', alpha=0.5)

# Drawdown
returns = top_20_equity.pct_change().dropna()
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max * 100

axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Drawdown')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Backtest: Full Universe (188 Pairs)

# %%
print("=" * 60)
print("BACKTEST: FULL UNIVERSE (188 PAIRS)")
print("=" * 60)

full_instruments = [i for i in instruments_with_data]
print(f"Instruments in full universe: {len(full_instruments)}")

full_system = build_ah_backtest_system(full_instruments, lookback=20)
print(f"System created with {len(full_instruments)} instruments")

# Get equity curve
full_equity = full_system.accounts.portfolio().curve() + full_system.config.notional_trading_capital
full_metrics = calc_performance_metrics(full_equity)

print("\nPerformance Metrics:")
for k, v in full_metrics.items():
    print(f"  {k}: {v}")

# %%
# Plot Full universe equity curve and drawdown
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(full_equity.index, full_equity.values, 'g-', linewidth=1.5)
axes[0].set_ylabel('Portfolio Value (CNY)')
axes[0].set_title('Full A+H Universe (188 Pairs) — Equity Curve')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=full_system.config.notional_trading_capital, color='gray', linestyle='--', alpha=0.5)

full_returns = full_equity.pct_change().dropna()
full_cumulative = (1 + full_returns).cumprod()
full_running_max = full_cumulative.cummax()
full_drawdown = (full_cumulative - full_running_max) / full_running_max * 100

axes[1].fill_between(full_drawdown.index, full_drawdown.values, 0, color='red', alpha=0.3)
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Drawdown')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Strategy Comparison: Top 20 vs Full Universe

# %%
# Combined equity curve comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

axes[0].plot(top_20_equity.index, top_20_equity.values, 'b-', linewidth=2, label='Top 20 by Liquidity')
axes[0].plot(full_equity.index, full_equity.values, 'g-', linewidth=1.5, label='Full Universe (188 pairs)')
axes[0].set_ylabel('Portfolio Value (CNY)')
axes[0].set_title('Strategy Comparison: Top 20 vs Full Universe')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=10_000_000, color='gray', linestyle='--', alpha=0.5)

axes[1].fill_between(drawdown.index, drawdown.values, 0, color='blue', alpha=0.3, label='Top 20')
axes[1].fill_between(full_drawdown.index, full_drawdown.values, 0, color='green', alpha=0.2, label='Full Universe')
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Drawdown Comparison')
axes[1].legend(loc='lower left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Summary table
print("\n" + "=" * 65)
print("STRATEGY COMPARISON SUMMARY")
print("=" * 65)
print(f"{'Metric':<22} {'Top 20':>18} {'Full Universe':>18}")
print("-" * 65)
for metric in top_20_metrics.keys():
    top_val = top_20_metrics.get(metric, "N/A")
    full_val = full_metrics.get(metric, "N/A")
    print(f"{metric:<22} {top_val:>18} {full_val:>18}")
print("=" * 65)

# %% [markdown]
# ## Case Studies: Representative Pairs
#
# We examine 3 representative A+H pairs in detail:
# - **601318.SH / 02318.HK** (中国平安) — Large-cap financial
# - **601899.SH / 02899.HK** (紫金矿业) — Materials/commodity
# - **000333.SZ / 00300.HK** (美的集团) — Consumer discretionary

# %%
CASE_STUDIES = [
    ("601318.SH", "02318.HK", "中国平安"),
    ("601899.SH", "02899.HK", "紫金矿业"),
    ("000333.SZ", "00300.HK", "美的集团"),
]

for a_code, h_code, name in CASE_STUDIES:
    print(f"\n{'=' * 60}")
    print(f"CASE STUDY: {name} ({a_code} / {h_code})")
    print(f"{'=' * 60}")

    # Get spread and z-score
    spread = full_system.ah_data.get_ah_log_return_spread(a_code)
    zscore = full_system.ah_data.get_ah_spread_zscore(a_code, lookback=20)

    # Get raw forecast and capped forecast
    raw_forecast = full_system.rules.get_raw_forecast(a_code, "ah_spread")
    capped_forecast = full_system.forecastScaleCap.get_capped_forecast(a_code, "ah_spread")

    # Get prices
    a_prices = data.get_raw_price(a_code)
    h_prices = data.get_raw_price(h_code)

    print(f"Spread data: {len(spread)} points, {spread.index[0].date()} to {spread.index[-1].date()}")
    print(f"Z-score range: [{zscore.min():.2f}, {zscore.max():.2f}]")
    print(f"Raw forecast range: [{raw_forecast.min():.2f}, {raw_forecast.max():.2f}]")
    print(f"Capped forecast range: [{capped_forecast.min():.2f}, {capped_forecast.max():.2f}]")

    # Get normalized positions
    try:
        norm_pos_a = full_system.ah_data.get_ah_pair_normalized_position(a_code)
        norm_pos_h = full_system.ah_data.get_ah_pair_normalized_position(h_code)

        if not norm_pos_a.empty and not norm_pos_h.empty:
            print(f"Normalized position A range: [{norm_pos_a.min():.4f}, {norm_pos_a.max():.4f}]")
            print(f"Normalized position H range: [{norm_pos_h.min():.4f}, {norm_pos_h.max():.4f}]")
            # Verify dollar-neutrality
            diff = (norm_pos_a + norm_pos_h).abs().max()
            print(f"Max |pos_A + pos_H| (should be ~0): {diff:.2e}")
    except Exception as e:
        print(f"Normalized positions: Error — {e}")

    # Plot case study
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Prices
    axes[0].plot(a_prices.index, a_prices.values, 'b-', linewidth=1, label=f'{a_code} (A)')
    axes[0].plot(h_prices.index, h_prices.values, 'r-', linewidth=1, label=f'{h_code} (H)')
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'{name} — Prices')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Spread
    axes[1].plot(spread.index, spread.values, 'k-', linewidth=1)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Spread')
    axes[1].set_title('Cumulative Log Return Spread (A - H)')
    axes[1].grid(True, alpha=0.3)

    # Z-score
    axes[2].plot(zscore.index, zscore.values, 'purple', linewidth=1)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axhline(y=1, color='orange', linestyle=':', alpha=0.5)
    axes[2].axhline(y=-1, color='orange', linestyle=':', alpha=0.5)
    axes[2].axhline(y=2, color='red', linestyle=':', alpha=0.5)
    axes[2].axhline(y=-2, color='red', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Z-Score')
    axes[2].set_title('20-Day Rolling Z-Score')
    axes[2].grid(True, alpha=0.3)

    # Forecast (raw and capped)
    axes[3].plot(raw_forecast.index, raw_forecast.values, 'b-', linewidth=0.8, label='Raw', alpha=0.7)
    axes[3].plot(capped_forecast.index, capped_forecast.values, 'r-', linewidth=1, label='Capped')
    axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[3].axhline(y=10, color='green', linestyle=':', alpha=0.5)
    axes[3].axhline(y=-10, color='green', linestyle=':', alpha=0.5)
    axes[3].axhline(y=20, color='red', linestyle=':', alpha=0.5)
    axes[3].axhline(y=-20, color='red', linestyle=':', alpha=0.5)
    axes[3].set_ylabel('Forecast')
    axes[3].set_title('Forecast Signal (Raw vs Capped)')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Per-instrument P&L with full performance metrics
    print(f"\n{'─' * 50}")
    print(f"Per-Instrument Performance:")
    print(f"{'─' * 50}")

    for instr_code, label in [(a_code, "A-share"), (h_code, "H-share")]:
        try:
            pnl_curve = full_system.accounts.pandl_for_instrument(instr_code).curve()
            metrics = calc_performance_metrics(pnl_curve)
            print(f"\n  {instr_code} ({label}):")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"\n  {instr_code} ({label}): Error — {e}")

    # Per-instrument P&L plot
    try:
        a_pnl = full_system.accounts.pandl_for_instrument(a_code).curve()
        h_pnl = full_system.accounts.pandl_for_instrument(h_code).curve()
        pair_pnl = a_pnl + h_pnl

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(a_pnl.index, a_pnl.values, 'b-', linewidth=1.5, label=f'{a_code} (A)')
        ax.plot(h_pnl.index, h_pnl.values, 'r-', linewidth=1.5, label=f'{h_code} (H)')
        ax.plot(pair_pnl.index, pair_pnl.values, 'k-', linewidth=2, label='Pair Total')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Cumulative P&L (CNY)')
        ax.set_title(f'{name} — Per-Instrument P&L')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"  Could not plot P&L: {e}")

# %% [markdown]
# ## P&L Decomposition: Long A, Short A, Long H, Short H
#
# This section decomposes the total portfolio P&L into 4 buckets based on
# instrument type (A-share vs H-share) and position direction (long vs short).
#
# This reveals whether the strategy profits more from:
# - Going long on undervalued A-shares
# - Shorting overvalued A-shares
# - Going long on undervalued H-shares
# - Shorting overvalued H-shares

# %%
print("=" * 60)
print("P&L DECOMPOSITION ANALYSIS")
print("=" * 60)

# Decompose full universe
decomp = calc_instrument_pnl_decomposition(full_system, full_instruments, pairs_with_data)

if decomp is not None:
    print(f"\nCumulative P&L by bucket:")
    for bucket, curve in decomp.items():
        final_pnl = curve.iloc[-1]
        print(f"  {bucket:>10}: {final_pnl:>12,.0f} CNY")

    total_decomp = sum(curve.iloc[-1] for curve in decomp.values())
    print(f"  {'Total':>10}: {total_decomp:>12,.0f} CNY")

    # Plot decomposition
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Individual bucket equity curves
    colors = {'Long A': '#2ecc71', 'Short A': '#e74c3c', 'Long H': '#3498db', 'Short H': '#f39c12'}
    for bucket, curve in decomp.items():
        axes[0].plot(curve.index, curve.values, color=colors.get(bucket, 'gray'),
                     linewidth=1.5, label=bucket)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Cumulative P&L (CNY)')
    axes[0].set_title('P&L Decomposition by Position Type')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Stacked area chart
    decomp_df = pd.DataFrame(decomp)
    axes[1].stackplot(decomp_df.index,
                      decomp_df['Long A'].values,
                      decomp_df['Short A'].values,
                      decomp_df['Long H'].values,
                      decomp_df['Short H'].values,
                      labels=['Long A', 'Short A', 'Long H', 'Short H'],
                      colors=[colors[b] for b in ['Long A', 'Short A', 'Long H', 'Short H']],
                      alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Cumulative P&L (CNY)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Stacked P&L Decomposition')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Rolling contribution (60-day window)
    fig, ax = plt.subplots(figsize=(14, 5))
    for bucket, curve in decomp.items():
        rolling_contrib = curve.rolling(60).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
        ax.plot(rolling_contrib.index, rolling_contrib.values, color=colors.get(bucket, 'gray'),
                linewidth=1.5, label=bucket)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('60-Day Rolling P&L (CNY)')
    ax.set_xlabel('Date')
    ax.set_title('60-Day Rolling P&L by Position Type')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Bucket':>12} {'Total P&L':>14} {'% of Total':>10}")
    print(f"{'─' * 70}")

    total_pnl = sum(curve.iloc[-1] for curve in decomp.values())
    for bucket, curve in decomp.items():
        total_ret = curve.iloc[-1]
        pct = (total_ret / total_pnl * 100) if total_pnl != 0 else 0
        print(f"{bucket:>12} {total_ret:>14,.0f} {pct:>9.1f}%")
    print(f"{'─' * 70}")
    print(f"{'Total':>12} {total_pnl:>14,.0f}")
    print(f"{'=' * 70}")
else:
    print("Could not compute decomposition — insufficient position data")

# %%
# Decomposition for Top 20
print("\n" + "=" * 60)
print("P&L DECOMPOSITION: TOP 20 PAIRS")
print("=" * 60)

decomp_top20 = calc_instrument_pnl_decomposition(top_20_system, top_20_instruments, top_20_pairs_fixed)

if decomp_top20 is not None:
    print(f"\nCumulative P&L by bucket:")
    for bucket, curve in decomp_top20.items():
        final_pnl = curve.iloc[-1]
        print(f"  {bucket:>10}: {final_pnl:>12,.0f} CNY")

    total_decomp_t20 = sum(curve.iloc[-1] for curve in decomp_top20.values())
    print(f"  {'Total':>10}: {total_decomp_t20:>12,.0f} CNY")

    # Plot Top 20 decomposition
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {'Long A': '#2ecc71', 'Short A': '#e74c3c', 'Long H': '#3498db', 'Short H': '#f39c12'}
    for bucket, curve in decomp_top20.items():
        ax.plot(curve.index, curve.values, color=colors.get(bucket, 'gray'),
                linewidth=1.5, label=bucket)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Cumulative P&L (CNY)')
    ax.set_title('Top 20 Pairs — P&L Decomposition')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Bucket':>12} {'Total P&L':>14} {'% of Total':>10}")
    print(f"{'─' * 70}")

    total_pnl_t20 = sum(curve.iloc[-1] for curve in decomp_top20.values())
    for bucket, curve in decomp_top20.items():
        total_ret = curve.iloc[-1]
        pct = (total_ret / total_pnl_t20 * 100) if total_pnl_t20 != 0 else 0
        print(f"{bucket:>12} {total_ret:>14,.0f} {pct:>9.1f}%")
    print(f"{'─' * 70}")
    print(f"{'Total':>12} {total_pnl_t20:>14,.0f}")
    print(f"{'=' * 70}")
else:
    print("Could not compute decomposition — insufficient position data")

# %% [markdown]
# ## Pair-Level P&L with Dollar-Normalized Positions
#
# This section recomputes P&L using dollar-neutral pair positions where
# `|position_A| == |position_H|` and `position_A = -position_H`.

# %%
print("=" * 60)
print("PAIR-LEVEL P&L (DOLLAR-NORMALIZED)")
print("=" * 60)

pair_curves, total_pair_pnl = calc_pair_normalized_pnl(full_system, pairs_with_data, data)

if total_pair_pnl is not None and not total_pair_pnl.empty:
    pair_metrics = calc_performance_metrics(total_pair_pnl)
    print("\nNormalized Pair Performance Metrics:")
    for k, v in pair_metrics.items():
        print(f"  {k}: {v}")

    # Plot normalized pair equity curve
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(total_pair_pnl.index, total_pair_pnl.values, 'purple', linewidth=1.5)
    axes[0].set_ylabel('Portfolio Value (CNY)')
    axes[0].set_title('Dollar-Normalized Pair P&L — Equity Curve')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=full_system.config.notional_trading_capital, color='gray', linestyle='--', alpha=0.5)

    # Drawdown
    pair_returns = total_pair_pnl.pct_change().dropna()
    pair_cumulative = (1 + pair_returns).cumprod()
    pair_running_max = pair_cumulative.cummax()
    pair_drawdown = (pair_cumulative - pair_running_max) / pair_running_max * 100

    axes[1].fill_between(pair_drawdown.index, pair_drawdown.values, 0, color='red', alpha=0.3)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Drawdown')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Top and bottom pairs
    if pair_curves:
        final_pnls = {pair: curve.iloc[-1] for pair, curve in pair_curves.items()}
        top_pairs = sorted(final_pnls.items(), key=lambda x: x[1], reverse=True)[:10]
        bottom_pairs = sorted(final_pnls.items(), key=lambda x: x[1])[:10]

        print(f"\nTop 10 Pairs by P&L:")
        for pair, pnl in top_pairs:
            print(f"  {pair}: {pnl:>12,.0f} CNY")

        print(f"\nBottom 10 Pairs by P&L:")
        for pair, pnl in bottom_pairs:
            print(f"  {pair}: {pnl:>12,.0f} CNY")
else:
    print("Could not compute normalized pair P&L")

# %% [markdown]
# ## Forecast Analysis
#
# Examining the distribution, turnover, and extremity of forecast signals.

# %%
print("=" * 60)
print("FORECAST ANALYSIS")
print("=" * 60)

# Collect all capped forecasts
all_forecasts = {}
for instr in full_instruments[:50]:  # Sample first 50 for speed
    try:
        fc = full_system.forecastScaleCap.get_capped_forecast(instr, "ah_spread")
        if len(fc) > 0:
            all_forecasts[instr] = fc
    except Exception:
        pass

print(f"Instruments with forecast data: {len(all_forecasts)}")

# Forecast distribution
all_fc_values = []
for fc in all_forecasts.values():
    all_fc_values.extend(fc.dropna().values)

all_fc_values = np.array(all_fc_values)

print(f"\nForecast Statistics:")
print(f"  Mean: {np.mean(all_fc_values):.4f}")
print(f"  Std:  {np.std(all_fc_values):.4f}")
print(f"  Median: {np.median(all_fc_values):.4f}")
print(f"  Min: {np.min(all_fc_values):.4f}")
print(f"  Max: {np.max(all_fc_values):.4f}")
print(f"  % at +20: {np.mean(all_fc_values >= 19.9) * 100:.1f}%")
print(f"  % at -20: {np.mean(all_fc_values <= -19.9) * 100:.1f}%")
print(f"  % near 0 (|fc| < 1): {np.mean(np.abs(all_fc_values) < 1) * 100:.1f}%")

# %%
# Forecast distribution histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(all_fc_values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
axes[0].axvline(x=10, color='green', linestyle=':', alpha=0.5)
axes[0].axvline(x=-10, color='green', linestyle=':', alpha=0.5)
axes[0].axvline(x=20, color='orange', linestyle=':', alpha=0.5)
axes[0].axvline(x=-20, color='orange', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Forecast Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Forecast Distribution (All Instruments)')
axes[0].grid(True, alpha=0.3)

# Time spent at extremes
extreme_pct = np.mean(np.abs(all_fc_values) > 10) * 100
neutral_pct = np.mean(np.abs(all_fc_values) < 1) * 100

categories = ['Strong\n(|fc| > 10)', 'Moderate\n(1 < |fc| < 10)', 'Neutral\n(|fc| < 1)']
counts = [
    np.mean(np.abs(all_fc_values) > 10) * 100,
    np.mean((np.abs(all_fc_values) >= 1) & (np.abs(all_fc_values) <= 10)) * 100,
    np.mean(np.abs(all_fc_values) < 1) * 100,
]
colors = ['#e74c3c', '#f39c12', '#3498db']
axes[1].bar(categories, counts, color=colors, edgecolor='white')
axes[1].set_ylabel('% of Time')
axes[1].set_title('Forecast Signal Strength Distribution')
axes[1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(counts):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Forecast turnover: how often does the signal change sign?
print("\nForecast Turnover Analysis:")

turnover_rates = []
for instr, fc in list(all_forecasts.items())[:30]:
    fc_clean = fc.dropna()
    if len(fc_clean) > 1:
        sign_changes = (fc_clean * fc_clean.shift(1) < 0).sum()
        turnover = sign_changes / len(fc_clean) * 100
        turnover_rates.append(turnover)

if turnover_rates:
    print(f"  Mean turnover (sign changes / days): {np.mean(turnover_rates):.1f}%")
    print(f"  Median turnover: {np.median(turnover_rates):.1f}%")
    print(f"  Implies average signal duration: {100 / np.mean(turnover_rates):.0f} days")

# %% [markdown]
# ## Predictive Power: Forecast vs Forward Returns
#
# We measure the Information Coefficient (IC) — the rank correlation between the forecast
# signal and forward log returns. For a mean-reversion strategy, we expect **negative IC**:
# a positive forecast (A is expensive) should predict negative forward returns.

# %%
print("=" * 60)
print("PREDICTIVE POWER: FORECAST vs FORWARD RETURNS")
print("=" * 60)

# Calculate IC for different forward horizons
horizons = [1, 5, 10, 20]  # days
ic_results = {h: [] for h in horizons}

# Sample instruments for IC calculation
sample_instruments = full_instruments[:100]

for instr in sample_instruments:
    try:
        fc = full_system.forecastScaleCap.get_capped_forecast(instr, "ah_spread")
        prices = data.get_raw_price(instr)

        if len(fc) < 30 or len(prices) < 30:
            continue

        # Align forecast and prices
        common_dates = fc.index.intersection(prices.index)
        fc_aligned = fc.loc[common_dates]
        prices_aligned = prices.loc[common_dates]

        # Calculate log returns
        log_prices = np.log(prices_aligned)

        for h in horizons:
            forward_returns = log_prices.shift(-h) - log_prices
            # Drop NaN
            valid = fc_aligned.dropna()
            fwd = forward_returns.loc[valid.index].dropna()
            fc_valid = valid.loc[fwd.index]

            if len(fwd) > 20:
                # Rank correlation (Spearman)
                ic, p_value = stats.spearmanr(fc_valid, fwd)
                if not np.isnan(ic):
                    ic_results[h].append(ic)

    except Exception:
        pass

# Print IC results
print(f"\nInformation Coefficient (Spearman Rank Correlation)")
print(f"Sample: {len(sample_instruments)} instruments")
print()
print(f"{'Horizon (days)':<18} {'Mean IC':>10} {'Std IC':>10} {'t-stat':>10} {'Hit Rate':>10}")
print("-" * 60)

for h in horizons:
    ics = np.array(ic_results[h])
    if len(ics) > 0:
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0
        # Hit rate: % of instruments with negative IC (as expected for mean-reversion)
        hit_rate = np.mean(ics < 0) * 100
        print(f"{h:<18} {mean_ic:>10.4f} {std_ic:>10.4f} {t_stat:>10.2f} {hit_rate:>9.1f}%")

# %%
# IC decay curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IC by horizon
mean_ics = [np.mean(ic_results[h]) for h in horizons]
std_ics = [np.std(ic_results[h]) for h in horizons]

axes[0].errorbar(horizons, mean_ics, yerr=std_ics, fmt='o-', color='steelblue', capsize=5)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Forward Horizon (days)')
axes[0].set_ylabel('Mean IC (Spearman)')
axes[0].set_title('IC Decay Curve: Forecast vs Forward Returns')
axes[0].grid(True, alpha=0.3)

# IC distribution for 1-day horizon
if len(ic_results[1]) > 0:
    axes[1].hist(ic_results[1], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=np.mean(ic_results[1]), color='green', linestyle='-', alpha=0.7,
                    label=f'Mean = {np.mean(ic_results[1]):.4f}')
    axes[1].set_xlabel('IC (Spearman)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('IC Distribution (1-Day Forward)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Rolling IC over time (60-day window)
print("\nRolling IC Analysis (60-day window, 1-day forward):")

# Pick a representative instrument for rolling IC
rep_instr = "601318.SH"
try:
    fc_rep = full_system.forecastScaleCap.get_capped_forecast(rep_instr, "ah_spread")
    prices_rep = data.get_raw_price(rep_instr)

    common = fc_rep.index.intersection(prices_rep.index)
    fc_a = fc_rep.loc[common]
    prices_a = prices_rep.loc[common]

    log_p = np.log(prices_a)
    fwd_1d = log_p.shift(-1) - log_p

    def rolling_spearman(x, window=60):
        result = pd.Series(np.nan, index=x.index)
        for i in range(window, len(x)):
            valid = pd.DataFrame({'fc': x.iloc[i-window:i], 'fwd': fwd_1d.iloc[i-window:i]}).dropna()
            if len(valid) > 10:
                result.iloc[i] = valid['fc'].rank().corr(valid['fwd'].rank())
        return result

    rolling_ic = rolling_spearman(fc_a, window=60)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_ic.index, rolling_ic.values, 'b-', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Rolling IC (60-day)')
    ax.set_xlabel('Date')
    ax.set_title(f'Rolling IC: {rep_instr} Forecast vs 1-Day Forward Return')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"  Mean rolling IC: {rolling_ic.mean():.4f}")
    print(f"  % of time IC < 0: {(rolling_ic < 0).mean() * 100:.1f}%")
except Exception as e:
    print(f"  Error: {e}")

# %% [markdown]
# ## Assumptions and Limitations
#
# This backtest makes several important assumptions that may affect real-world performance:
#
# 1. **No short-selling constraints:** A-shares can only be shorted via margin trading (融资融券),
#    which has limited stock availability and higher borrowing costs. H-shares have fewer restrictions
#    but may still face borrow constraints for smaller names.
#
# 2. **Transaction costs:** We use default pysystemtrade A-stock costs (~0.1%). Actual costs vary:
#    - A-shares: 0.1% stamp duty (sell only) + ~0.025% commission + market impact
#    - H-shares: 0.13% stamp duty (both sides) + 0.005% SFC levy + brokerage fees
#    - Stock Connect: Additional fees for cross-border trading
#
# 3. **FX risk unhedged:** H-share positions are in HKD. The portfolio reports in CNY.
#    HKD/CNY fluctuations add noise to H-share returns. The spread calculation uses
#    log returns (unitless) so FX doesn't affect the signal, but it does affect P&L.
#
# 4. **Survivorship bias:** The A+H universe includes only currently dual-listed stocks.
#    Pairs that delisted (e.g., H-share privatization, A-share delisting) are not included.
#
# 5. **Liquidity assumptions:** The backtest assumes all trades execute at the daily close price.
#    In reality, large orders may move the market, especially for smaller A+H names.
#
# 6. **No regime filter:** The mean-reversion signal works best in range-bound markets.
#    During strong trends (e.g., 2015 A-share crash, 2020 pandemic), the signal may
#    generate persistent losses as the spread continues to widen.
#
# 7. **Equal weighting:** All instruments receive equal notional allocation. This overweights
#    small-cap names and underweights large caps vs. a market-cap-weighted approach.
#
# 8. **Lookback sensitivity:** The 20-day lookback is arbitrary. Different lookbacks may
#    produce significantly different results. A walk-forward optimization would be needed
#    to find the optimal parameter.
#
# 9. **Data quality:** Forward-adjusted (qfq) prices are used. Corporate actions (dividends,
#    splits, rights issues) are handled by the adjustment factor, but special situations
#    may not be fully captured.
#
# 10. **No leverage constraints:** The forecast cap of ±20 implies maximum 2x vol-targeted
#     position size. Real accounts may have margin limits that restrict leverage.

# %%
