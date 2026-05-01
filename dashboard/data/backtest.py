"""
Backtest engine for the risk parity dashboard.

Mirrors the logic from examples/risk_parity_example.py:
- Rolling inverse-volatility weights (monthly recalculation, 63-day lookback)
- Long-only trading rule with constant forecast
- Vol-scaled position sizing
- Equal-weight comparison
"""

import logging
import numpy as np
import pandas as pd

from systems.portfolio import Portfolios

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 242


def run_backtest(
    instruments: list[str] = None,
    capital: float = 1_000_000,
    vol_lookback: int = 63,
    max_history_years: float = 3.0,
) -> dict:
    """
    Run the rolling risk parity backtest.

    Args:
        max_history_years: Limit data fetch to this many years (default 3).
            Set to 0 for unlimited (slower).
    """
    if instruments is None:
        instruments = ["510300.SH", "518880.SH", "511260.SH"]
    from sysdata.sim.astock_sim_data import AStockSimData
    from sysdata.config.configdata import Config
    from systems.basesystem import System
    from systems.rawdata import RawData
    from systems.trading_rules import TradingRule
    from systems.forecasting import Rules
    from systems.provided.rules.long_only import long_only
    from systems.forecast_scale_cap import ForecastScaleCap
    from systems.forecast_combine import ForecastCombine
    from systems.positionsizing import PositionSizing
    from systems.accounts.accounts_stage import Account

    data = AStockSimData()

    # Verify instruments exist
    available = data.get_instrument_list()
    valid_instruments = [i for i in instruments if i in available]
    if not valid_instruments:
        raise ValueError(f"No valid instruments found from {instruments}")

    config = Config()
    config.instruments = valid_instruments
    config.notional_trading_capital = capital

    # Find common start date — latest first date among all instruments
    common_start = None
    for instr in valid_instruments:
        prices = data.get_raw_price(instr)
        if not prices.empty:
            first = prices.index[0]
            common_start = max(common_start, first) if common_start else first

    rules = Rules({"long_only": TradingRule(long_only)})

    # Calculate rolling risk parity weights
    daily_weights = _calc_rolling_weights(data, valid_instruments, vol_lookback)

    portfolio = RollingRiskParityPortfolio(precomputed_weights=daily_weights)

    stages = [
        Account(),
        ForecastScaleCap(),
        rules,
        ForecastCombine(),
        PositionSizing(),
        portfolio,
        RawData(),
    ]
    system = System(stages, data=data, config=config)

    # Extract metrics
    equity_curve = system.accounts.portfolio().curve() + capital

    # Trim to common start date (when all instruments have data)
    if common_start is not None:
        equity_curve = equity_curve[equity_curve.index >= common_start]

    # Compute equal-weight equity analytically from per-instrument P&L
    # Equal-weight = mean of N equally-weighted instrument returns, compounded from capital
    instr_daily_returns = {}
    for instr in valid_instruments:
        instr_curve = system.accounts.pandl_for_instrument(instr).curve()
        if len(instr_curve) > 0:
            instr_capital = capital / len(valid_instruments)
            instr_equity = instr_curve + instr_capital
            instr_daily_returns[instr] = instr_equity.pct_change().dropna()

    if instr_daily_returns:
        returns_df = pd.DataFrame(instr_daily_returns).dropna()
        daily_portfolio_return = returns_df.mean(axis=1)
        # Normalize cumulative product to start at 1.0, then scale to capital
        cum_prod = (1 + daily_portfolio_return).cumprod()
        cum_prod = cum_prod / cum_prod.iloc[0]
        # Prepend 1.0 at the start of equity_curve index so ffill covers all dates
        start_date = equity_curve.index[0]
        cum_prod = pd.concat([pd.Series([1.0], index=[start_date]), cum_prod])
        cum_prod = cum_prod[~cum_prod.index.duplicated(keep="first")]
        equal_equity = cum_prod.reindex(equity_curve.index).ffill() * capital
    else:
        equal_equity = equity_curve.copy()

    returns = equity_curve.pct_change().dropna()
    equal_returns = equal_equity.pct_change().dropna()

    performance = _calc_performance(equity_curve, returns)
    equal_performance = _calc_performance(equal_equity, equal_returns)

    # Per-instrument metrics
    instr_metrics = {}
    raw_prices = {}
    for instr in valid_instruments:
        prices = data.get_raw_price(instr)
        raw_prices[instr] = prices[prices.index >= common_start] if common_start else prices
        instr_curve = system.accounts.pandl_for_instrument(instr).curve()
        if len(instr_curve) > 0:
            instr_capital = capital / len(valid_instruments)
            instr_equity = instr_curve + instr_capital
            instr_returns = instr_equity.pct_change().dropna()
            instr_metrics[instr] = _calc_performance(instr_equity, instr_returns)

    # Volatility
    vol_data = _calc_volatility(system, valid_instruments, equity_curve, capital)

    # Equal-weight weights (constant 1/N for each instrument)
    ew_weights = _calc_equal_weights(equity_curve.index, valid_instruments)

    # Volume data for price charts
    volume_data = _get_volume_data(data, valid_instruments, common_start)

    return {
        "equity_curve": equity_curve,
        "equal_weight_equity": equal_equity,
        "weights": system.portfolio.get_instrument_weights(),
        "rolling_weights": _extract_rolling_recalc_points(daily_weights, valid_instruments),
        "equal_weight_weights": ew_weights,
        "volatility": vol_data,
        "performance": performance,
        "equal_performance": equal_performance,
        "instruments": instr_metrics,
        "positions": _get_positions(system, valid_instruments),
        "forecasts": _get_forecasts(system, valid_instruments),
        "raw_prices": raw_prices,
        "volume_data": volume_data,
    }


class RollingRiskParityPortfolio(Portfolios):
    """Portfolio stage with precomputed rolling risk parity weights."""

    def __init__(self, precomputed_weights: pd.DataFrame = None):
        super().__init__()
        self._precomputed_weights = precomputed_weights

    def get_unsmoothed_raw_instrument_weights(self) -> pd.DataFrame:
        """Override to inject precomputed inverse-vol weights into the Portfolios pipeline."""
        if self._precomputed_weights is None:
            return super().get_unsmoothed_raw_instrument_weights()

        subsystem_positions = self._get_all_subsystem_positions()
        position_series_index = subsystem_positions.index

        weights = self._precomputed_weights.reindex(position_series_index).ffill()

        instrument_list = self.get_instrument_list()
        weights = weights[instrument_list]
        return weights


def _calc_rolling_weights(data, instruments: list[str], vol_lookback: int) -> pd.DataFrame:
    """Calculate rolling inverse-volatility weights with monthly recalculation."""
    price_dict = {}
    for instr in instruments:
        price_dict[instr] = data.get_raw_price(instr)

    returns_dict = {instr: prices.pct_change().dropna() for instr, prices in price_dict.items()}
    returns_df = pd.DataFrame(returns_dict).dropna()

    all_dates = returns_df.index
    monthly_dates = all_dates.to_frame().set_index(
        all_dates.to_period("M").to_timestamp()
    ).index.unique()

    rolling_weights_list = []
    for i, month_end in enumerate(monthly_dates):
        if i == 0:
            continue

        lookback_start_idx = max(0, i - int(vol_lookback / 21))
        lookback_start = monthly_dates[lookback_start_idx]
        lookback_returns = returns_df.loc[lookback_start:month_end]

        if len(lookback_returns) < 20:
            continue

        vol_dict = {}
        for instr in instruments:
            instr_returns = lookback_returns[instr].dropna()
            if len(instr_returns) > 0:
                ann_vol = instr_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                vol_dict[instr] = ann_vol

        if len(vol_dict) == len(instruments):
            inv_vols = {instr: 1 / vol for instr, vol in vol_dict.items()}
            total = sum(inv_vols.values())
            weights = {instr: w / total for instr, w in inv_vols.items()}
            rolling_weights_list.append({"date": month_end, **weights})

    rolling_weights_df = pd.DataFrame(rolling_weights_list).set_index("date")
    daily_weights = rolling_weights_df.reindex(all_dates).ffill().dropna()
    return daily_weights


def _calc_performance(equity_curve: pd.Series, returns: pd.Series) -> dict:
    """Calculate performance metrics from equity curve."""
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
        }

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    negative_returns = returns[returns < 0]
    downside_dev = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if len(negative_returns) > 0 else 0
    sortino = annualized_return / downside_dev if downside_dev > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return": round(total_return, 2),
        "annualized_return": round(annualized_return, 2),
        "annualized_vol": round(annualized_vol, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_drawdown": round(max_drawdown, 2),
    }


def _calc_volatility(system, instruments, equity_curve, capital) -> pd.DataFrame:
    """Calculate rolling volatility for portfolio and instruments."""
    window = 21
    portfolio_returns = equity_curve.pct_change().dropna()
    portfolio_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    vol_dict = {"portfolio": portfolio_vol}
    for instr in instruments:
        try:
            instr_curve = system.accounts.pandl_for_instrument(instr).curve()
            instr_capital = capital / len(instruments)
            instr_equity = instr_curve + instr_capital
            instr_returns = instr_equity.pct_change().dropna()
            instr_vol = instr_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
            vol_dict[instr] = instr_vol
        except Exception:
            pass

    return pd.DataFrame(vol_dict)


def _extract_rolling_recalc_points(daily_weights, instruments) -> pd.DataFrame:
    """Extract the monthly recalculation points from daily weights."""
    if daily_weights.empty:
        return daily_weights

    # Find points where weights changed (monthly recalculation)
    changed = daily_weights.diff().abs().sum(axis=1) > 1e-10
    recalc_dates = daily_weights[changed]
    if recalc_dates.empty:
        return daily_weights.iloc[0:1]
    return recalc_dates


def _get_positions(system, instruments) -> pd.DataFrame:
    """Get vol-scaled positions for all instruments."""
    positions = {}
    for instr in instruments:
        try:
            pos = system.portfolio.get_actual_position(instr)
            positions[instr] = pos
        except Exception:
            pass
    if positions:
        return pd.DataFrame(positions)
    return pd.DataFrame()


def _get_forecasts(system, instruments) -> pd.DataFrame:
    """Get raw forecasts for all instruments."""
    forecasts = {}
    for instr in instruments:
        try:
            fc = system.rules.get_raw_forecast(instr, "long_only")
            forecasts[instr] = fc
        except Exception:
            pass
    if forecasts:
        return pd.DataFrame(forecasts)
    return pd.DataFrame()


def _calc_equal_weights(dates: pd.DatetimeIndex, instruments: list[str]) -> pd.DataFrame:
    """Generate constant equal-weight (1/N) series aligned to given dates."""
    n = len(instruments)
    weights = {instr: [1.0 / n] * len(dates) for instr in instruments}
    return pd.DataFrame(weights, index=dates)


def _get_volume_data(data, instruments: list[str], common_start) -> dict:
    """Get volume time series for each instrument."""
    volume = {}
    for instr in instruments:
        try:
            ohlcv = data.get_ohlcv(instr)
            if not ohlcv.empty and "volume" in ohlcv.columns:
                vol = ohlcv["volume"]
                if common_start is not None:
                    vol = vol[vol.index >= common_start]
                volume[instr] = vol
        except Exception:
            pass
    return volume
