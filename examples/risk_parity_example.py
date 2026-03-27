#!/usr/bin/env python
"""
Risk Parity Example - Long Only Equal Weight Strategy

Demonstrates using the long_only trading rule with A-Stock ETFs.

Instruments:
- 510300.SH: Huatai-PB CSI 300 ETF (Equity)
- 518880.SH: Gold ETF (Commodity)
- 511260.SH: SSE Corporate Bond ETF (Fixed Income)

Risk parity is achieved through:
1. Trading rule: constant 10.0 forecast (long bias, aligned with typical forecast range)
2. Equal weights: pysystemtrade uses 1/n when no instrument_weights configured
3. Vol scaling: positionSize stage divides notional by instrument volatility

Prerequisites:
- Data must be fetched first using the fetcher:
    python -m sysinit.astock.fetcher --once --freq daily --universe all
- Or use parquet files in data/astock/daily_prices_parquet/

Usage:
    python examples/risk_parity_example.py
"""

from sysdata.sim.astock_sim_data import AStockSimData
from systems.basesystem import System
from systems.rawdata import RawData
from systems.trading_rules import TradingRule
from systems.provided.rules.long_only import long_only


TARGET_INSTRUMENTS = ["510300.SH", "518880.SH", "511260.SH"]


def create_risk_parity_system():
    """
    Create a pysystemtrade system with long only risk parity.
    
    Returns:
        System: Configured pysystemtrade system
    """
    data = AStockSimData()
    rule = TradingRule(long_only)
    
    system = System([RawData(), rule], data=data)
    
    return system


def filter_to_target_instruments(system):
    """Filter portfolio to only target instruments."""
    all_instruments = system.get_instrument_list()
    available = [i for i in TARGET_INSTRUMENTS if i in all_instruments]
    missing = [i for i in TARGET_INSTRUMENTS if i not in all_instruments]
    
    if missing:
        print(f"Warning: Instruments not in data store: {missing}")
    
    return available


def show_positions(system, instruments):
    """Display positions and weights."""
    print("=" * 60)
    print("Risk Parity Strategy - Long Only Equal Weight")
    print("=" * 60)
    
    print("\nInstruments:")
    for instr in instruments:
        print(f"  - {instr}")
    
    print("\nForecast (should be constant 10.0):")
    for instr in instruments:
        forecast = system.rules.get_raw_forecast(instr, "long_only")
        print(f"  {instr}: {forecast.iloc[-1]:.4f}")
    
    print("\nEqual weights (1/3 each):")
    weights = system.portfolio.get_instrument_weights()
    # Filter to target instruments
    if all(i in weights.columns for i in instruments):
        print(weights[instruments].tail())
    
    print("\nPosition sizes (vol-scaled):")
    for instr in instruments:
        pos = system.positionSize.get_actual_position(instr)
        print(f"  {instr}: {pos.iloc[-1]:.4f}")


if __name__ == "__main__":
    system = create_risk_parity_system()
    instruments = filter_to_target_instruments(system)
    
    if not instruments:
        print("Error: None of the target instruments are available in the data store.")
        print("Run the fetcher first to download data:")
        print("  python -m sysinit.astock.fetcher --once --freq daily --universe all")
    else:
        show_positions(system, instruments)
