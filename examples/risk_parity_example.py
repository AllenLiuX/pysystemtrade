#!/usr/bin/env python
"""
Risk Parity Example - Long Only Equal Weight Strategy

Demonstrates using the long_only_risk_parity trading rule with A-Stock ETFs.

Instruments:
- 510300.SH: Huatai-PB CSI 300 ETF (Equity)
- 518880.SH: Gold ETF (Commodity)
- 511260.SH: SSE Corporate Bond ETF (Fixed Income)

Risk parity is achieved through:
1. Trading rule: constant 10.0 forecast (long bias, aligned with typical forecast range)
2. Equal weights: pysystemtrade uses 1/n when no instrument_weights configured
3. Vol scaling: positionSize stage divides notional by instrument volatility

Usage:
    python examples/risk_parity_example.py
"""

from sysdata.sim.astock_sim_data import AStockSimData
from systems.basesystem import System
from systems.rawdata import RawData
from systems.trading_rules import TradingRule
from systems.provided.rules.long_only_risk_parity import long_only_risk_parity_with_defaults


def create_risk_parity_system():
    """
    Create a pysystemtrade system with long only risk parity.
    
    Returns:
        System: Configured pysystemtrade system
    """
    instruments = ["510300.SH", "518880.SH", "511260.SH"]
    
    data = AStockSimData(instruments=instruments)
    rule = TradingRule(long_only_risk_parity_with_defaults)
    
    system = System([RawData(), rule], data=data)
    
    return system


def show_positions(system):
    """Display positions and weights."""
    print("=" * 60)
    print("Risk Parity Strategy - Long Only Equal Weight")
    print("=" * 60)
    
    print("\nInstruments:")
    for instr in system.get_instrument_list():
        print(f"  - {instr}")
    
    print("\nForecast (should be constant 10.0):")
    for instr in system.get_instrument_list():
        forecast = system.rules.get_raw_forecast(instr, "long_only_risk_parity_with_defaults")
        print(f"  {instr}: {forecast.iloc[-1]:.4f}")
    
    print("\nEqual weights (1/3 each):")
    weights = system.portfolio.get_instrument_weights()
    print(weights.tail())
    
    print("\nPosition sizes (vol-scaled):")
    for instr in system.get_instrument_list():
        pos = system.positionSize.get_actual_position(instr)
        print(f"  {instr}: {pos.iloc[-1]:.4f}")


if __name__ == "__main__":
    system = create_risk_parity_system()
    show_positions(system)
