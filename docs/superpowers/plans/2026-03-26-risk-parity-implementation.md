# Long Only Risk Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a long-only risk parity trading rule that always outputs a constant 1.0 forecast, allowing pysystemtrade's position sizing to handle equal risk weighting across 3 A-Stock ETFs.

**Architecture:** Single trading rule function that returns constant 1.0, relying on pysystemtrade's built-in equal weight (when no instrument_weights configured) and vol-scaled position sizing.

**Tech Stack:** pysystemtrade, pandas, AStockSimData

---

## File Structure

| File | Purpose |
|------|---------|
| `systems/provided/rules/long_only_risk_parity.py` | Trading rule function |
| `examples/risk_parity_example.py` | Usage example with A-StockSimData |

---

## Task 1: Create Trading Rule Function

**Files:**
- Create: `systems/provided/rules/long_only_risk_parity.py`
- Test: `systems/tests/test_rules.py` (existing file - add test)

- [ ] **Step 1: Create `systems/provided/rules/long_only_risk_parity.py`**

```python
from typing import Optional
import pandas as pd


def long_only_risk_parity_forecast(
    price: pd.Series,
    long_only_scalar: float = 1.0
) -> pd.Series:
    """
    Long only risk parity forecast - always bullish.

    Returns constant forecast value, allowing pysystemtrade's
    position sizing to handle equal risk weighting across instruments.

    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.Series

    :param long_only_scalar: The scalar to use for long positions (default 1.0)
    :type long_only_scalar: float

    :returns: pd.Series -- constant forecast

    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.date_range('2024-01-01', periods=5, freq='D')
    >>> price = pd.Series([100.0, 101.0, 102.0, 101.5, 103.0], index=idx)
    >>> result = long_only_risk_parity_forecast(price)
    >>> len(result) == 5
    True
    >>> all(result == 1.0)
    True
    """
    return pd.Series(long_only_scalar, index=price.index)


def long_only_risk_parity_with_defaults(price):
    """
    Wrapper with no additional arguments for use as TradingRule.

    :param price: The price or other series to use
    :type price: pd.Series

    :returns: pd.Series -- constant forecast

    >>> import pandas as pd
    >>> idx = pd.date_range('2024-01-01', periods=3, freq='D')
    >>> price = pd.Series([100.0, 101.0, 102.0], index=idx)
    >>> result = long_only_risk_parity_with_defaults(price)
    >>> all(result == 1.0)
    True
    """
    return long_only_risk_parity_forecast(price)
```

- [ ] **Step 2: Add test to existing test file**

Check if `systems/tests/test_rules.py` exists or look for appropriate test location:

```bash
# Check for existing rules tests
ls systems/tests/test_*.py | grep -i rule
```

- [ ] **Step 3: Run tests to verify implementation**

```bash
cd pysystemtrade
pytest systems/provided/rules/long_only_risk_parity.py -v
```

---

## Task 2: Create Example Script

**Files:**
- Create: `examples/risk_parity_example.py`

- [ ] **Step 1: Create `examples/risk_parity_example.py`**

```python
#!/usr/bin/env python
"""
Risk Parity Example - Long Only Equal Weight Strategy

Demonstrates using the long_only_risk_parity trading rule with A-Stock ETFs.

Instruments:
- 510300.SH: Huatai-PB CSI 300 ETF (Equity)
- 518880.SH: Gold ETF (Commodity)
- 511260.SH: SSE Corporate Bond ETF (Fixed Income)

Risk parity is achieved through:
1. Trading rule: constant 1.0 forecast (long bias)
2. Equal weights: pysystemtrade uses 1/n when no instrument_weights configured
3. Vol scaling: positionSize stage divides notional by instrument volatility

Usage:
    python examples/risk_parity_example.py
"""

from systems.sim.astock_sim_data import AStockSimData
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
    
    print("\nForecast (should be constant 1.0):")
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
```

---

## Task 3: Verify Integration

- [ ] **Step 1: Verify imports work**

```bash
cd pysystemtrade
python -c "from systems.provided.rules.long_only_risk_parity import long_only_risk_parity_forecast; print('OK')"
```

- [ ] **Step 2: Run example script**

```bash
cd pysystemtrade
python examples/risk_parity_example.py
```

---

## Summary

| Task | Files | Status |
|------|-------|--------|
| 1 | `systems/provided/rules/long_only_risk_parity.py` | ☐ |
| 2 | `examples/risk_parity_example.py` | ☐ |
| 3 | Integration verification | ☐ |
