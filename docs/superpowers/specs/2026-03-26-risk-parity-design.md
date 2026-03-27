# Long Only Risk Parity Trading Rule - Design

## Overview

Implement a trading rule for long-only equal-weight risk parity strategy on A-Stock ETFs, using pysystemtrade's existing infrastructure.

## Goal

Create a complete trading rule + example that:
1. Always generates a long signal (forecast = 1.0)
2. Achieves equal risk contribution across instruments via pysystemtrade's vol-scaled position sizing
3. Uses 3 A-Stock ETFs: 510300.SH, 518880.SH, 511260.SH

## Instruments

| Code | Name | Type |
|------|------|------|
| 510300.SH | Huatai-Pb CSI 300 ETF | Equity |
| 518880.SH | Gold ETF | Commodity |
| 511260.SH | SSE Corporate Bond ETF | Fixed Income |

## Design

### Trading Rule Function

```python
# systems/provided/rules/long_only_risk_parity.py

def long_only_risk_parity_forecast(price):
    """
    Long only risk parity forecast - always bullish.
    
    Returns constant 1.0 forecast, allowing pysystemtrade's
    position sizing to handle equal risk weighting.
    
    :param price: pd.Series of prices
    :returns: pd.Series of forecasts (all 1.0)
    """
    return pd.Series(1.0, index=price.index)
```

### How Risk Parity Works

1. **Trading rule**: Always outputs `1.0` (long bias)
2. **Equal weights**: pysystemtrade uses `1/n` weight per instrument when no explicit weights configured
3. **Vol scaling**: `positionSize` stage divides notional by instrument volatility, achieving equal risk contribution

### Files

| File | Purpose |
|------|---------|
| `systems/provided/rules/long_only_risk_parity.py` | Trading rule function |
| `examples/risk_parity_example.py` | Usage example with A-Stock data |

## Implementation Steps

1. Create `systems/provided/rules/long_only_risk_parity.py` with `long_only_risk_parity_forecast` function
2. Create `examples/risk_parity_example.py` demonstrating usage with A-StockSimData
3. Add docstring with doctest examples

## Example Usage

```python
from sysdata.sim.astock_sim_data import AStockSimData
from systems.basesystem import System
from systems.rawdata import RawData
from systems.trading_rules import TradingRule

data = AStockSimData(instruments=["510300.SH", "518880.SH", "511260.SH"])
rule = TradingRule(long_only_risk_parity_forecast)

system = System([RawData(), rule], data=data)
```

## Verification

- All 3 instruments should have equal weights in portfolio
- Position sizes should be inversely proportional to volatility
- Forecast should be constant 1.0 for all instruments
