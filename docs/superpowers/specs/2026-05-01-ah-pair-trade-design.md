# Design: AH Pair Trade — Dollar-Neutral Positions

**Date:** 2026-05-01
**Status:** Draft

## Problem

The current AH spread trading system gives both A-share and H-share legs the **same forecast signal**, causing both legs to be long or short simultaneously. This is not a true pairs trade — it should be dollar-neutral (long one leg, short the other).

## Solution Overview

Two changes to achieve dollar-neutral pair trading:

1. **Signal flip for H shares** — H shares get negated z-scores so that after the trading rule applies its negation, A and H receive opposite-signed forecasts.
2. **Position normalization** — Post-process subsystem positions to enforce `position_H = -position_A` with equal magnitude.

## Architecture

### Change 1: Signal Direction Fix

**File:** `systems/ah_data.py`
**Method:** `get_ah_spread_zscore()`

**Current behavior:**
```python
return zscore  # Same value for both A and H
```

**New behavior:**
```python
if instrument_code.endswith(".HK"):
    return -zscore  # H share: flip sign
return zscore       # A share: keep as-is
```

**Signal flow after fix:**

| z-score | `get_ah_spread_zscore()` A | `get_ah_spread_zscore()` H | After `ah_spread()` negation A | After `ah_spread()` negation H |
|---------|---------------------------|---------------------------|-------------------------------|-------------------------------|
| +2.0    | +2.0                      | -2.0                      | -2.0 (short)                  | +2.0 (long)                   |
| -1.5    | -1.5                      | +1.5                      | +1.5 (long)                   | -1.5 (short)                  |

### Change 2: Position Normalization

**File:** `systems/ah_data.py`
**New method:** `get_ah_pair_normalized_position(instrument_code: str) -> pd.Series`

This method:
1. Gets the raw subsystem positions for both legs of the pair
2. Computes average absolute position: `pair_size = (|pos_A| + |pos_H|) / 2`
3. Enforces dollar-neutrality:
   - A-share position = `pair_size * sign(pos_A)`
   - H-share position = `-pair_size * sign(pos_A)` (opposite of A)

**Integration:** This method is called as a post-processing step, either:
- Directly in the backtest pipeline, or
- Wired into the system as a custom stage

**Design decision:** Add it as a method on `AHData` so it can be called from the backtest or any consumer. It does NOT replace the standard PositionSizing stage — it provides an alternative position that enforces pair neutrality.

### Change 3: Update Tests

**Files:**
- `systems/tests/test_ah_data.py`
- `systems/provided/rules/tests/test_ah_spread.py`

**New test cases:**
1. `test_h_share_zscore_is_negated` — verifies H shares get `-zscore`
2. `test_forecasts_are_opposite_signed` — after rule application, A and H forecasts have opposite signs
3. `test_pair_positions_are_dollar_neutral` — normalized positions satisfy `pos_A + pos_H ≈ 0`
4. `test_pair_positions_have_equal_magnitude` — `|pos_A| == |pos_H|`

### Change 4: Update Backtest Example

**File:** `examples/ah_spread_backtest.py`

Update:
- Assumptions section to reflect new pair-trading logic
- Comments explaining the signal flip and position normalization
- P&L decomposition to show pair-level results (long A / short H vs short A / long H)

## Data Flow

```
A-share prices ──→ spread ──→ zscore ──→ +zscore (A) / -zscore (H)
H-share prices ──→ (same) ──→ (same) ──→
                                              ↓
                                    ah_spread rule (-zscore)
                                              ↓
                              A: -zscore    H: +zscore  (opposite signs)
                                              ↓
                                    ForecastScaleCap
                                              ↓
                                  PositionSizing (raw)
                                              ↓
                            AHData.get_ah_pair_normalized_position()
                                              ↓
                        pos_A = avg(|pos_A|,|pos_H|) * sign(pos_A)
                        pos_H = -pos_A  (dollar-neutral)
```

## Edge Cases

1. **Missing data for one leg** — If either leg has no position data, return NaN for both.
2. **Zero position on both legs** — If both raw positions are zero, normalized position is zero.
3. **Sign disagreement** — If raw positions somehow have the same sign (shouldn't happen after signal fix), use A-share's sign as the canonical direction.

## Rollback Plan

The changes are additive:
- `get_ah_spread_zscore()` change modifies existing behavior — can be reverted by removing the sign flip
- `get_ah_pair_normalized_position()` is a new method — no impact on existing code unless explicitly called
