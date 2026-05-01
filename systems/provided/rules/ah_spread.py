"""
A+H spread mean-reversion trading rule.

Generates a forecast signal based on the z-score of the cumulative
log return spread between A-share and H-share dual-listed stocks.

When A-share outperforms H-share:
  - Spread rises, z-score positive
  - Signal is negative (sell A, buy H)

When A-share underperforms H-share:
  - Spread falls, z-score negative
  - Signal is positive (buy A, sell H)

Usage in config:
    trading_rules:
      ah_spread:
        function: systems.provided.rules.ah_spread.ah_spread
        data:
          - "ah_data.get_ah_spread_zscore"
        other_args:
          lookback: 20
"""

import pandas as pd


def ah_spread(zscore: pd.Series) -> pd.Series:
    """
    A+H spread mean-reversion rule.

    Returns the negative of the z-score as a mean-reversion signal.

    :param zscore: Rolling z-score of cumulative log return spread
                   (from ah_data.get_ah_spread_zscore)
    :returns: pd.Series -- unscaled forecast (mean-reversion)

    >>> import pandas as pd
    >>> zscore = pd.Series([1.5, 0.5, -0.5, -1.5], index=pd.date_range('2026-01-01', periods=4))
    >>> ah_spread(zscore)
    2026-01-01   -1.5
    2026-01-02   -0.5
    2026-01-03    0.5
    2026-01-04    1.5
    Freq: D, dtype: float64
    """
    return -zscore
