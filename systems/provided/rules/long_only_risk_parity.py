from typing import Optional
import pandas as pd


def long_only_risk_parity_forecast(
    price: pd.Series,
    long_only_scalar: float = 10.0
) -> pd.Series:
    """
    Long only risk parity forecast - always bullish.

    Returns constant forecast value (10.0), aligned with typical forecast
    range for proper scaling with pysystemtrade's forecastScaleCap.

    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.Series

    :param long_only_scalar: The scalar to use for long positions (default 10.0)
    :type long_only_scalar: float

    :returns: pd.Series -- constant forecast

    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.date_range('2024-01-01', periods=5, freq='D')
    >>> price = pd.Series([100.0, 101.0, 102.0, 101.5, 103.0], index=idx)
    >>> result = long_only_risk_parity_forecast(price)
    >>> len(result) == 5
    True
    >>> all(result == 10.0)
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
    >>> all(result == 10.0)
    True
    """
    return long_only_risk_parity_forecast(price)
