"""
AHData - A+H pair data stage for pysystemtrade.

Provides methods for computing log return spread and z-score
between A-share and H-share dual-listed stocks.

Usage:
    from systems.ah_data import AHData
    system = System([..., RawData(), AHData(), ...], data, config)
    spread = system.ah_data.get_ah_log_return_spread("601318.SH")
    zscore = system.ah_data.get_ah_spread_zscore("601318.SH", lookback=20)
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from systems.stage import SystemStage
from systems.system_cache import input, diagnostic, output


class AHData(SystemStage):
    """
    A+H pair data stage.

    Computes the cumulative difference of log returns between
    paired A-share and H-share instruments, and calculates
    rolling z-scores for mean-reversion signals.

    Name: ah_data
    """

    @property
    def name(self):
        return "ah_data"

    def _load_ah_mapping(self) -> Dict[str, str]:
        """
        Load A+H pair mapping into a bidirectional dict.

        Returns:
            Dict mapping instrument_code -> paired_instrument_code
            e.g. {'601318.SH': '02318.HK', '02318.HK': '601318.SH'}
        """
        from sysdata.astock.akshare_client import AkshareClient

        pairs = AkshareClient.get_ah_pairs()
        mapping = {}
        for a_code, h_code in pairs:
            h_full = h_code + ".HK"
            mapping[a_code] = h_full
            mapping[h_full] = a_code
        return mapping

    def get_ah_pair_code(self, instrument_code: str) -> Optional[str]:
        """
        Returns the paired instrument code for an A+H pair.

        If instrument is '601318.SH' (A-share), returns '02318.HK' (H-share).
        If instrument is '02318.HK' (H-share), returns '601318.SH' (A-share).
        Returns None if instrument is not part of an A+H pair.

        :param instrument_code: Instrument code to look up
        :returns: Paired instrument code, or None
        """
        mapping = self._load_ah_mapping()
        return mapping.get(instrument_code)

    def get_list_of_ah_instruments(self) -> List[str]:
        """
        Returns all A and H instruments that have a valid pair.

        :returns: List of instrument codes
        """
        mapping = self._load_ah_mapping()
        return sorted(mapping.keys())

    @input
    def get_ah_log_return_spread(self, instrument_code: str) -> pd.Series:
        """
        Cumulative sum of (log_ret_A - log_ret_H) on common trading days.

        Steps:
        1. Look up the paired instrument
        2. Fetch daily prices for both legs via rawdata
        3. Align to common trading days (intersection of both calendars)
        4. Compute daily log returns for both legs
        5. Compute spread = cumsum(log_ret_A - log_ret_H)

        :param instrument_code: A-share or H-share code
        :returns: pd.Series with DatetimeIndex, NaN for non-AH instruments
        """
        pair_code = self.get_ah_pair_code(instrument_code)
        if pair_code is None:
            return pd.Series(dtype=float)

        self.log.debug(
            "Computing AH spread for %s (pair: %s)",
            instrument_code, pair_code,
        )

        # Determine which leg is A and which is H
        if instrument_code.endswith(".HK"):
            h_code = instrument_code
            a_code = pair_code
        else:
            a_code = instrument_code
            h_code = pair_code

        # Fetch prices for both legs
        try:
            a_prices = self.parent.rawdata.get_daily_prices(a_code)
            h_prices = self.parent.rawdata.get_daily_prices(h_code)
        except Exception:
            return pd.Series(dtype=float)

        if a_prices.empty or h_prices.empty:
            return pd.Series(dtype=float)

        # Ensure both are pd.Series
        if isinstance(a_prices, pd.DataFrame):
            a_prices = a_prices.iloc[:, 0]
        if isinstance(h_prices, pd.DataFrame):
            h_prices = h_prices.iloc[:, 0]

        # Align to common trading days
        common_dates = a_prices.index.intersection(h_prices.index)
        if len(common_dates) < 2:
            return pd.Series(dtype=float)

        a_aligned = a_prices.loc[common_dates]
        h_aligned = h_prices.loc[common_dates]

        # Compute log returns
        a_log_ret = np.log(a_aligned).diff()
        h_log_ret = np.log(h_aligned).diff()

        # Compute spread = cumulative sum of (log_ret_A - log_ret_H)
        spread = (a_log_ret - h_log_ret).cumsum()

        # Drop initial NaN from diff()
        spread = spread.dropna()

        return spread

    @input
    def get_ah_spread_zscore(self, instrument_code: str, lookback: int = 20) -> pd.Series:
        """
        Rolling z-score of the spread level.

        z = (spread - rolling_mean(spread, lookback)) / rolling_std(spread, lookback)

        :param instrument_code: A-share or H-share code
        :param lookback: Rolling window in days (default: 20)
        :returns: pd.Series with DatetimeIndex, NaN for non-AH instruments
                  or when insufficient lookback data is available
        """
        spread = self.get_ah_log_return_spread(instrument_code)

        if spread.empty:
            return pd.Series(dtype=float)

        self.log.debug(
            "Computing AH z-score for %s (lookback: %d)",
            instrument_code, lookback,
        )

        # Compute rolling mean and std
        rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()

        # Compute z-score
        zscore = (spread - rolling_mean) / rolling_std

        return zscore
