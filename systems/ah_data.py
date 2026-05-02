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

    def __init__(self):
        self._ah_mapping = None
        self._price_cache = {}
        self._spread_cache = {}

    def _get_cached_price(self, instrument_code: str) -> pd.Series:
        """
        Cache daily prices to avoid redundant DB queries.
        Each instrument's price is fetched only once per AHData instance.
        """
        if instrument_code not in self._price_cache:
            self._price_cache[instrument_code] = self.parent.rawdata.get_daily_prices(instrument_code)
        return self._price_cache[instrument_code]

    def _get_spread_for_pair(self, a_code: str, h_code: str) -> pd.Series:
        """
        Compute spread for a pair once, cache it so both legs share the result.
        Uses a normalized pair key so (A,H) and (H,A) map to the same cache entry.
        """
        pair_key = tuple(sorted([a_code, h_code]))
        if pair_key not in self._spread_cache:
            try:
                a_prices = self._get_cached_price(a_code)
                h_prices = self._get_cached_price(h_code)
            except Exception as e:
                self.log.warning("Failed to fetch prices for %s/%s: %s", a_code, h_code, e)
                self._spread_cache[pair_key] = pd.Series(dtype=float)
                return self._spread_cache[pair_key]

            if a_prices.empty or h_prices.empty:
                self._spread_cache[pair_key] = pd.Series(dtype=float)
                return self._spread_cache[pair_key]

            if isinstance(a_prices, pd.DataFrame):
                a_prices = a_prices.iloc[:, 0]
            if isinstance(h_prices, pd.DataFrame):
                h_prices = h_prices.iloc[:, 0]

            common_dates = a_prices.index.intersection(h_prices.index)
            if len(common_dates) < 2:
                self._spread_cache[pair_key] = pd.Series(dtype=float)
                return self._spread_cache[pair_key]

            a_aligned = a_prices.loc[common_dates]
            h_aligned = h_prices.loc[common_dates]

            a_log_ret = np.log(a_aligned).diff()
            h_log_ret = np.log(h_aligned).diff()

            spread = (a_log_ret - h_log_ret).cumsum().dropna()
            self._spread_cache[pair_key] = spread

        return self._spread_cache[pair_key]

    def _load_ah_mapping(self) -> Dict[str, str]:
        """
        Load A+H pair mapping into a bidirectional dict.

        Returns:
            Dict mapping instrument_code -> paired_instrument_code
            e.g. {'601318.SH': '02318.HK', '02318.HK': '601318.SH'}
        """
        if self._ah_mapping is not None:
            return self._ah_mapping
        from sysdata.astock.akshare_client import AkshareClient

        pairs = AkshareClient.get_ah_pairs()
        mapping = {}
        for a_code, h_code in pairs:
            h_full = h_code + ".HK"
            mapping[a_code] = h_full
            mapping[h_full] = a_code
        self._ah_mapping = mapping
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

    @diagnostic()
    def get_ah_log_return_spread(self, instrument_code: str) -> pd.Series:
        """
        Cumulative sum of (log_ret_A - log_ret_H) on common trading days.

        Uses internal price cache and pair-level spread cache to avoid
        redundant DB queries and duplicate computations.

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

        if instrument_code.endswith(".HK"):
            h_code = instrument_code
            a_code = pair_code
        else:
            a_code = instrument_code
            h_code = pair_code

        return self._get_spread_for_pair(a_code, h_code)

    @diagnostic()
    def get_ah_spread_zscore(self, instrument_code: str, lookback: int = 20) -> pd.Series:
        """
        Rolling z-score of the spread level.

        z = (spread - rolling_mean(spread, lookback)) / rolling_std(spread, lookback)

        For H-shares, the z-score is negated so that after the ah_spread rule
        applies its negation, A and H shares receive opposite-signed forecasts.

        :param instrument_code: A-share or H-share code
        :param lookback: Rolling window in days (default: 20)
        :returns: pd.Series with DatetimeIndex, NaN for non-AH instruments
        """
        spread = self.get_ah_log_return_spread(instrument_code)

        if spread.empty:
            return pd.Series(dtype=float)

        self.log.debug(
            "Computing AH z-score for %s (lookback: %d)",
            instrument_code, lookback,
        )

        rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()

        zscore = (spread - rolling_mean) / rolling_std

        if instrument_code.endswith(".HK"):
            return -zscore
        return zscore
