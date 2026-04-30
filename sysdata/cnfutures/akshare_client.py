"""
CnFuturesClient - Chinese futures data API client via akshare.

Wraps akshare futures endpoints:
    - futures_display_main_sina:  list all Sina continuous contracts
    - futures_contract_detail:    single contract specification details
    - futures_zh_daily_sina:      daily OHLCV + open interest + settlement

No API token required - data sourced from Sina Finance.
"""

import logging
import akshare as ak
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class CnFuturesClient:
    """Akshare-based Chinese futures data client."""

    def get_contract_list(self) -> Optional[pd.DataFrame]:
        """
        Get all available Sina futures continuous contracts.

        Returns:
            DataFrame with columns: symbol, exchange, name
            e.g. symbol='RB0', exchange='dce', name='Rebar continuous'
            Returns None on failure.
        """
        try:
            df = ak.futures_display_main_sina()
            return df
        except Exception as e:
            logger.warning("Failed to get contract list: %s", e)
            return None

    def get_contract_detail(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get contract specification details from Sina Finance.

        Args:
            symbol: contract code, e.g. 'RB2410', 'RB0'

        Returns:
            DataFrame with columns: item, value (key-value format)
            The 'item' column contains Chinese field names from the API.
            Use AKSHARE_FIELD_MAP from cnfutures_pg.py to map to English columns.
            Returns None on failure.
        """
        try:
            df = ak.futures_contract_detail(symbol=symbol)
            return df
        except Exception as e:
            logger.warning("Failed to get contract detail for %s: %s", symbol, e)
            return None

    def get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get daily OHLCV data for a futures contract from Sina Finance.

        Args:
            symbol: contract code, e.g. 'RB0' (continuous) or 'RB2410' (specific)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, hold, settle
            Sorted by date ascending. Returns None on failure.
        """
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if df is not None and not df.empty:
                df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning("Failed to get daily data for %s: %s", symbol, e)
            return None
