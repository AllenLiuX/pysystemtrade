"""
AH Premium Calculator — SSE Stock Connect Reference Exchange Rates

Uses the official SSE Stock Connect settlement rates (stock_sgt_reference_exchange_rate_sse)
to convert H-share prices (HKD) to CNY equivalents and calculate AH premium/discount.

Rate convention from SSE:
    参考汇率买入价: 1 CNY = X HKD (buying HKD with CNY)
    参考汇率卖出价: 1 CNY = X HKD (selling HKD for CNY)

To convert H-share price to CNY:
    price_cny = price_hkd / sell_rate

AH Premium formula:
    premium = (price_a_cny / price_h_cny) - 1
            = (price_a_cny / (price_h_hkd / sell_rate)) - 1

Usage:
    from sysdata.astock.fx_rate import AHFXRate

    fx = AHFXRate()

    # Get rate for a specific date
    rate = fx.get_hkd_sell_rate('2026-04-29')
    print(f'1 CNY = {rate} HKD  =>  1 HKD = {1/rate:.4f} CNY')

    # Calculate AH premium
    premium = fx.calc_ah_premium(
        a_price_cny=59.37,
        h_price_hkd=63.70,
        date='2026-04-29',
    )
    print(f'AH Premium: {premium:.1%}')
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AHFXRate:
    """
    SSE Stock Connect reference exchange rate manager.

    Fetches HKD/CNY rates once and caches them in memory.
    Uses the 卖出价 (sell rate) for H→CNY conversion, which is the
    rate actually applied when converting H-share proceeds back to CNY.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False

    def _load(self) -> pd.DataFrame:
        """Load SSE Stock Connect reference rates (cached)."""
        if self._loaded:
            return self._df

        import akshare as ak

        try:
            df = ak.stock_sgt_reference_exchange_rate_sse()
            df = df.rename(columns={
                '适用日期': 'date',
                '参考汇率买入价': 'buy_rate',
                '参考汇率卖出价': 'sell_rate',
                '货币种类': 'currency',
            })
            df['date'] = pd.to_datetime(df['date'])
            # Filter to HKD only
            df = df[df['currency'] == 'HKD'].copy()
            df = df.sort_values('date').reset_index(drop=True)
            self._df = df
            self._loaded = True
            logger.info(
                "Loaded SSE Stock Connect HKD rates: %d rows, %s to %s",
                len(df), df['date'].iloc[0].date(), df['date'].iloc[-1].date(),
            )
        except Exception as e:
            logger.error("Failed to load SSE Stock Connect rates: %s", e)
            self._df = pd.DataFrame()
            self._loaded = True

        return self._df

    def get_hkd_sell_rate(self, date: str) -> Optional[float]:
        """
        Get HKD sell rate for a given date.

        The sell rate represents: 1 CNY = X HKD
        This is the rate used when selling HKD to buy CNY.

        Args:
            date: '2026-04-29' or '20260429'

        Returns:
            Sell rate (HKD per 1 CNY), or None if not found.
            To get CNY per 1 HKD: 1 / sell_rate
        """
        df = self._load()
        if df.empty:
            return None

        target = pd.to_datetime(date)
        match = df[df['date'] == target]
        if not match.empty:
            return float(match.iloc[0]['sell_rate'])

        # Try previous trading day (rates not published on weekends/holidays)
        mask = df['date'] <= target
        available = df[mask]
        if available.empty:
            logger.warning("No HKD rate on or before %s", date)
            return None

        rate = float(available.iloc[-1]['sell_rate'])
        logger.info(
            "No rate for %s, using %s: %.4f",
            date, available.iloc[-1]['date'].date(), rate,
        )
        return rate

    def get_hkd_buy_rate(self, date: str) -> Optional[float]:
        """
        Get HKD buy rate for a given date.

        The buy rate represents: 1 CNY = X HKD
        This is the rate used when buying HKD with CNY.
        """
        df = self._load()
        if df.empty:
            return None

        target = pd.to_datetime(date)
        match = df[df['date'] == target]
        if not match.empty:
            return float(match.iloc[0]['buy_rate'])

        mask = df['date'] <= target
        available = df[mask]
        if available.empty:
            return None

        return float(available.iloc[-1]['buy_rate'])

    def hkd_to_cny(self, amount_hkd: float, date: str) -> Optional[float]:
        """
        Convert HKD amount to CNY using the sell rate.

        Args:
            amount_hkd: Amount in HKD
            date: Trade date

        Returns:
            Equivalent amount in CNY, or None if rate unavailable.
        """
        rate = self.get_hkd_sell_rate(date)
        if rate is None or rate == 0:
            return None
        return amount_hkd / rate

    def cny_to_hkd(self, amount_cny: float, date: str) -> Optional[float]:
        """
        Convert CNY amount to HKD using the buy rate.

        Args:
            amount_cny: Amount in CNY
            date: Trade date

        Returns:
            Equivalent amount in HKD, or None if rate unavailable.
        """
        rate = self.get_hkd_buy_rate(date)
        if rate is None or rate == 0:
            return None
        return amount_cny * rate

    def calc_ah_premium(
        self,
        a_price_cny: float,
        h_price_hkd: float,
        date: str,
    ) -> Optional[float]:
        """
        Calculate AH premium/discount.

        Args:
            a_price_cny: A-share closing price in CNY
            h_price_hkd: H-share closing price in HKD
            date: Trade date

        Returns:
            Premium as a decimal (0.15 = 15% premium, -0.10 = 10% discount).
            None if rate unavailable.

        Formula:
            h_price_cny = h_price_hkd / sell_rate
            premium = (a_price_cny / h_price_cny) - 1
        """
        h_price_cny = self.hkd_to_cny(h_price_hkd, date)
        if h_price_cny is None or h_price_cny == 0:
            return None
        return (a_price_cny / h_price_cny) - 1

    def get_rate_series(self) -> pd.DataFrame:
        """
        Get full HKD rate history as DataFrame.

        Returns:
            DataFrame with columns: date, buy_rate, sell_rate, currency
        """
        return self._load().copy()

    def refresh(self):
        """Force reload from API."""
        self._df = None
        self._loaded = False
        self._load()

    def get_h_code(self, a_code: str) -> Optional[str]:
        """
        Get H-share code for an A-share code.

        Args:
            a_code: '601318.SH'

        Returns:
            H-share code without exchange prefix, e.g. '02318'.
            None if not found.
        """
        from sysdata.astock.akshare_client import AkshareClient
        return AkshareClient.get_h_code(a_code)

    def calc_ah_premium_for_pair(
        self,
        a_code: str,
        a_price_cny: float,
        h_price_hkd: float,
        date: str,
    ) -> Optional[dict]:
        """
        Calculate AH premium with full context for a single pair.

        Args:
            a_code: A-share code, e.g. '601318.SH'
            a_price_cny: A-share closing price in CNY
            h_price_hkd: H-share closing price in HKD
            date: Trade date

        Returns:
            Dict with keys: a_code, h_code, a_price_cny, h_price_hkd,
                           h_price_cny, rate, premium, or None if unavailable.
        """
        h_code = self.get_h_code(a_code)
        if h_code is None:
            return None

        h_price_cny = self.hkd_to_cny(h_price_hkd, date)
        if h_price_cny is None:
            return None

        premium = self.calc_ah_premium(a_price_cny, h_price_hkd, date)
        rate = self.get_hkd_sell_rate(date)

        return {
            "a_code": a_code,
            "h_code": h_code,
            "a_price_cny": a_price_cny,
            "h_price_hkd": h_price_hkd,
            "h_price_cny": h_price_cny,
            "rate": rate,
            "premium": premium,
        }
