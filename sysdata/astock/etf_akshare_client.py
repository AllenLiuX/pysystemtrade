"""
ETFAkshareClient — ETF daily-price fetcher backed by akshare.

The xiximiao API proxy is stocks-only and does not return data for ETFs
(`pro.daily` and `pro.fund_daily` both return empty). For the all-weather
buckets we need ETF data, so we use akshare (free, public CN-data lib)
which natively supports ETFs via ``ak.fund_etf_hist_em``.

The client returns a DataFrame with the same column convention as the
xiximiao-fed daily store, so it can be passed straight into
``PGDailyPricesData.add_ohlcv``:

    columns = [open, high, low, close, volume, amount, pct_chg]
    index   = DatetimeIndex (date)

Usage:
    from sysdata.astock.etf_akshare_client import ETFAkshareClient
    client = ETFAkshareClient()
    df = client.fetch_daily("510300.SH", start="2015-01-01")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _strip_suffix(ts_code: str) -> str:
    """'510300.SH' -> '510300'  (akshare uses bare 6-digit codes for ETFs)."""
    return ts_code.split(".")[0]


class ETFAkshareClient:
    """Daily-bar ETF fetcher via akshare's East-Money endpoint."""

    DEFAULT_RATE_LIMIT_S = 0.4

    def __init__(self, rate_limit_s: float = DEFAULT_RATE_LIMIT_S):
        try:
            import akshare as ak  # noqa: F401  (import-check)
        except ImportError as e:
            raise ImportError(
                "akshare is required for ETF data fetch. "
                "Install with: pip install akshare"
            ) from e
        self._rate_limit_s = rate_limit_s
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit_s:
            time.sleep(self._rate_limit_s - elapsed)
        self._last_call = time.time()

    def fetch_daily(
        self,
        ts_code: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Fetch ETF daily OHLCV.

        Args:
            ts_code: '510300.SH' / '159934.SZ' (with exchange suffix)
            start:   'YYYY-MM-DD' or 'YYYYMMDD'; defaults to '20100101'
            end:     'YYYY-MM-DD' or 'YYYYMMDD'; defaults to today
            adjust:  '' (raw), 'qfq' (forward-adjusted), 'hfq' (back-adjusted)

        Returns:
            DataFrame with DatetimeIndex and columns
            [open, high, low, close, volume, amount, pct_chg].
            Empty DataFrame if no data found.
        """
        import akshare as ak

        symbol = _strip_suffix(ts_code)
        start_str = (start or "2010-01-01").replace("-", "")
        end_str = (end or datetime.now().strftime("%Y%m%d")).replace("-", "")

        self._throttle()
        try:
            raw = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=adjust,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("akshare fetch failed for %s: %s", ts_code, e)
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        # Map Chinese column names to canonical English names
        rename_map = {
            "日期": "dt",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }
        df = raw.rename(columns=rename_map)

        keep = ["dt", "open", "high", "low", "close",
                "volume", "amount", "pct_chg"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values("dt").set_index("dt")
        df.index.name = "DATETIME"
        return df


__all__ = ["ETFAkshareClient"]
