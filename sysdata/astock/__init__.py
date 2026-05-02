"""
A-stock data storage and access objects for pysystemtrade.

Mirrors the sysdata/csv/ and sysdata/futures/ patterns but for Chinese equities
sourced from xiximiao.com (Tushare-compatible API) or akshare (free backup).
"""

from sysdata.astock.akshare_client import (
    AkshareClient,
    to_sina_symbol,
    from_sina_symbol,
)
from sysdata.astock.fx_rate import AHFXRate

__all__ = [
    "AkshareClient",
    "to_sina_symbol",
    "from_sina_symbol",
    "AHFXRate",
]
