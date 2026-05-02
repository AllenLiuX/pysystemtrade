"""
CnFuturesClient - Chinese futures data API client via akshare.

Wraps akshare futures endpoints:
    - futures_display_main_sina:  list all Sina continuous contracts
    - futures_contract_detail:    single contract specification details
    - futures_zh_daily_sina:      daily OHLCV + open interest + settlement
    - futures_contract_info_*:    individual contract calendars per exchange
    - futures_zh_realtime:        active contract discovery (DCE fallback)

No API token required - data sourced from Sina Finance.
"""

import logging
import time
import akshare as ak
import pandas as pd
from typing import Optional, List

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

    # ── Individual contract discovery ──────────────────────────────────

    def get_individual_contracts_shfe(self) -> List[str]:
        """Get SHFE individual contract codes (e.g. RB2410, AU2506)."""
        try:
            df = ak.futures_contract_info_shfe()
            codes = df["合约代码"].dropna().unique().tolist()
            return sorted(codes)
        except Exception as e:
            logger.warning("Failed SHFE contract info: %s", e)
            return []

    def get_individual_contracts_czce(self) -> List[str]:
        """
        Get CZCE individual contract codes converted to Sina format.
        Uses the `年份代码` and `月份代码` columns for reliable conversion.
        """
        try:
            df = ak.futures_contract_info_czce()
            if df is None or df.empty:
                return []
            codes = []
            for _, row in df.iterrows():
                raw = str(row.get("合约代码", ""))
                year = str(row.get("年份代码", ""))
                month = str(row.get("月份代码", ""))
                if not raw or not year or not month:
                    continue
                product = "".join(c for c in raw if c.isalpha())
                if not product:
                    continue
                yy = year[-2:]  # "2024" -> "24"
                mm = month.zfill(2)  # "3" -> "03"
                sina_code = f"{product}{yy}{mm}"
                if int(yy) >= 18:  # filter to 2018+
                    codes.append(sina_code)
            return sorted(set(codes))
        except Exception as e:
            logger.warning("Failed CZCE contract info: %s", e)
            return []

    def get_individual_contracts_cffex(self) -> List[str]:
        """Get CFFEX individual futures contracts only (filters out options)."""
        try:
            df = ak.futures_contract_info_cffex()
            if df is None or df.empty:
                return []
            codes = df["合约代码"].dropna().unique().tolist()
            # Filter out options (codes containing '-' like MO2412-P-6600)
            codes = [c for c in codes if "-" not in c]
            return sorted(codes)
        except Exception as e:
            logger.warning("Failed CFFEX contract info: %s", e)
            return []

    def get_individual_contracts_ine(self) -> List[str]:
        """Get INE individual contract codes."""
        try:
            df = ak.futures_contract_info_ine()
            codes = df["合约代码"].dropna().unique().tolist()
            return sorted(codes)
        except Exception as e:
            logger.warning("Failed INE contract info: %s", e)
            return []

    def get_individual_contracts_gfex(self) -> List[str]:
        """Get GFEX individual contract codes."""
        try:
            df = ak.futures_contract_info_gfex()
            codes = df["合约代码"].dropna().unique().tolist()
            return sorted(codes)
        except Exception as e:
            logger.warning("Failed GFEX contract info: %s", e)
            return []

    def get_individual_contracts_dce(self) -> List[str]:
        """
        Get DCE individual contract codes via futures_zh_realtime.
        Discovers contracts per product using the Chinese product names.
        """
        dce_products = [
            ("豆一",   "a"),   ("豆二",   "b"),   ("玉米",   "c"),
            ("淀粉",   "cs"),  ("苯乙烯", "eb"),  ("乙二醇", "eg"),
            ("纤维板", "fb"),  ("铁矿石", "i"),   ("焦炭",   "j"),
            ("鸡蛋",   "jd"),  ("焦煤",   "jm"),  ("塑料",   "l"),
            ("生猪",   "lh"),  ("原木",   "lg"),  ("豆粕",   "m"),
            ("棕榈油", "p"),   ("液化石油气", "pg"), ("聚丙烯", "pp"),
            ("粳米",   "rr"),  ("PVC",   "v"),   ("豆油",   "y"),
            ("纯苯",   "bz"),
        ]

        all_codes = []
        for name, _ in dce_products:
            try:
                rt = ak.futures_zh_realtime(symbol=name)
                if rt is not None and not rt.empty:
                    codes = rt["symbol"].dropna().unique().tolist()
                    all_codes.extend(codes)
            except Exception as e:
                logger.debug("Failed realtime for %s: %s", name, e)
            time.sleep(0.3)

        return sorted(set(all_codes))

    def get_all_individual_contracts(self) -> List[str]:
        """
        Get all individual contract codes across all exchanges.
        Returns a combined sorted list.
        """
        all_codes = []
        all_codes.extend(self.get_individual_contracts_shfe())
        all_codes.extend(self.get_individual_contracts_czce())
        all_codes.extend(self.get_individual_contracts_cffex())
        all_codes.extend(self.get_individual_contracts_ine())
        all_codes.extend(self.get_individual_contracts_gfex())
        all_codes.extend(self.get_individual_contracts_dce())
        return sorted(set(all_codes))

    def get_current_contracts(self) -> List[str]:
        """
        Get currently trading individual contracts across ALL exchanges
        via futures_zh_realtime. Covers all 86 products in futures_symbol_mark().
        Returns Sina-format codes (continuous + current front months).
        """
        try:
            products = ak.futures_symbol_mark()
        except Exception as e:
            logger.warning("Failed to get product list: %s", e)
            return []

        all_codes = []
        for _, row in products.iterrows():
            name = row["symbol"]  # Chinese product name
            try:
                rt = ak.futures_zh_realtime(symbol=name)
                if rt is not None and not rt.empty:
                    codes = rt["symbol"].dropna().unique().tolist()
                    all_codes.extend(codes)
            except Exception as e:
                logger.debug("Failed realtime for %s: %s", name, e)
            time.sleep(0.15)

        return sorted(set(all_codes))
