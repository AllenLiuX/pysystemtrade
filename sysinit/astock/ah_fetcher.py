"""
A+H 双市数据拉取器

从 akshare 拉取 A+H 双市日线数据，分别存入 A股和H股价格存储。
支持增量更新、全市场扫描、AH溢价计算。

用法:
    # 一次性拉取 (测试)
    python -m sysinit.astock.ah_fetcher --once --universe test

    # 一次性拉取全量 A+H
    python -m sysinit.astock.ah_fetcher --once --universe ah --years 5

    # 指定品种
    python -m sysinit.astock.ah_fetcher --once --symbols 601318.SH 600036.SH
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from sysdata.astock.akshare_client import AkshareClient
from sysdata.astock.astock_prices import AStockDailyPricesData
from sysdata.astock.astock_hk_prices import HKStockDailyPricesData
from sysdata.astock.astock_pg import PGDailyPricesData, PGHKDailyPricesData
from sysdata.astock.universe import AStockUniverse
from sysdata.astock.china_calendar import ChinaMarketCalendar
from sysdata.astock.fx_rate import AHFXRate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("astock.ah_fetcher")


# ── 数据标准化 ──────────────────────────────────────────────────

def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准 OHLCV DataFrame (DatetimeIndex)

    输入: ts_code, trade_date, open, high, low, close, vol, amount
    输出: open, high, low, close, volume, amount, price (index=DatetimeIndex)
    """
    if df.empty:
        return df
    df = df.copy()
    df["DATETIME"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.set_index("DATETIME").sort_index()
    df["volume"] = df["vol"] * 100  # 手 → 股
    df["price"] = df["close"]
    cols = ["open", "high", "low", "close", "volume", "amount", "price"]
    return df[[c for c in cols if c in df.columns]]


# ── Fetcher ─────────────────────────────────────────────────────

class AHStockFetcher:
    """
    A+H 双市数据拉取器

    功能:
      - 增量拉取 A股和H股日线
      - 全市场扫描发现有效品种
      - 交易日历感知调度
      - AH溢价计算
    """

    DEFAULT_LOOKBACK_DAYS = 365 * 5

    def __init__(
        self,
        symbols: List[str],
        client: AkshareClient = None,
        initial_years: float = 5.0,
        backend: str = "parquet",
    ):
        self.symbols = symbols
        self.client = client or AkshareClient(adjust="qfq")
        self.calendar = ChinaMarketCalendar
        self.backend = backend

        if backend == "pg":
            self._a_store = PGDailyPricesData()
            self._h_store = PGHKDailyPricesData()
        else:
            self._a_store = AStockDailyPricesData()
            self._h_store = HKStockDailyPricesData()

        self._fx = AHFXRate()

        if initial_years != 5.0:
            self.DEFAULT_LOOKBACK_DAYS = int(365 * initial_years)

        self._fetch_count = 0
        self._error_count = 0

        logger.info("AHStockFetcher initialized:")
        logger.info("  A+H Universe: %d symbols", len(self.symbols))
        logger.info("  Adjust: %s", self.client.adjust)
        logger.info("  Backend: %s", backend)

    def fetch_symbol_pair(
        self, a_code: str, force_start: datetime = None
    ) -> Dict[str, int]:
        """
        增量拉取单个A+H配对。

        Returns:
            {'a': new_rows_a, 'h': new_rows_h}
        """
        h_code = self.client.get_h_code(a_code)
        if not h_code:
            logger.warning("No H-share mapping for %s", a_code)
            return {"a": 0, "h": 0}

        now = datetime.now()

        # ── A-share ──
        latest_a = self._a_store.get_latest_date(a_code)
        if force_start:
            start_a = force_start
        elif latest_a:
            start_a = (latest_a - timedelta(days=3)).to_pydatetime()
        else:
            start_a = now - timedelta(days=self.DEFAULT_LOOKBACK_DAYS)

        raw_a = self.client.fetch_daily_range(
            a_code, start_a, now
        )
        new_a = 0
        if not raw_a.empty:
            df_a = _normalize_daily(raw_a)
            new_a = self._a_store.append_prices(a_code, df_a)

        # ── H-share ──
        h_code_full = f"{h_code}.HK"
        latest_h = self._h_store.get_latest_date(h_code_full)
        if force_start:
            start_h = force_start
        elif latest_h:
            start_h = (latest_h - timedelta(days=3)).to_pydatetime()
        else:
            start_h = now - timedelta(days=self.DEFAULT_LOOKBACK_DAYS)

        raw_h = self.client.fetch_hk_daily_range(
            h_code, start_h, now
        )
        new_h = 0
        if not raw_h.empty:
            df_h = _normalize_daily(raw_h)
            new_h = self._h_store.append_prices(h_code_full, df_h)

        self._fetch_count += 1
        return {"a": new_a, "h": new_h}

    def calc_ah_premiums(self, date: str = None) -> pd.DataFrame:
        """
        Calculate AH premiums for all pairs.

        Returns:
            DataFrame with columns: a_code, h_code, a_price_cny, h_price_hkd,
                                   h_price_cny, rate, premium
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        results = []
        for a_code in self.symbols:
            h_code = self.client.get_h_code(a_code)
            if not h_code:
                continue

            a_prices = self._a_store.get_prices(a_code)
            h_prices = self._h_store.get_prices(f"{h_code}.HK")

            if a_prices.empty or h_prices.empty:
                continue

            # Get latest common date
            a_last = a_prices.iloc[-1]
            h_last = h_prices.iloc[-1]
            a_date = a_prices.index[-1].strftime("%Y-%m-%d")
            h_date = h_prices.index[-1].strftime("%Y-%m-%d")

            result = self._fx.calc_ah_premium_for_pair(
                a_code=a_code,
                a_price_cny=float(a_last),
                h_price_hkd=float(h_last),
                date=a_date,
            )
            if result:
                results.append(result)

        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()

    def fetch_all(self, show_progress: bool = True) -> Dict[str, Dict[str, int]]:
        """拉取所有A+H配对。返回 {a_code: {'a': rows, 'h': rows}}"""
        results = {}
        total = len(self.symbols)
        errors = 0

        for i, a_code in enumerate(self.symbols):
            try:
                r = self.fetch_symbol_pair(a_code)
                if r["a"] > 0 or r["h"] > 0:
                    results[a_code] = r
            except Exception as e:
                logger.debug("Error fetching %s: %s", a_code, e)
                errors += 1
                self._error_count += 1

            if show_progress and (i + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d — updated=%d errors=%d",
                    i + 1, total, len(results), errors,
                )

        logger.info(
            "Fetch complete: %d/%d updated, %d errors",
            len(results), total, errors,
        )
        return results


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A+H 双市数据拉取器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--universe", type=str, default="ah",
        help="Stock universe: ah, popular, sse50, hs300, test, etc.",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Specific symbols (overrides --universe)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit",
    )
    parser.add_argument(
        "--years", type=float, default=5.0,
        help="Initial history lookback in years (default: 5)",
    )
    parser.add_argument(
        "--adjust", type=str, default="qfq", choices=["", "qfq", "hfq"],
        help="Price adjustment (default: qfq)",
    )
    parser.add_argument(
        "--backend", type=str, default="parquet", choices=["parquet", "pg"],
        help="Storage backend (default: parquet)",
    )
    parser.add_argument(
        "--premiums", action="store_true",
        help="Calculate and display AH premiums after fetch",
    )

    args = parser.parse_args()

    symbols = args.symbols or AStockUniverse.get(args.universe)
    logger.info("Universe: %s, Symbols: %d", args.universe, len(symbols))

    client = AkshareClient(adjust=args.adjust)
    fetcher = AHStockFetcher(
        symbols=symbols,
        client=client,
        initial_years=args.years,
        backend=args.backend,
    )

    if args.once:
        results = fetcher.fetch_all()
        total_a = sum(r["a"] for r in results.values())
        total_h = sum(r["h"] for r in results.values())
        logger.info("Total new rows: A=%d, H=%d", total_a, total_h)

        if args.premiums:
            premiums_df = fetcher.calc_ah_premiums()
            if not premiums_df.empty:
                print("\n=== AH Premiums ===")
                for _, row in premiums_df.iterrows():
                    sign = "+" if row["premium"] > 0 else ""
                    print(
                        f"{row['a_code']}/{row['h_code']}: "
                        f"A={row['a_price_cny']:.2f} CNY, "
                        f"H={row['h_price_hkd']:.2f} HKD, "
                        f"Premium={sign}{row['premium']*100:.1f}%"
                    )
    else:
        logger.info("Use --once for single run")


if __name__ == "__main__":
    main()
