"""
H-share price data storage objects.

Mirrors AStockDailyPricesData but for Hong Kong-listed stocks.
Data schema and interface are identical.

File layout:
    data/astock/hk_daily_prices_parquet/<ts_code>.parquet
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

HK_DAILY_PRICES_DIRECTORY = "data.astock.hk_daily_prices_parquet"


def _resolve_parquet_dir(dotpath: str) -> Path:
    """
    将 pysystemtrade 风格的 dot-separated 路径转为绝对目录。
    """
    parts = dotpath.split(".")
    base = Path(__file__).resolve()
    for _ in range(5):
        base = base.parent
        if (base / "setup.py").exists():
            break
    dirpath = base.joinpath(*parts)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


class HKStockDailyPricesData:
    """
    H股日线价格存储

    读写 Parquet 文件，每个品种一个文件。
    index = DatetimeIndex, 列 = ['price']（即 close 价格）
    """

    def __init__(self, datapath: str = HK_DAILY_PRICES_DIRECTORY):
        self._dirpath = _resolve_parquet_dir(datapath)

    def __repr__(self):
        return f"HKStockDailyPricesData @ {self._dirpath}"

    def _filepath(self, ts_code: str) -> Path:
        safe = ts_code.replace(".", "_")
        return self._dirpath / f"{safe}.parquet"

    # ── read ───────────────────────────────────────────────────

    def get_list_of_instruments(self) -> List[str]:
        return sorted(
            p.stem.replace("_", ".") for p in self._dirpath.glob("*.parquet")
        )

    def get_prices(self, ts_code: str) -> pd.Series:
        """返回 pd.Series, index=DatetimeIndex, values=close price"""
        fpath = self._filepath(ts_code)
        if not fpath.exists():
            return pd.Series(dtype=float)
        df = pd.read_parquet(fpath)
        if "price" not in df.columns:
            return pd.Series(dtype=float)
        s = df["price"]
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s.name = "price"
        return s

    get_adjusted_prices = get_prices

    def get_prices_dataframe(self, ts_code: str) -> pd.DataFrame:
        """返回完整 OHLCV DataFrame"""
        fpath = self._filepath(ts_code)
        if not fpath.exists():
            return pd.DataFrame()
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # ── write ──────────────────────────────────────────────────

    def write_prices(self, ts_code: str, df: pd.DataFrame):
        """
        覆盖写入完整价格数据。

        Args:
            ts_code: '02318.HK'
            df: DataFrame with DatetimeIndex and columns including 'price'
        """
        fpath = self._filepath(ts_code)
        if df.empty:
            if fpath.exists():
                fpath.unlink()
                logger.info("Deleted empty price file for %s", ts_code)
            return
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.to_parquet(fpath)
        logger.info("Wrote %d rows (OHLCV) for %s → %s", len(df), ts_code, fpath)

    def append_prices(self, ts_code: str, df: pd.DataFrame) -> int:
        """
        增量追加价格数据，去重。

        Returns:
            新增行数
        """
        if df.empty:
            return 0

        existing = self.get_prices_dataframe(ts_code)

        if not existing.empty:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = df.copy()
            if not isinstance(combined.index, pd.DatetimeIndex):
                combined.index = pd.to_datetime(combined.index)
            combined = combined.sort_index()

        new_rows = len(combined) - len(existing) if not existing.empty else len(combined)
        self.write_prices(ts_code, combined)
        return new_rows

    def get_latest_date(self, ts_code: str) -> Optional[pd.Timestamp]:
        """获取该品种最新日期"""
        prices = self.get_prices(ts_code)
        if prices.empty:
            return None
        return prices.index[-1]

    def delete_prices(self, ts_code: str):
        """删除该品种的价格文件"""
        fpath = self._filepath(ts_code)
        if fpath.exists():
            fpath.unlink()
            logger.info("Deleted price file for %s", ts_code)
