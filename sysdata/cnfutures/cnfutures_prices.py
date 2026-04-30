"""
Parquet storage backend for Chinese futures daily prices.

File layout: data/cnfutures/daily_prices_parquet/<symbol>.parquet
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cnfutures" / "daily_prices_parquet"


class CnFuturesDailyPricesData:
    """Parquet storage for cnfutures daily prices."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_date(self, symbol: str):
        path = self.data_dir / f"{symbol}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df["date"].max()

    def append_prices(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        path = self.data_dir / f"{symbol}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(path, index=False)
        logger.info("Saved %d rows for %s to parquet", len(df), symbol)
