"""
Integration test: fetch -> store -> retrieve cycle for cnfutures.
Uses parquet backend to avoid needing a live DB.
"""

import os
import pytest
import pandas as pd
from pathlib import Path
import tempfile

# Force parquet backend for tests
os.environ["CNFUTURES_BACKEND"] = "parquet"

from sysdata.cnfutures.akshare_client import CnFuturesClient
from sysdata.cnfutures.cnfutures_prices import CnFuturesDailyPricesData


class TestFetchAndStore:
    """End-to-end: fetch from akshare, store to parquet, verify."""

    def test_fetch_and_store_rb0(self):
        """Fetch RB0 daily data and store it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = CnFuturesClient()
            store = CnFuturesDailyPricesData(Path(tmpdir))

            # Fetch
            df = client.get_daily_data("RB0")
            assert df is not None
            assert not df.empty
            assert "date" in df.columns
            assert "close" in df.columns

            # Store
            store.append_prices("RB0", df)
            assert (Path(tmpdir) / "RB0.parquet").exists()

            # Verify: read back and check
            stored = pd.read_parquet(Path(tmpdir) / "RB0.parquet")
            assert len(stored) == len(df)
            assert stored["date"].iloc[-1] == df["date"].iloc[-1]

    def test_incremental_append(self):
        """Test that appending doesn't duplicate rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = CnFuturesClient()
            store = CnFuturesDailyPricesData(Path(tmpdir))

            df = client.get_daily_data("RB0")
            store.append_prices("RB0", df)
            store.append_prices("RB0", df)  # append same data again

            stored = pd.read_parquet(Path(tmpdir) / "RB0.parquet")
            # Should deduplicate on date
            assert len(stored) == len(df)
