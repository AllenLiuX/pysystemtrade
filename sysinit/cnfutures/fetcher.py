"""
CnFuturesFetcher - Orchestrates Chinese futures data fetching.

Workflow:
    1. Get contract list from Sina via futures_display_main_sina
    2. For each continuous contract (e.g. RB0):
       a. Check latest date in store
       b. Fetch daily data from futures_zh_daily_sina (returns full history)
       c. Filter to only new rows since latest date
       d. Upsert into store
    3. Optionally fetch contract details for instrument metadata

CLI:
    python -m sysinit.cnfutures.fetcher              # incremental fetch all
    python -m sysinit.cnfutures.fetcher --force      # re-fetch all data
    python -m sysinit.cnfutures.fetcher --symbol RB0 # fetch single symbol
    python -m sysinit.cnfutures.fetcher --init-db    # create DB tables
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

# Load .env
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    pass

from sysdata.cnfutures.akshare_client import CnFuturesClient
from sysdata.cnfutures.db_config import get_daily_prices_store, get_instrument_store

logger = logging.getLogger(__name__)

# Rate limit: sleep between API calls to avoid being blocked
API_SLEEP = 0.5


class CnFuturesFetcher:
    """Orchestrates fetching Chinese futures data from akshare."""

    def __init__(self, force: bool = False):
        self.client = CnFuturesClient()
        self.price_store = get_daily_prices_store()
        self.instrument_store = get_instrument_store()
        self.force = force

    def init_db(self):
        """Create database tables (PostgreSQL only)."""
        from sysdata.cnfutures.db_config import BACKEND
        if BACKEND not in ("pg", "postgresql", "postgres"):
            logger.info("Backend is %s, no DB tables to create", BACKEND)
            return
        self.price_store.create_table()
        if self.instrument_store:
            self.instrument_store.create_table()
        logger.info("Database tables created")

    def fetch_all(self):
        """Fetch daily data for all continuous contracts."""
        logger.info("Fetching contract list from Sina...")
        contract_list = self.client.get_contract_list()
        if contract_list is None or contract_list.empty:
            logger.error("Failed to get contract list")
            return
        symbols = contract_list["symbol"].tolist()
        logger.info("Found %d contracts", len(symbols))

        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(symbols):
            logger.info("[%d/%d] Fetching %s", i + 1, len(symbols), symbol)
            try:
                self._fetch_single(symbol)
                success_count += 1
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                fail_count += 1
            time.sleep(API_SLEEP)

        logger.info("Done. Success: %d, Failed: %d", success_count, fail_count)

    def fetch_single(self, symbol: str):
        """Fetch daily data for a single symbol."""
        self._fetch_single(symbol)

    def _fetch_single(self, symbol: str):
        """Internal: fetch and store daily data for one symbol."""
        # Fetch contract detail for metadata (best-effort, use continuous symbol)
        if self.instrument_store:
            detail_df = self.client.get_contract_detail(symbol)
            if detail_df is not None and not detail_df.empty:
                detail_map = dict(zip(detail_df["item"], detail_df["value"]))
                # Add name from contract list
                try:
                    cl = self.client.get_contract_list()
                    if cl is not None:
                        match = cl[cl["symbol"] == symbol]
                        if not match.empty:
                            detail_map["name"] = match.iloc[0]["name"]
                except Exception:
                    pass
                self.instrument_store.upsert_instrument(symbol, detail_map)

        # Check latest date for incremental fetch
        latest = self.price_store.get_latest_date(symbol)
        logger.info("  Latest date in store: %s", latest)

        # akshare returns full history, so we always fetch and filter
        df = self.client.get_daily_data(symbol)
        if df is None or df.empty:
            logger.warning("  No data returned for %s", symbol)
            return

        # Filter to new data only (unless force mode)
        if latest and not self.force:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] > pd.to_datetime(latest)]
            if df.empty:
                logger.info("  No new data for %s", symbol)
                return

        logger.info("  Storing %d rows for %s", len(df), symbol)
        self.price_store.append_prices(symbol, df)

    def fetch_instruments_only(self):
        """Fetch only contract metadata, no price data."""
        contract_list = self.client.get_contract_list()
        if contract_list is None or contract_list.empty:
            logger.error("Failed to get contract list")
            return
        if not self.instrument_store:
            logger.error("Instrument store not available")
            return

        for _, row in contract_list.iterrows():
            symbol = row["symbol"]
            logger.info("Fetching detail for %s", symbol)
            detail_df = self.client.get_contract_detail(symbol)
            if detail_df is not None and not detail_df.empty:
                detail_map = dict(zip(detail_df["item"], detail_df["value"]))
                detail_map["name"] = row["name"]
                detail_map["exchange"] = row["exchange"]
                self.instrument_store.upsert_instrument(symbol, detail_map)
            time.sleep(API_SLEEP)


def main():
    parser = argparse.ArgumentParser(description="Fetch Chinese futures data from akshare")
    parser.add_argument("--force", action="store_true", help="Re-fetch all data (ignore incremental)")
    parser.add_argument("--symbol", type=str, help="Fetch single symbol only")
    parser.add_argument("--init-db", action="store_true", help="Create database tables")
    parser.add_argument("--instruments-only", action="store_true", help="Fetch only contract metadata")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    fetcher = CnFuturesFetcher(force=args.force)

    if args.init_db:
        fetcher.init_db()
        return

    if args.instruments_only:
        fetcher.fetch_instruments_only()
        return

    if args.symbol:
        fetcher.fetch_single(args.symbol)
    else:
        fetcher.fetch_all()


if __name__ == "__main__":
    main()
