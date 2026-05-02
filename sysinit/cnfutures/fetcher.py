"""
CnFuturesFetcher - Orchestrates Chinese futures data fetching.

Workflow:
    Continuous contracts (default):
        1. Get contract list from Sina via futures_display_main_sina
        2. For each continuous contract (e.g. RB0):
           a. Check latest date in store
           b. Fetch daily data from futures_zh_daily_sina (returns full history)
           c. Filter to only new rows since latest date
           d. Upsert into store
        3. Optionally fetch contract details for instrument metadata

    Individual contracts (--individual):
        1. Discover all individual contract codes from exchange info endpoints
        2. For each contract:
           a. Check latest date in store
           b. Fetch daily data
           c. Upsert into store
        3. No instrument metadata fetched (too many contracts)

CLI:
    python -m sysinit.cnfutures.fetcher                  # incremental continuous
    python -m sysinit.cnfutures.fetcher --force          # re-fetch continuous
    python -m sysinit.cnfutures.fetcher --symbol RB0     # fetch single symbol
    python -m sysinit.cnfutures.fetcher --init-db        # create DB tables
    python -m sysinit.cnfutures.fetcher --individual     # fetch individual contracts
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

        name_lookup = dict(zip(contract_list["symbol"], contract_list["name"]))

        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(symbols):
            logger.info("[%d/%d] Fetching %s", i + 1, len(symbols), symbol)
            contract_name = name_lookup.get(symbol, "")
            try:
                self._fetch_single(symbol, contract_name)
                success_count += 1
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                fail_count += 1
            time.sleep(API_SLEEP)

        logger.info("Done. Success: %d, Failed: %d", success_count, fail_count)

    def fetch_single(self, symbol: str):
        """Fetch daily data for a single symbol."""
        cl = self.client.get_contract_list()
        name = cl[cl["symbol"] == symbol]["name"].values if cl is not None else []
        self._fetch_single(symbol, name[0] if len(name) > 0 else "")

    def _fetch_single(self, symbol: str, contract_name: str = "", skip_instrument: bool = False):
        """Internal: fetch and store daily data for one symbol."""
        # Fetch contract detail for metadata (best-effort, use continuous symbol)
        if self.instrument_store and not skip_instrument:
            detail_df = self.client.get_contract_detail(symbol)
            if detail_df is not None and not detail_df.empty:
                detail_map = dict(zip(detail_df["item"], detail_df["value"]))
                if contract_name:
                    detail_map["name"] = contract_name
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

    def fetch_individual_all(self):
        """Discover and fetch daily data for all individual contract months."""
        logger.info("Discovering individual contracts across all exchanges...")
        contracts = self.client.get_all_individual_contracts()
        logger.info("Found %d individual contracts", len(contracts))

        if not contracts:
            logger.error("No individual contracts discovered")
            return

        success_count = 0
        fail_count = 0
        skip_count = 0

        for i, symbol in enumerate(contracts):
            logger.info("[%d/%d] %s", i + 1, len(contracts), symbol)

            # Check if already up to date (fast path, skip instrument detail)
            latest = self.price_store.get_latest_date(symbol)
            if latest and not self.force:
                logger.info("  Already up to date: %s", latest)
                skip_count += 1
                time.sleep(0.1)  # fast sleep for already-cached
                continue
            if latest:
                logger.info("  Latest in store: %s (re-fetching)", latest)

            try:
                self._fetch_single(symbol, skip_instrument=True)
                success_count += 1
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                fail_count += 1
            time.sleep(API_SLEEP)

        logger.info("Done. Success: %d, Failed: %d, Skipped: %d",
                     success_count, fail_count, skip_count)

    def fetch_current_all(self):
        """Discover and fetch daily data for currently traded individual contracts."""
        logger.info("Discovering current contracts via futures_zh_realtime...")
        contracts = self.client.get_current_contracts()
        logger.info("Found %d current contracts", len(contracts))

        if not contracts:
            logger.error("No current contracts discovered")
            return

        success_count = 0
        fail_count = 0
        skip_count = 0

        for i, symbol in enumerate(contracts):
            logger.info("[%d/%d] %s", i + 1, len(contracts), symbol)

            latest = self.price_store.get_latest_date(symbol)
            if latest and not self.force:
                logger.info("  Already up to date: %s", latest)
                skip_count += 1
                time.sleep(0.05)
                continue
            if latest:
                logger.info("  Latest in store: %s (re-fetching)", latest)

            try:
                self._fetch_single(symbol, skip_instrument=True)
                success_count += 1
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                fail_count += 1
            time.sleep(API_SLEEP)

        logger.info("Done. Success: %d, Failed: %d, Skipped: %d",
                     success_count, fail_count, skip_count)


def main():
    parser = argparse.ArgumentParser(description="Fetch Chinese futures data from akshare")
    parser.add_argument("--force", action="store_true", help="Re-fetch all data (ignore incremental)")
    parser.add_argument("--symbol", type=str, help="Fetch single symbol only")
    parser.add_argument("--init-db", action="store_true", help="Create database tables")
    parser.add_argument("--instruments-only", action="store_true", help="Fetch only contract metadata")
    parser.add_argument("--individual", action="store_true",
                        help="Fetch individual contract months instead of continuous")
    parser.add_argument("--current", action="store_true",
                        help="Fetch currently traded contracts via real-time market data")
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

    if args.individual:
        fetcher.fetch_individual_all()
        return

    if args.current:
        fetcher.fetch_current_all()
        return

    if args.symbol:
        fetcher.fetch_single(args.symbol)
    else:
        fetcher.fetch_all()


if __name__ == "__main__":
    main()
