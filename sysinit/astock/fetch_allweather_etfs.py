"""
fetch_allweather_etfs — One-shot loader for the All-Weather ETF universe.

Pulls daily history for all primary + backup ETFs in BUCKET_SPECS via
akshare and upserts into the local PostgreSQL daily-price store.

This is intentionally a one-shot script (not part of the pm2 fetcher
loop, since the underlying API is different from xiximiao). After the
initial backfill, run periodically to keep ETFs current.

Usage:
    # Backfill from 2014 (covers most ETF inception dates)
    python -m sysinit.astock.fetch_allweather_etfs --start 2014-01-01

    # Daily incremental (only fetch since latest stored date per symbol)
    python -m sysinit.astock.fetch_allweather_etfs --incremental

    # Specific symbols only
    python -m sysinit.astock.fetch_allweather_etfs --symbols 510300.SH 518880.SH
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from sysdata.astock.db_config import get_daily_prices_store
from sysdata.astock.etf_akshare_client import ETFAkshareClient
from sysdata.astock.universe_allweather import full_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("astock.fetch_aw_etfs")


def fetch_one(client: ETFAkshareClient, store, ts_code: str,
              start: str, incremental: bool) -> int:
    """Fetch + upsert one ETF; return rows written."""
    if incremental:
        latest = store.get_latest_date(ts_code)
        if latest is not None:
            start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(start) >= pd.Timestamp.today().normalize():
                logger.info("%s up-to-date (latest=%s) — skipping",
                            ts_code, latest.date())
                return 0

    df = client.fetch_daily(ts_code, start=start)
    if df.empty:
        logger.warning("%s: no data returned (start=%s)", ts_code, start)
        return 0

    store.add_ohlcv(ts_code, df)
    logger.info("%s: wrote %d rows (%s → %s)",
                ts_code, len(df),
                df.index[0].date(), df.index[-1].date())
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--start", default="2014-01-01",
                        help="Backfill start date (YYYY-MM-DD). Default 2014-01-01.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch since latest stored date per symbol.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Specific ETF symbols (overrides full_universe).")
    args = parser.parse_args()

    symbols: List[str] = args.symbols or full_universe()
    logger.info("Target symbols: %d → %s", len(symbols), symbols)

    client = ETFAkshareClient()
    store = get_daily_prices_store()

    total = 0
    failed: List[str] = []
    for sym in symbols:
        try:
            n = fetch_one(client, store, sym,
                          start=args.start,
                          incremental=args.incremental)
            total += n
        except Exception as e:  # noqa: BLE001
            logger.error("%s: failed — %s", sym, e)
            failed.append(sym)

    logger.info("=" * 60)
    logger.info("Done. %d total rows written across %d symbols.",
                total, len(symbols) - len(failed))
    if failed:
        logger.warning("Failed: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
