"""
Initialize PostgreSQL tables for Chinese futures data.

Usage:
    python -m sysinit.cnfutures.init_pg
"""

import logging
import sys
from pathlib import Path

# Load .env
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    pass

from sysdata.cnfutures.db_config import get_daily_prices_store, get_instrument_store, BACKEND

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if BACKEND not in ("pg", "postgresql", "postgres"):
        logger.error("Backend must be 'pg', got '%s'", BACKEND)
        sys.exit(1)

    price_store = get_daily_prices_store()
    instrument_store = get_instrument_store()

    logger.info("Creating cnfutures_daily_prices table...")
    price_store.create_table()

    if instrument_store:
        logger.info("Creating cnfutures_instruments table...")
        instrument_store.create_table()

    logger.info("Done. Tables created successfully.")


if __name__ == "__main__":
    main()
