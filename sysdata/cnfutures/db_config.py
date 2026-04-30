"""
Backend factory for Chinese futures data storage.

Environment variables:
    CNFUTURES_BACKEND: 'parquet' (default) or 'pg' / 'postgresql' / 'postgres'
    CNFUTURES_PG_URL:  PostgreSQL connection string (required if backend=pg)
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load .env
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    pass

BACKEND = os.environ.get("CNFUTURES_BACKEND", "parquet").lower()
PG_URL = os.environ.get("CNFUTURES_PG_URL", "")


def _get_engine():
    """Create SQLAlchemy engine for PostgreSQL."""
    from sqlalchemy import create_engine
    return create_engine(PG_URL)


def get_daily_prices_store():
    """Return the daily prices storage backend."""
    if BACKEND in ("pg", "postgresql", "postgres"):
        from sysdata.cnfutures.cnfutures_pg import PGDailyPricesData
        return PGDailyPricesData(_get_engine())
    else:
        from sysdata.cnfutures.cnfutures_prices import CnFuturesDailyPricesData
        return CnFuturesDailyPricesData()


def get_instrument_store():
    """Return the instrument metadata storage backend."""
    if BACKEND in ("pg", "postgresql", "postgres"):
        from sysdata.cnfutures.cnfutures_pg import PGInstrumentData
        return PGInstrumentData(_get_engine())
    else:
        logger.warning("Instrument store not available for parquet backend")
        return None
