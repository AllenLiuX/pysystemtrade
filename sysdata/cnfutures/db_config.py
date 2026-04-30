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

# Module-level engine singleton to avoid leaking connection pools
_engine = None


def _get_engine():
    """Create or return cached SQLAlchemy engine for PostgreSQL."""
    global _engine
    if _engine is None:
        if not PG_URL:
            raise ValueError(
                "CNFUTURES_PG_URL is not set. "
                "Add it to your .env file or set the environment variable."
            )
        from sqlalchemy import create_engine
        _engine = create_engine(
            PG_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


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
