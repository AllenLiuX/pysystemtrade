"""
Backend factory for Chinese futures data storage.

Environment variables:
    CNFUTURES_BACKEND: 'parquet' (default) or 'pg' / 'postgresql' / 'postgres'
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB:
        Used to construct the connection string when backend=pg.
"""

import os
from pathlib import Path
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Load .env
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    pass

def get_backend() -> str:
    """Return the configured backend type."""
    return os.environ.get("CNFUTURES_BACKEND", "parquet").lower()


def get_pg_url() -> str:
    """
    Build PostgreSQL connection URL from environment variables.

    Priority:
    1. CNFUTURES_PG_URL
    2. POSTGRES_* individual variables
    """
    pg_url = os.environ.get("CNFUTURES_PG_URL", "")
    if pg_url:
        return pg_url

    host = os.environ.get("POSTGRES_HOST", "")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    dbname = os.environ.get("POSTGRES_DB", "postgres")
    if host and user and password:
        return (
            f"postgresql://{user}:{quote(password, safe='')}@"
            f"{host}:{port}/{dbname}?sslmode=require"
        )
    return ""


# Legacy module-level variables (callers should use get_backend() / get_pg_url())
BACKEND = get_backend()
PG_URL = get_pg_url()

# Module-level engine singleton to avoid leaking connection pools
_engine = None


def _get_engine():
    """Create or return cached SQLAlchemy engine for PostgreSQL."""
    global _engine
    if _engine is None:
        pg_url = get_pg_url()
        if not pg_url:
            raise ValueError(
                "CNFUTURES_PG_URL is not set. "
                "Add it to your .env file or set the environment variable."
            )
        from sqlalchemy import create_engine
        _engine = create_engine(
            pg_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_daily_prices_store():
    """Return the daily prices storage backend."""
    if get_backend() in ("pg", "postgresql", "postgres"):
        from sysdata.cnfutures.cnfutures_pg import PGDailyPricesData
        return PGDailyPricesData(_get_engine())
    else:
        from sysdata.cnfutures.cnfutures_prices import CnFuturesDailyPricesData
        return CnFuturesDailyPricesData()


def get_instrument_store():
    """Return the instrument metadata storage backend."""
    if get_backend() in ("pg", "postgresql", "postgres"):
        from sysdata.cnfutures.cnfutures_pg import PGInstrumentData
        return PGInstrumentData(_get_engine())
    else:
        logger.warning("Instrument store not available for parquet backend")
        return None
