"""
Supabase data access layer for the risk parity dashboard.

Uses the existing sysdata.astock PG backend (already configured to point
to Supabase via ASTOCK_PG_URL).
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def check_connection() -> bool:
    """Test Supabase connectivity."""
    try:
        from sysdata.astock.db_config import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Supabase connection failed: %s", e)
        return False


def get_daily_prices(
    instruments: list[str],
    start_date: Optional[str] = None,
) -> dict[str, pd.Series]:
    """
    Fetch daily close prices for given instruments.

    Returns dict of {instrument: pd.Series} with DatetimeIndex.
    """
    from sysdata.sim.astock_sim_data import AStockSimData

    data = AStockSimData()
    result = {}
    for instr in instruments:
        if start_date:
            sd = pd.Timestamp(start_date)
            prices = data.get_raw_price_from_start_date(instr, sd)
        else:
            prices = data.get_raw_price(instr)
        if not prices.empty:
            result[instr] = prices
    return result


def get_instruments() -> pd.DataFrame:
    """Fetch instrument metadata."""
    from sysdata.astock.db_config import get_instrument_data_store

    store = get_instrument_data_store()
    df = store.get_all_instrument_data_as_df()
    return df


def get_spread_costs() -> dict[str, float]:
    """Fetch spread cost mappings."""
    from sysdata.astock.db_config import get_spread_cost_store

    store = get_spread_cost_store()
    series = store.get_spread_costs_as_series()
    return series.to_dict()


def get_last_update() -> Optional[datetime]:
    """Get the most recent data timestamp from Supabase."""
    try:
        from sysdata.astock.db_config import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(dt) FROM astock_daily_prices")
            ).scalar()
        if result:
            return pd.Timestamp(result).to_pydatetime()
        return None
    except Exception as e:
        logger.error("Failed to get last update: %s", e)
        return None
