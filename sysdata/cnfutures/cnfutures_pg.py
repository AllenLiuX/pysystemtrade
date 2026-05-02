"""
PostgreSQL storage backend for Chinese futures data.

Tables:
    - cnfutures_daily_prices: daily OHLCV + open interest + settlement
    - cnfutures_instruments: contract specification metadata
"""

import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from sqlalchemy import (
    Table, Column, BigInteger, Float, String, Date, DateTime, Text,
    UniqueConstraint, Index, MetaData, func, select,
)
from sqlalchemy.dialects.postgresql import insert

logger = logging.getLogger(__name__)

# Mapping from akshare API Chinese field names to English database columns.
# The akshare futures_contract_detail endpoint returns key-value pairs where
# the "item" column contains these Chinese field names.
AKSHARE_FIELD_MAP = {
    # Akshare API key (Chinese)   -> English column name
    "交易品种":                    "product_name",    # Product name
    "交易代码":                    "product_code",    # Product/symbol code
    "上市交易所":                  "exchange",        # Exchange
    "交易单位":                    "trading_unit",    # Contract size
    "最小变动价位":                "tick_size",       # Minimum price movement
    "合约交割月份":                "delivery_months", # Delivery months
    "最后交易日":                  "last_trading_day",# Last trading day rule
    "最后交割日":                  "last_delivery_day",# Last delivery day rule
    "最低交易保证金":              "margin_spec",     # Margin requirements
    "交易手续费":                  "fee_spec",        # Fee structure
    "涨跌停板幅度":                "limit_spec",      # Daily price limit
    "交易时间":                    "trading_hours",   # Trading session hours
    "交割方式":                    "delivery_method", # Settlement method
    "报价单位":                    "quotation_unit",  # Price quotation unit
}


class PGDailyPricesData:
    """PostgreSQL storage for cnfutures daily prices."""

    def __init__(self, engine):
        self.engine = engine
        self._table = None

    @property
    def table(self):
        if self._table is None:
            metadata = MetaData()
            self._table = Table(
                "cnfutures_daily_prices",
                metadata,
                Column("id", BigInteger, primary_key=True, autoincrement=True),
                Column("symbol", String(20), nullable=False),
                Column("dt", Date, nullable=False),
                Column("open", Float),
                Column("high", Float),
                Column("low", Float),
                Column("close", Float),
                Column("volume", BigInteger),
                Column("hold", BigInteger),
                Column("settle", Float),
                Column("created_at", DateTime, server_default=func.now()),
                UniqueConstraint("symbol", "dt", name="uq_cnfutures_daily_symbol_dt"),
                Index("ix_cnfutures_daily_symbol_dt", "symbol", "dt"),
            )
        return self._table

    def create_table(self):
        """Create the table if it doesn't exist."""
        self.table.create(self.engine, checkfirst=True)
        logger.info("Table cnfutures_daily_prices ready")

    def get_latest_date(self, symbol: str) -> Optional[str]:
        """Get the latest date for a symbol. Returns 'YYYY-MM-DD' or None."""
        with self.engine.connect() as conn:
            stmt = select(func.max(self.table.c.dt)).where(
                self.table.c.symbol == symbol
            )
            result = conn.execute(stmt).scalar()
            if result:
                return result.strftime("%Y-%m-%d")
            return None

    def append_prices(self, symbol: str, df: pd.DataFrame):
        """
        Append daily prices for a symbol. Upserts on (symbol, dt).

        Args:
            symbol: e.g. 'RB0', 'RB2410'
            df: must have columns: date, open, high, low, close, volume, hold, settle
        """
        if df is None or df.empty:
            return

        # Build rows using to_dict for efficiency
        records = df.to_dict("records")
        rows = []
        for rec in records:
            date_val = rec.get("date")
            if date_val is None:
                continue
            rows.append({
                "symbol": symbol,
                "dt": pd.to_datetime(date_val).date(),
                "open": float(rec["open"]) if pd.notna(rec.get("open")) else None,
                "high": float(rec["high"]) if pd.notna(rec.get("high")) else None,
                "low": float(rec["low"]) if pd.notna(rec.get("low")) else None,
                "close": float(rec["close"]) if pd.notna(rec.get("close")) else None,
                "volume": int(rec["volume"]) if pd.notna(rec.get("volume")) else 0,
                "hold": int(rec["hold"]) if pd.notna(rec.get("hold")) else 0,
                "settle": float(rec["settle"]) if pd.notna(rec.get("settle")) else None,
            })

        stmt = insert(self.table).values(rows)
        upsert_stmt = stmt.on_conflict_do_update(
            constraint="uq_cnfutures_daily_symbol_dt",
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "hold": stmt.excluded.hold,
                "settle": stmt.excluded.settle,
            }
        )

        with self.engine.begin() as conn:
            conn.execute(upsert_stmt)

        logger.info("Upserted %d rows for %s", len(rows), symbol)


class PGInstrumentData:
    """PostgreSQL storage for cnfutures instrument metadata."""

    def __init__(self, engine):
        self.engine = engine
        self._table = None

    @property
    def table(self):
        if self._table is None:
            metadata = MetaData()
            self._table = Table(
                "cnfutures_instruments",
                metadata,
                Column("symbol", String(20), primary_key=True),
                Column("name", String(100)),
                Column("exchange", String(20)),
                Column("product_code", String(10)),
                Column("product_name", String(100)),
                Column("trading_unit", String(50)),
                Column("tick_size", String(50)),
                Column("delivery_months", String(100)),
                Column("last_trading_day", String(200)),
                Column("last_delivery_day", String(200)),
                Column("margin_spec", String(200)),
                Column("fee_spec", String(200)),
                Column("limit_spec", String(200)),
                Column("trading_hours", Text),
                Column("delivery_method", String(50)),
                Column("quotation_unit", String(50)),
                Column("updated_at", DateTime, server_default=func.now()),
            )
        return self._table

    def create_table(self):
        self.table.create(self.engine, checkfirst=True)
        logger.info("Table cnfutures_instruments ready")

    def upsert_instrument(self, symbol: str, detail_map: dict):
        """
        Upsert a single instrument's metadata.

        Args:
            symbol: e.g. 'RB0'
            detail_map: dict mapping akshare API field names to values,
                        from CnFuturesClient.get_contract_detail()
        """
        row = {"symbol": symbol}
        for akshare_key, col_name in AKSHARE_FIELD_MAP.items():
            row[col_name] = detail_map.get(akshare_key)

        stmt = insert(self.table).values(**row)
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={k: stmt.excluded[k] for k in row if k != "symbol"},
        )
        upsert_stmt = upsert_stmt.values(updated_at=datetime.now(timezone.utc))

        with self.engine.begin() as conn:
            conn.execute(upsert_stmt)
