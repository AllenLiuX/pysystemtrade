"""
CnFuturesSimData - pysystemtrade simData for Chinese futures from Supabase.

Provides:
    - Back-adjusted daily prices
    - Raw carry data (front month vs next month prices)
    - Instrument metadata and costs

Builds carry from individual contract prices stored in cnfutures_daily_prices.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from syscore.exceptions import missingData
from sysdata.sim.sim_data import simData
from sysdata.cnfutures.db_config import _get_engine
from syslogging.logger import *

from sysobjects.spot_fx_prices import fxPrices
from sysobjects.instruments import instrumentCosts
from sysobjects.carry_data import rawCarryData

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


# ── Instrument config for all 82 continuous contracts ──────────────────────
# Pointsize, asset class, costs (0.03bps commission, 0.0 stamp)
_CNFUTURES_CONFIG = {
    "A0":  ("豆一",        10,   "Agricultural", "DCE"),
    "AD0": ("铸造铝合金",  10,   "Metals",       "GFEX"),
    "AG0": ("白银",        15,   "Metals",       "SHFE"),
    "AL0": ("铝",           5,   "Metals",       "SHFE"),
    "AO0": ("氧化铝",      20,   "Metals",       "SHFE"),
    "AP0": ("苹果",        10,   "Agricultural", "CZCE"),
    "AU0": ("黄金",      1000,   "Metals",       "SHFE"),
    "B0":  ("豆二",        10,   "Agricultural", "DCE"),
    "BC0": ("国际铜",       5,   "Metals",       "INE"),
    "BR0": ("丁二烯橡胶",   5,   "Metals",       "SHFE"),
    "BU0": ("沥青",        10,   "Energy",       "SHFE"),
    "BZ0": ("纯苯",        30,   "Energy",       "DCE"),
    "C0":  ("玉米",        10,   "Agricultural", "DCE"),
    "CF0": ("棉花",         5,   "Agricultural", "CZCE"),
    "CJ0": ("红枣",         5,   "Agricultural", "CZCE"),
    "CS0": ("淀粉",        10,   "Agricultural", "DCE"),
    "CU0": ("铜",           5,   "Metals",       "SHFE"),
    "CY0": ("棉纱",         5,   "Agricultural", "CZCE"),
    "EB0": ("苯乙烯",       5,   "Energy",       "DCE"),
    "EC0": ("集运欧线",     1,   "Shipping",     "INE"),
    "EG0": ("乙二醇",      10,   "Energy",       "DCE"),
    "FB0": ("纤维板",     500,   "Agricultural", "DCE"),
    "FG0": ("玻璃",        20,   "Metals",       "CZCE"),
    "FU0": ("燃料油",      10,   "Energy",       "SHFE"),
    "HC0": ("热轧卷板",    10,   "Metals",       "SHFE"),
    "I0":  ("铁矿石",     100,   "Metals",       "DCE"),
    "IC0": ("中证500",    200,   "Equity",       "CFFEX"),
    "IF0": ("沪深300",    300,   "Equity",       "CFFEX"),
    "IH0": ("上证50",     300,   "Equity",       "CFFEX"),
    "IM0": ("中证1000",   200,   "Equity",       "CFFEX"),
    "J0":  ("焦炭",       100,   "Energy",       "DCE"),
    "JD0": ("鸡蛋",         5,   "Agricultural", "DCE"),
    "JM0": ("焦煤",        60,   "Energy",       "DCE"),
    "JR0": ("粳稻",        20,   "Agricultural", "CZCE"),
    "L0":  ("塑料",         5,   "Energy",       "DCE"),
    "LC0": ("碳酸锂",       1,   "Metals",       "GFEX"),
    "LG0": ("原木",        90,   "Agricultural", "DCE"),
    "LH0": ("生猪",        16,   "Agricultural", "DCE"),
    "LR0": ("晚籼稻",      20,   "Agricultural", "CZCE"),
    "LU0": ("低硫燃油",    10,   "Energy",       "INE"),
    "M0":  ("豆粕",        10,   "Agricultural", "DCE"),
    "MA0": ("甲醇",        10,   "Energy",       "CZCE"),
    "NI0": ("镍",           1,   "Metals",       "SHFE"),
    "NR0": ("20号胶",      10,   "Agricultural", "INE"),
    "OI0": ("菜油",        10,   "Agricultural", "CZCE"),
    "OP0": ("胶版纸",      40,   "Agricultural", "SHFE"),
    "P0":  ("棕榈油",      10,   "Agricultural", "DCE"),
    "PB0": ("铅",           5,   "Metals",       "SHFE"),
    "PD0": ("钯",        1000,   "Metals",       "GFEX"),
    "PF0": ("短纤",         5,   "Energy",       "CZCE"),
    "PG0": ("LPG",         20,   "Energy",       "DCE"),
    "PK0": ("花生",         5,   "Agricultural", "CZCE"),
    "PL0": ("丙烯",        20,   "Energy",       "CZCE"),
    "PP0": ("聚丙烯",       5,   "Energy",       "DCE"),
    "PR0": ("瓶片",        15,   "Energy",       "CZCE"),
    "PS0": ("多晶硅",       3,   "Metals",       "GFEX"),
    "PT0": ("铂",        1000,   "Metals",       "GFEX"),
    "PX0": ("二甲苯",       5,   "Energy",       "CZCE"),
    "RB0": ("螺纹钢",      10,   "Metals",       "SHFE"),
    "RI0": ("早籼稻",      20,   "Agricultural", "CZCE"),
    "RM0": ("菜粕",        10,   "Agricultural", "CZCE"),
    "RR0": ("粳米",        10,   "Agricultural", "DCE"),
    "RS0": ("菜籽",        10,   "Agricultural", "CZCE"),
    "RU0": ("橡胶",        10,   "Agricultural", "SHFE"),
    "SA0": ("纯碱",        20,   "Energy",       "CZCE"),
    "SC0": ("原油",      1000,   "Energy",       "INE"),
    "SF0": ("硅铁",         5,   "Metals",       "CZCE"),
    "SH0": ("烧碱",        30,   "Energy",       "CZCE"),
    "SI0": ("工业硅",       5,   "Metals",       "GFEX"),
    "SM0": ("锰硅",         5,   "Metals",       "CZCE"),
    "SN0": ("锡",           1,   "Metals",       "SHFE"),
    "SP0": ("纸浆",        10,   "Agricultural", "SHFE"),
    "SR0": ("白糖",        10,   "Agricultural", "CZCE"),
    "SS0": ("不锈钢",       5,   "Metals",       "SHFE"),
    "TA0": ("PTA",          5,   "Energy",       "CZCE"),
    "TF0": ("5年国债",   10000,   "Bond",         "CFFEX"),
    "TS0": ("2年国债",   20000,   "Bond",         "CFFEX"),
    "UR0": ("尿素",        20,   "Agricultural", "CZCE"),
    "V0":  ("PVC",          5,   "Energy",       "DCE"),
    "WH0": ("强麦",        20,   "Agricultural", "CZCE"),
    "Y0":  ("豆油",        10,   "Agricultural", "DCE"),
    "ZN0": ("锌",           5,   "Metals",       "SHFE"),
}

ALL_INSTRUMENTS = sorted(_CNFUTURES_CONFIG.keys())


class CnFuturesSimData(simData):
    """simData for Chinese futures backtesting from Supabase."""

    def __init__(self, log=get_logger("CnFuturesSimData")):
        super().__init__(log=log)
        self._engine = _get_engine()
        self._carry_cache = {}
        self._price_cache = {}
        self._preloaded = False

    def preload_all_data(self) -> int:
        """Batch-load all individual contract data and build carry + prices.

        One SQL query replaces 82 individual queries. Builds carry data for
        all 82 instruments and caches the results.

        :returns: Number of instruments successfully loaded
        """
        if self._preloaded:
            return len(self._carry_cache)

        from sqlalchemy import text
        import re

        # One query: all individual contracts (exclude continuous by length check)
        query = text("""
            SELECT symbol, dt, close
            FROM cnfutures_daily_prices
            WHERE CHAR_LENGTH(symbol) >= 4
            ORDER BY symbol, dt
        """)
        with self._engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            self._preloaded = True
            return 0

        df = pd.DataFrame(rows, columns=["symbol", "dt", "close"])
        df["dt"] = pd.to_datetime(df["dt"])

        # Extract product code from symbol (letters before digits)
        def product_of(sym: str) -> str:
            return "".join(c for c in sym if c.isalpha()).upper()

        df["product"] = df["symbol"].apply(product_of)

        # Contract code: e.g., "RB2505" -> 20250500
        def contract_code(sym: str) -> int:
            num = "".join(c for c in sym if c.isdigit())
            if len(num) == 4:
                yy = int(num[:2])
                mm = int(num[2:])
                return ((2000 + yy) * 100 + mm) * 100
            return 99999999

        df["ccode"] = df["symbol"].apply(contract_code)

        # Build carry data per product + continuous pair
        loaded = 0
        for cont_sym in ALL_INSTRUMENTS:
            product = re.sub(r"0$", "", cont_sym).upper()
            subset = df[df["product"] == product]
            if subset.empty:
                continue
            self._carry_cache[cont_sym] = self._build_carry_from_subset(subset, product)
            if not self._carry_cache[cont_sym].empty:
                # Also build and cache the back-adjusted price
                self._price_cache[cont_sym] = self._build_price_from_carry(
                    self._carry_cache[cont_sym]
                )
                loaded += 1

        self._preloaded = True
        return loaded

    def _build_carry_from_subset(self, subset: pd.DataFrame, product: str) -> pd.DataFrame:
        """Build carry DataFrame from a product's individual contract rows."""
        # Pivot: dates x symbols
        pivot = subset.pivot_table(
            index="dt", columns="symbol", values="close", aggfunc="last"
        ).sort_index()

        if pivot.empty:
            return pd.DataFrame()

        # Map symbol -> contract code
        cc_map = subset.groupby("symbol")["ccode"].first().to_dict()
        cols_sorted = sorted(pivot.columns, key=lambda s: cc_map.get(s, 99999999))

        # Build PRICE/CARRY records
        dates, records = [], []
        for dt_val, row in pivot.iterrows():
            active = []
            for col in cols_sorted:
                val = row[col]
                if pd.notna(val) and val > 0:
                    active.append((col, val, cc_map.get(col, 99999999)))
            if len(active) >= 2:
                dates.append(dt_val)
                records.append({
                    "PRICE": active[0][1],
                    "CARRY": active[1][1],
                    "PRICE_CONTRACT": float(active[0][2]),
                    "CARRY_CONTRACT": float(active[1][2]),
                    "FORWARD": active[1][1],
                    "FORWARD_CONTRACT": float(active[1][2]),
                })
            elif len(active) == 1:
                dates.append(dt_val)
                records.append({
                    "PRICE": active[0][1],
                    "CARRY": active[0][1],
                    "PRICE_CONTRACT": float(active[0][2]),
                    "CARRY_CONTRACT": float(active[0][2]),
                    "FORWARD": active[0][1],
                    "FORWARD_CONTRACT": float(active[0][2]),
                })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records, index=pd.DatetimeIndex(dates))

    def _build_price_from_carry(self, carry_data: pd.DataFrame) -> pd.Series:
        """Build back-adjusted price from carry data."""
        if carry_data.empty:
            return pd.Series(dtype=float)

        price = carry_data["PRICE"].copy()
        carry_price = carry_data["CARRY"].copy()
        contract_changes = carry_data["PRICE_CONTRACT"].diff().fillna(0) != 0

        adjust = 0.0
        adjusted = []
        for i in range(len(price)):
            if contract_changes.iloc[i] and i > 0:
                roll_diff = price.iloc[i - 1] - carry_price.iloc[i - 1]
                if not np.isnan(roll_diff):
                    adjust += roll_diff
            adjusted.append(price.iloc[i] + adjust)

        result = pd.Series(adjusted, index=price.index)
        return result.dropna()

    def get_instrument_list(self) -> list:
        return ALL_INSTRUMENTS

    def get_instrument_currency(self, instrument_code: str) -> str:
        return "CNY"

    def get_value_of_block_price_move(self, instrument_code: str) -> float:
        cfg = _CNFUTURES_CONFIG.get(instrument_code)
        return float(cfg[1]) if cfg else 1.0

    def get_instrument_asset_class(self, instrument_code: str) -> str:
        cfg = _CNFUTURES_CONFIG.get(instrument_code)
        return cfg[2] if cfg else ""

    def get_raw_cost_data(self, instrument_code: str) -> instrumentCosts:
        # Chinese futures: ~0.03bps per side, no stamp duty
        cfg = _CNFUTURES_CONFIG.get(instrument_code)
        if cfg and cfg[1] > 100:  # large face value (bonds)
            pct = 0.000025
        else:
            pct = 0.0003
        return instrumentCosts(price_slippage=pct, value_of_block_commission=0.0)

    def _get_fx_data_from_start_date(
        self, currency1: str, currency2: str, start_date: datetime.datetime
    ) -> fxPrices:
        if currency1 == currency2:
            dates = pd.date_range(start_date, datetime.datetime.now(), freq="B")
            return fxPrices(pd.Series(1.0, index=dates))
        raise missingData(f"No FX data for {currency1}/{currency2}")

    # ── Price data ─────────────────────────────────────────────────────────

    def get_raw_price(self, instrument_code: str) -> pd.Series:
        """Back-adjusted continuous price series."""
        if instrument_code in self._price_cache:
            return self._price_cache[instrument_code]

        carry_data = self._get_raw_carry_data_df(instrument_code)
        if carry_data.empty:
            raise missingData(f"No price data for {instrument_code}")

        price = self._build_price_from_carry(carry_data)
        self._price_cache[instrument_code] = price
        return price

    # ── Carry data ─────────────────────────────────────────────────────────

    def get_instrument_raw_carry_data(self, instrument_code: str) -> rawCarryData:
        """Return rawCarryData with PRICE, CARRY, PRICE_CONTRACT, CARRY_CONTRACT.

        PRICE = front month close, CARRY = next month close.
        Contracts are identified by numeric codes (e.g., 202506 for June 2025).
        """
        df = self._get_raw_carry_data_df(instrument_code)
        if df.empty:
            raise missingData(f"No carry data for {instrument_code}")
        return rawCarryData(df)

    def _get_raw_carry_data_df(self, instrument_code: str) -> pd.DataFrame:
        """Build the 6-column multiple prices DataFrame from individual contracts."""
        if instrument_code in self._carry_cache:
            return self._carry_cache[instrument_code]

        import re
        product = re.sub(r"0$", "", instrument_code).upper()

        # Query all individual contracts for this product from Supabase
        from sqlalchemy import text

        # Use exact length matching: product_code + 4 digits = individual contract
        # e.g., product "I" → matches "I2605" (len=5) but not "IC2403" (len=6)
        query = text("""
            SELECT symbol, dt, close
            FROM cnfutures_daily_prices
            WHERE UPPER(symbol) LIKE :pat
              AND CHAR_LENGTH(symbol) = :expected_len
            ORDER BY symbol, dt
        """)
        expected_len = len(product) + 4  # product + YYMM
        with self._engine.connect() as conn:
            result = conn.execute(
                query,
                {"pat": f"{product}%", "expected_len": expected_len},
            )
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=["symbol", "dt", "close"])

        if df.empty:
            self._carry_cache[instrument_code] = pd.DataFrame()
            return pd.DataFrame()

        # Pivot: columns = contracts, rows = dates, values = close
        df["dt"] = pd.to_datetime(df["dt"])
        pivot = df.pivot_table(
            index="dt", columns="symbol", values="close", aggfunc="last"
        )
        pivot = pivot.sort_index()

        # Extract contract month codes from symbol names (e.g., "RB2505" -> 20250500)
        # pysystemtrade expects 8-digit codes: YYYYMM00
        def contract_code(sym: str) -> int:
            num = "".join(c for c in sym if c.isdigit())
            if len(num) == 4:
                yy = int(num[:2])
                mm = int(num[2:])
                return ((2000 + yy) * 100 + mm) * 100
            return 99999999

        cols_sorted = sorted(pivot.columns, key=contract_code)

        # Build PRICE/CARRY by date: for each date, find the two nearest active contracts
        dates = []
        records = []
        for dt_val, row in pivot.iterrows():
            active = []
            for col in cols_sorted:
                val = row[col]
                if pd.notna(val) and val > 0:
                    active.append((col, val))
            if len(active) >= 2:
                dates.append(dt_val)
                records.append({
                    "PRICE": active[0][1],
                    "CARRY": active[1][1],
                    "PRICE_CONTRACT": float(contract_code(active[0][0])),
                    "CARRY_CONTRACT": float(contract_code(active[1][0])),
                    "FORWARD": active[1][1],
                    "FORWARD_CONTRACT": float(contract_code(active[1][0])),
                })
            elif len(active) == 1:
                dates.append(dt_val)
                records.append({
                    "PRICE": active[0][1],
                    "CARRY": active[0][1],
                    "PRICE_CONTRACT": float(contract_code(active[0][0])),
                    "CARRY_CONTRACT": float(contract_code(active[0][0])),
                    "FORWARD": active[0][1],
                    "FORWARD_CONTRACT": float(contract_code(active[0][0])),
                })

        if not records:
            self._carry_cache[instrument_code] = pd.DataFrame()
            return pd.DataFrame()

        result = pd.DataFrame(records, index=pd.DatetimeIndex(dates))
        self._carry_cache[instrument_code] = result
        return result
