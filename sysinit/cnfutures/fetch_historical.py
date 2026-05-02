"""
Fetch historical individual contracts for Chinese futures.

Generates contract codes for the past N years based on known delivery month patterns,
then fetches and stores daily data for each contract.

Usage:
    python -m sysinit.cnfutures.fetch_historical --years 4
    python -m sysinit.cnfutures.fetch_historical --years 4 --symbol RB
    python -m sysinit.cnfutures.fetch_historical --years 4 --dry-run
"""

import argparse
import logging
import time
from datetime import datetime
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
from sysdata.cnfutures.db_config import get_daily_prices_store

logger = logging.getLogger(__name__)

API_SLEEP = 0.5

# Delivery month patterns per exchange/product
# Most products trade: 1, 5, 9 (Jan, May, Sep) or 1-12 (monthly)
DELIVERY_PATTERNS = {
    # DCE agricultural/industrial: 1, 5, 9
    "DCE_159": ["01", "05", "09"],
    # DCE specific products
    "DCE_JM": ["01", "05", "09"],  # 焦煤
    "DCE_J": ["01", "05", "09"],   # 焦炭
    "DCE_I": ["01", "05", "09"],   # 铁矿石
    "DCE_M": ["01", "03", "05", "07", "08", "09", "11", "12"],  # 豆粕 (more months)
    "DCE_Y": ["01", "03", "05", "07", "08", "09", "11", "12"],  # 豆油
    "DCE_P": ["01", "05", "09"],   # 棕榈油
    "DCE_C": ["01", "03", "05", "07", "09", "11"],  # 玉米
    "DCE_CS": ["01", "03", "05", "07", "09", "11"], # 淀粉
    "DCE_L": ["01", "05", "09"],   # 塑料
    "DCE_V": ["01", "05", "09"],   # PVC
    "DCE_PP": ["01", "05", "09"],  # 聚丙烯
    "DCE_EG": ["01", "05", "09"],  # 乙二醇
    "DCE_EB": ["01", "05", "09"],  # 苯乙烯
    "DCE_LG": ["01", "05", "09"],  # 原木
    "DCE_LH": ["01", "03", "05", "07", "09", "11"], # 生猪
    "DCE_PG": ["01", "05", "09"],  # 液化石油气
    "DCE_BZ": ["01", "05", "09"],  # 纯苯
    "DCE_FB": ["01", "05", "09"],  # 纤维板
    "DCE_JD": ["01", "05", "09"],  # 鸡蛋
    "DCE_RR": ["01", "05", "09"],  # 粳米
    "DCE_A": ["01", "05", "09"],   # 豆一
    "DCE_B": ["01", "05", "09"],   # 豆二
    
    # SHFE metals/energy: 1-12 (monthly)
    "SHFE_MONTHLY": [f"{m:02d}" for m in range(1, 13)],
    
    # CZCE: varies by product, most are 1, 5, 9
    "CZCE_159": ["01", "05", "09"],
    "CZCE_MA": ["01", "05", "09"],  # 甲醇
    "CZCE_TA": ["01", "05", "09"],  # PTA
    "CZCE_SR": ["01", "05", "09"],  # 白糖
    "CZCE_CF": ["01", "05", "09"],  # 棉花
    "CZCE_FG": ["01", "05", "09"],  # 玻璃
    "CZCE_RM": ["01", "05", "09"],  # 菜粕
    "CZCE_OI": ["01", "05", "09"],  # 菜油
    "CZCE_AP": ["01", "05", "09"],  # 苹果
    "CZCE_CJ": ["01", "05", "09"],  # 红枣
    "CZCE_SF": ["01", "05", "09"],  # 硅铁
    "CZCE_SM": ["01", "05", "09"],  # 锰硅
    "CZCE_UR": ["01", "05", "09"],  # 尿素
    "CZCE_SA": ["01", "05", "09"],  # 纯碱
    "CZCE_PK": ["01", "05", "09"],  # 花生
    "CZCE_CY": ["01", "05", "09"],  # 棉纱
    "CZCE_JR": ["01", "05", "09"],  # 稻
    "CZCE_LR": ["01", "05", "09"],  # 晚籼稻
    "CZCE_RI": ["01", "05", "09"],  # 早籼稻
    "CZCE_WH": ["01", "05", "09"],  # 强麦
    "CZCE_PM": ["01", "05", "09"],  # 普麦
    "CZCE_RS": ["01", "05", "09"],  # 油菜籽
    
    # CFFEX equity index: 1-12
    "CFFEX_MONTHLY": [f"{m:02d}" for m in range(1, 13)],
    
    # INE: varies
    "INE_SC": ["01", "05", "09"],  # 原油
    "INE_FU": ["01", "05", "09"],  # 低硫燃油 (actually LU)
    "INE_LU": ["01", "05", "09"],  # 低硫燃油
    "INE_BC": ["01", "05", "09"],  # 国际铜
    "INE_NR": ["01", "05", "09"],  # 20号胶
    "INE_EC": ["01", "05", "09"],  # 集运欧线
    
    # GFEX: 1-12
    "GFEX_MONTHLY": [f"{m:02d}" for m in range(1, 13)],
}

# Product -> exchange mapping
PRODUCT_EXCHANGE = {
    # DCE
    "A": "DCE", "B": "DCE", "C": "DCE", "CS": "DCE", "EB": "DCE", "EG": "DCE",
    "FB": "DCE", "I": "DCE", "J": "DCE", "JD": "DCE", "JM": "DCE", "L": "DCE",
    "LH": "DCE", "LG": "DCE", "M": "DCE", "P": "DCE", "PG": "DCE", "PP": "DCE",
    "RR": "DCE", "V": "DCE", "Y": "DCE", "BZ": "DCE",
    # SHFE
    "AG": "SHFE", "AL": "SHFE", "AU": "SHFE", "BU": "SHFE", "CU": "SHFE",
    "FU": "SHFE", "HC": "SHFE", "NI": "SHFE", "PB": "SHFE", "RB": "SHFE",
    "RU": "SHFE", "SN": "SHFE", "SP": "SHFE", "WR": "SHFE", "ZN": "SHFE",
    "SS": "SHFE", "AO": "SHFE", "BR": "SHFE", "AD": "SHFE",
    # CZCE
    "AP": "CZCE", "CF": "CZCE", "CJ": "CZCE", "CY": "CZCE", "FG": "CZCE",
    "JR": "CZCE", "LR": "CZCE", "MA": "CZCE", "OI": "CZCE", "PK": "CZCE",
    "PM": "CZCE", "RI": "CZCE", "RM": "CZCE", "RS": "CZCE", "SA": "CZCE",
    "SF": "CZCE", "SM": "CZCE", "SR": "CZCE", "TA": "CZCE", "UR": "CZCE",
    "WH": "CZCE", "ZC": "CZCE",
    # CFFEX
    "IC": "CFFEX", "IF": "CFFEX", "IH": "CFFEX", "IM": "CFFEX", "T": "CFFEX",
    "TF": "CFFEX", "TS": "CFFEX", "TL": "CFFEX",
    # INE
    "BC": "INE", "EC": "INE", "LU": "INE", "NR": "INE", "SC": "INE",
    # GFEX
    "LC": "GFEX", "SI": "GFEX", "PS": "GFEX", "PT": "GFEX", "PD": "GFEX",
}

# Product -> delivery pattern key
PRODUCT_PATTERN = {
    # DCE
    "A": "DCE_159", "B": "DCE_159", "C": "DCE_C", "CS": "DCE_CS", "EB": "DCE_EB",
    "EG": "DCE_EG", "FB": "DCE_FB", "I": "DCE_I", "J": "DCE_J", "JD": "DCE_JD",
    "JM": "DCE_JM", "L": "DCE_L", "LH": "DCE_LH", "LG": "DCE_LG", "M": "DCE_M",
    "P": "DCE_P", "PG": "DCE_PG", "PP": "DCE_PP", "RR": "DCE_RR", "V": "DCE_V",
    "Y": "DCE_Y", "BZ": "DCE_BZ",
    # SHFE (all monthly)
    "AG": "SHFE_MONTHLY", "AL": "SHFE_MONTHLY", "AU": "SHFE_MONTHLY",
    "BU": "SHFE_MONTHLY", "CU": "SHFE_MONTHLY", "FU": "SHFE_MONTHLY",
    "HC": "SHFE_MONTHLY", "NI": "SHFE_MONTHLY", "PB": "SHFE_MONTHLY",
    "RB": "SHFE_MONTHLY", "RU": "SHFE_MONTHLY", "SN": "SHFE_MONTHLY",
    "SP": "SHFE_MONTHLY", "WR": "SHFE_MONTHLY", "ZN": "SHFE_MONTHLY",
    "SS": "SHFE_MONTHLY", "AO": "SHFE_MONTHLY", "BR": "SHFE_MONTHLY",
    "AD": "SHFE_MONTHLY",
    # CZCE
    "AP": "CZCE_AP", "CF": "CZCE_CF", "CJ": "CZCE_CJ", "CY": "CZCE_CY",
    "FG": "CZCE_FG", "JR": "CZCE_JR", "LR": "CZCE_LR", "MA": "CZCE_MA",
    "OI": "CZCE_OI", "PK": "CZCE_PK", "PM": "CZCE_PM", "RI": "CZCE_RI",
    "RM": "CZCE_RM", "RS": "CZCE_RS", "SA": "CZCE_SA", "SF": "CZCE_SF",
    "SM": "CZCE_SM", "SR": "CZCE_SR", "TA": "CZCE_TA", "UR": "CZCE_UR",
    "WH": "CZCE_WH", "ZC": "CZCE_159",
    # CFFEX (all monthly)
    "IC": "CFFEX_MONTHLY", "IF": "CFFEX_MONTHLY", "IH": "CFFEX_MONTHLY",
    "IM": "CFFEX_MONTHLY", "T": "CFFEX_MONTHLY", "TF": "CFFEX_MONTHLY",
    "TS": "CFFEX_MONTHLY", "TL": "CFFEX_MONTHLY",
    # INE
    "BC": "INE_BC", "EC": "INE_EC", "LU": "INE_LU", "NR": "INE_NR", "SC": "INE_SC",
    # GFEX (all monthly)
    "LC": "GFEX_MONTHLY", "SI": "GFEX_MONTHLY", "PS": "GFEX_MONTHLY",
    "PT": "GFEX_MONTHLY", "PD": "GFEX_MONTHLY",
}


def generate_historical_contracts(years: int = 4) -> list[str]:
    """Generate all individual contract codes for the past N years."""
    current_year = datetime.now().year
    contracts = []
    
    for product, pattern_key in PRODUCT_PATTERN.items():
        months = DELIVERY_PATTERNS.get(pattern_key, ["01", "05", "09"])
        for year_offset in range(years):
            year = current_year - year_offset
            yy = str(year)[-2:]  # "2024" -> "24"
            for mm in months:
                symbol = f"{product}{yy}{mm}"
                contracts.append(symbol)
    
    return sorted(set(contracts))


def fetch_historical(years: int = 4, target_symbols: list[str] = None, dry_run: bool = False):
    """Fetch historical individual contract data."""
    client = CnFuturesClient()
    price_store = get_daily_prices_store()
    
    if target_symbols:
        symbols = target_symbols
    else:
        symbols = generate_historical_contracts(years)
    
    logger.info(f"Target: {len(symbols)} contracts")
    
    success = 0
    fail = 0
    skip = 0
    
    for i, symbol in enumerate(symbols):
        latest = price_store.get_latest_date(symbol)
        if latest:
            skip += 1
            if i % 50 == 0:
                logger.info(f"[{i}/{len(symbols)}] {symbol}: already has data ({latest})")
            continue
        
        if dry_run:
            logger.info(f"[{i}/{len(symbols)}] {symbol}: would fetch (dry run)")
            continue
        
        logger.info(f"[{i}/{len(symbols)}] Fetching {symbol}")
        try:
            df = client.get_daily_data(symbol)
            if df is not None and not df.empty:
                price_store.append_prices(symbol, df)
                logger.info(f"  Stored {len(df)} rows")
                success += 1
            else:
                logger.info(f"  No data returned")
                fail += 1
        except Exception as e:
            logger.error(f"  Failed: {e}")
            fail += 1
        
        time.sleep(API_SLEEP)
    
    logger.info(f"Done. Success: {success}, Failed: {fail}, Skipped: {skip}")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical Chinese futures contracts")
    parser.add_argument("--years", type=int, default=4, help="Number of years of history (default: 4)")
    parser.add_argument("--symbol", type=str, help="Fetch single product only (e.g., RB)")
    parser.add_argument("--dry-run", action="store_true", help="List contracts without fetching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    if args.symbol:
        product = args.symbol.upper()
        if product not in PRODUCT_PATTERN:
            logger.error(f"Unknown product: {product}")
            return
        months = DELIVERY_PATTERNS.get(PRODUCT_PATTERN[product], ["01", "05", "09"])
        current_year = datetime.now().year
        symbols = []
        for year_offset in range(args.years):
            year = current_year - year_offset
            yy = str(year)[-2:]
            for mm in months:
                symbols.append(f"{product}{yy}{mm}")
        symbols = sorted(set(symbols))
    else:
        symbols = None
    
    if args.dry_run:
        contracts = generate_historical_contracts(args.years) if not symbols else symbols
        logger.info(f"Would fetch {len(contracts)} contracts:")
        for s in contracts[:20]:
            logger.info(f"  {s}")
        if len(contracts) > 20:
            logger.info(f"  ... and {len(contracts) - 20} more")
        return
    
    fetch_historical(args.years, symbols, args.dry_run)


if __name__ == "__main__":
    main()
