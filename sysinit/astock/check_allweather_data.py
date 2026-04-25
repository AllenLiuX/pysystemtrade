"""
check_allweather_data — Data integrity check for the All-Weather ETF universe.

For every primary + backup proxy in BUCKET_SPECS, verifies:
  1. Symbol exists in the daily-prices store
  2. Earliest / latest dates and row count
  3. Whether the proxy has >= MIN_YEARS_HISTORY years of data

Exits non-zero if any *primary* proxy fails the requirement (backups
are advisory only — they're allowed to be missing).

Usage:
    python -m sysinit.astock.check_allweather_data
"""

from __future__ import annotations

import sys
from typing import List

import pandas as pd

from sysdata.astock.db_config import get_daily_prices_store
from sysdata.astock.universe_allweather import BUCKET_SPECS, BucketSpec

MIN_YEARS_HISTORY = 5
TRADING_DAYS_PER_YEAR = 250


def _check_symbol(store, symbol: str) -> dict:
    try:
        prices = store.get_prices(symbol)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "ok": False, "rows": 0,
                "first": None, "last": None, "years": 0.0,
                "error": str(e)[:80]}

    if prices is None or len(prices) == 0:
        return {"symbol": symbol, "ok": False, "rows": 0,
                "first": None, "last": None, "years": 0.0,
                "error": "no rows"}

    first = pd.Timestamp(prices.index[0])
    last = pd.Timestamp(prices.index[-1])
    rows = len(prices)
    years = rows / TRADING_DAYS_PER_YEAR
    return {"symbol": symbol, "ok": years >= MIN_YEARS_HISTORY,
            "rows": rows, "first": first.date(), "last": last.date(),
            "years": years, "error": ""}


def main() -> int:
    store = get_daily_prices_store()
    print("=" * 88)
    print("A-Share All-Weather Universe — Data Integrity Check")
    print(f"Requirement: primary proxies must have >= {MIN_YEARS_HISTORY} "
          f"years of daily history")
    print("=" * 88)

    fail_primary: List[str] = []
    header = f"{'bucket':<14}  {'role':<8}  {'symbol':<10}  {'rows':>6}  " \
             f"{'first':<11}  {'last':<11}  {'years':>5}  status"
    print(header)
    print("-" * len(header))

    for spec in BUCKET_SPECS.values():
        spec: BucketSpec
        for role, symbol in (("primary", spec.primary),
                             *(("backup", b) for b in spec.backups)):
            r = _check_symbol(store, symbol)
            status = "OK " if r["ok"] else "MISS"
            extra = f"  ({r['error']})" if r["error"] else ""
            print(
                f"{spec.bucket:<14}  {role:<8}  {symbol:<10}  "
                f"{r['rows']:>6}  {str(r['first']):<11}  "
                f"{str(r['last']):<11}  {r['years']:>5.1f}  {status}{extra}"
            )
            if role == "primary" and not r["ok"]:
                fail_primary.append(f"{spec.bucket}/{symbol}")

    print("-" * len(header))

    if fail_primary:
        print(f"\nFAIL: {len(fail_primary)} primary proxy(ies) below "
              f"{MIN_YEARS_HISTORY}y history:")
        for f in fail_primary:
            print(f"   - {f}")
        print("\nRemediation: add these symbols to the fetcher universe "
              "and re-run, or switch to the backup proxy in "
              "sysdata/astock/universe_allweather.py.")
        return 1

    print("\nAll primary proxies meet the data-history requirement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
