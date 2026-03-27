# Asset-Class Dispatch for A-Stock ETF Data Fetching

**Date:** 2026-03-26  
**Status:** Approved  

## Problem

`XiximiaoClient.fetch_daily_range()` only calls `pro.daily()` (Tushare equity endpoint), which returns empty results for ETF instruments (e.g., `510300.SH`). The current workaround is a fallback on empty result, which is inefficient and adds unnecessary latency.

## Solution

`AStockFetcher` looks up the `AssetClass` attribute per symbol from `instrumentconfig.csv` (via `AStockInstrumentData`) and dispatches to the appropriate client method directly — no retry logic or fallback needed.

## Changes

### 1. `sysdata/astock/xiximiao_client.py`

Add `fetch_fund_daily_range()` convenience method:

```python
def fetch_fund_daily_range(
    self,
    ts_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Convenience method: datetime args for fund_daily."""
    return self.fetch_fund_daily(
        ts_code=ts_code,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
```

Remove the fallback logic from `fetch_daily_range()` introduced in the previous hotfix.

### 2. `sysinit/astock/fetcher.py`

#### a) Import `AStockInstrumentData`

```python
from sysdata.astock.astock_instruments import AStockInstrumentData
```

#### b) Modify `AStockFetcher.__init__` to accept instrument data

```python
def __init__(
    self,
    symbols: List[str],
    freqs: List[str] = None,
    client: XiximiaoClient = None,
    instrument_data: AStockInstrumentData = None,
):
    self.symbols = symbols
    self.freqs = freqs or ["daily"]
    self.client = client or XiximiaoClient()
    self.instrument_data = instrument_data or AStockInstrumentData()
    ...
```

#### c) Modify `fetch_symbol_daily()` to dispatch by asset class

```python
def fetch_symbol_daily(self, ts_code: str, force_start: datetime = None) -> int:
    latest = self._daily_store.get_latest_date(ts_code)
    now = datetime.now()

    if force_start:
        start = force_start
    elif latest:
        start = (latest - timedelta(days=3)).to_pydatetime()
    else:
        start = now - timedelta(days=self.DEFAULT_LOOKBACK["daily"])

    asset_class = self.instrument_data.get_asset_class(ts_code)
    if asset_class == "ETF":
        raw = self.client.fetch_fund_daily_range(ts_code, start, now)
    else:
        raw = self.client.fetch_daily_range(ts_code, start, now)

    if raw.empty:
        return 0

    df = _normalize_daily(raw)
    return self._daily_store.append_prices(ts_code, df)
```

#### d) Revert `fetch_daily_range()` in client to non-fallback version (remove the ETF fallback logic added in hotfix)

## Data Flow

```
AStockFetcher.fetch_symbol_daily("510300.SH")
  → AStockInstrumentData.get_asset_class("510300.SH") → "ETF"
  → client.fetch_fund_daily_range(ts_code, start, end)
    → fetch_fund_daily() → pro.fund_daily() → Tushare API
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Unknown asset class | Default to `daily` (backward compatible) |
| Symbol not in config | Default to `daily` |
| API failure | Existing retry logic in `XiximiaoClient._call_with_retry()` |

## Backward Compatibility

- Symbols not in `instrumentconfig.csv` default to `daily` API
- No changes to public interfaces of `XiximiaoClient` or `AStockInstrumentData`
- Existing ETF entries (`510050.SH`, `510300.SH`, etc.) already have `AssetClass=ETF` in config

## Testing

- `AStockFetcher` testable with mocked `AStockInstrumentData`
- `XiximiaoClient` remains independently testable (no business logic)
- No changes to integration tests required
