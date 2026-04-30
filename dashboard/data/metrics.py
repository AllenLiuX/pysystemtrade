"""
Cache layer for dashboard metrics.

Handles serialization/deserialization of backtest results to/from JSON,
with TTL-based expiration and manual refresh bypass.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "metrics.json"
CACHE_TTL_HOURS = 24

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_metrics(refresh: bool = False) -> dict:
    """
    Get all dashboard metrics.

    If refresh=True or cache is stale/missing, runs backtest and caches results.
    """
    from dashboard.data.backtest import run_backtest

    if refresh or _is_cache_stale():
        logger.info("Cache miss or stale — running backtest")
        metrics = run_backtest()
        _save_cache(metrics)
        return metrics

    logger.info("Cache hit — loading from cache")
    try:
        raw = json.loads(CACHE_FILE.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupted cache detected — running backtest")
        metrics = run_backtest()
        _save_cache(metrics)
        return metrics

    return _deserialize_metrics(raw)


def get_cache_age() -> Optional[datetime]:
    """Return the timestamp of the last cache write."""
    if CACHE_FILE.exists():
        mtime = CACHE_FILE.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    return None


def clear_cache():
    """Delete all cache files."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logger.info("Cache cleared")


def _is_cache_stale() -> bool:
    """Check if cache is missing or older than TTL."""
    if not CACHE_FILE.exists():
        return True
    age = get_cache_age()
    if age is None:
        return True
    return datetime.now(timezone.utc) - age > timedelta(hours=CACHE_TTL_HOURS)


def _save_cache(metrics: dict):
    """Serialize and save metrics to cache file."""
    serialized = _serialize_metrics(metrics)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(serialized, default=str))
    logger.info("Cache saved to %s", CACHE_FILE)


def _serialize_metrics(metrics: dict) -> dict:
    """Convert metrics dict to JSON-serializable format."""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, pd.Series):
            result[key] = {
                "type": "series",
                "index": [str(d) for d in value.index],
                "values": value.tolist(),
                "name": value.name,
            }
        elif isinstance(value, pd.DataFrame):
            result[key] = {
                "type": "dataframe",
                "index": [str(d) for d in value.index],
                "columns": list(value.columns),
                "data": value.values.tolist(),
            }
        elif isinstance(value, dict):
            result[key] = _serialize_metrics(value)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result


def _deserialize_metrics(data: dict) -> dict:
    """Restore metrics dict from JSON-serializable format."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if value.get("type") == "series":
                result[key] = pd.Series(
                    value["values"],
                    index=pd.DatetimeIndex(value["index"]),
                    name=value.get("name"),
                )
            elif value.get("type") == "dataframe":
                result[key] = pd.DataFrame(
                    value["data"],
                    index=pd.DatetimeIndex(value["index"]),
                    columns=value["columns"],
                )
            else:
                result[key] = _deserialize_metrics(value)
        else:
            result[key] = value
    return result
