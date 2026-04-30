import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from dashboard.data.metrics import (
    get_metrics,
    get_cache_age,
    clear_cache,
    CACHE_DIR,
    CACHE_TTL_HOURS,
    _serialize_metrics,
    _deserialize_metrics,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


def test_cache_dir_exists():
    assert CACHE_DIR.exists()


def test_get_cache_age_returns_none_when_empty():
    assert get_cache_age() is None


def test_clear_cache_removes_files():
    cache_file = CACHE_DIR / "metrics.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{}")
    clear_cache()
    assert not cache_file.exists()


def test_serialize_deserialize_roundtrip():
    metrics = {
        "performance": {"sharpe": 0.87, "total_return": 12.5},
        "equity_curve": pd.Series([100, 105, 103], index=pd.date_range("2024-01-01", periods=3)),
        "weights": pd.DataFrame({"A": [0.5, 0.6], "B": [0.5, 0.4]}, index=pd.date_range("2024-01-01", periods=2)),
    }
    serialized = _serialize_metrics(metrics)
    deserialized = _deserialize_metrics(serialized)
    assert deserialized["performance"]["sharpe"] == 0.87
    assert len(deserialized["equity_curve"].values) == 3


@patch("dashboard.data.backtest.run_backtest")
def test_get_metrics_runs_backtest_on_cache_miss(mock_backtest):
    mock_backtest.return_value = {
        "equity_curve": pd.Series([100, 105], index=pd.date_range("2024-01-01", periods=2)),
        "performance": {"sharpe": 0.5, "total_return": 5.0},
    }
    result = get_metrics(refresh=True)
    mock_backtest.assert_called_once()
    assert "performance" in result


@patch("dashboard.data.backtest.run_backtest")
def test_get_metrics_uses_cache_on_hit(mock_backtest):
    mock_backtest.return_value = {
        "equity_curve": pd.Series([100, 105], index=pd.date_range("2024-01-01", periods=2)),
        "performance": {"sharpe": 0.5, "total_return": 5.0},
    }
    # First call runs backtest
    get_metrics(refresh=True)
    mock_backtest.reset_mock()
    # Second call should use cache
    get_metrics()
    mock_backtest.assert_not_called()


@patch("dashboard.data.backtest.run_backtest")
def test_get_metrics_refresh_bypasses_cache(mock_backtest):
    mock_backtest.return_value = {
        "equity_curve": pd.Series([100, 105], index=pd.date_range("2024-01-01", periods=2)),
        "performance": {"sharpe": 0.5, "total_return": 5.0},
    }
    get_metrics()
    mock_backtest.reset_mock()
    get_metrics(refresh=True)
    mock_backtest.assert_called_once()
