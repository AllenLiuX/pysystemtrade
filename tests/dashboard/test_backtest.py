import pytest
import numpy as np
from dashboard.data.backtest import run_backtest


@pytest.fixture
def instruments():
    return ["510300.SH", "518880.SH", "511260.SH"]


@pytest.mark.slow
def test_backtest_returns_equity_curve(instruments):
    result = run_backtest(instruments, capital=1_000_000)
    assert "equity_curve" in result
    assert len(result["equity_curve"]) > 0
    assert result["equity_curve"].iloc[0] > 0


@pytest.mark.slow
def test_backtest_weights_sum_to_one(instruments):
    result = run_backtest(instruments, capital=1_000_000)
    weights = result["weights"]
    # Check last row sums to ~1.0
    assert abs(weights.iloc[-1].sum() - 1.0) < 0.01


@pytest.mark.slow
def test_backtest_performance_metrics(instruments):
    result = run_backtest(instruments, capital=1_000_000)
    perf = result["performance"]
    assert "total_return" in perf
    assert "sharpe" in perf
    assert "max_drawdown" in perf
    assert "annualized_return" in perf
    assert "annualized_vol" in perf


@pytest.mark.slow
def test_backtest_equal_weight_comparison(instruments):
    result = run_backtest(instruments, capital=1_000_000)
    assert "equal_weight_equity" in result
    assert "equal_performance" in result
    assert len(result["equal_weight_equity"]) > 0


@pytest.mark.slow
def test_backtest_per_instrument_metrics(instruments):
    result = run_backtest(instruments, capital=1_000_000)
    for instr in instruments:
        assert instr in result["instruments"]
        instr_metrics = result["instruments"][instr]
        assert "total_return" in instr_metrics
        assert "sharpe" in instr_metrics
