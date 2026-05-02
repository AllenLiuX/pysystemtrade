"""
Tests for ah_spread trading rule.

Tests cover:
- Rule returns negative of zscore, clamped to >= 0 (long-only)
- Rule handles NaN values correctly
- Rule handles empty series
- Long-only constraint is enforced
"""

import pytest
import pandas as pd
import numpy as np

from systems.provided.rules.ah_spread import ah_spread


class TestAhSpreadRule:
    """Tests for ah_spread rule function."""

    def test_returns_negative_of_zscore_clipped(self):
        """Rule returns the negative of the input zscore, clipped to >= 0."""
        zscore = pd.Series([1.5, 0.5, -0.5, -1.5],
                          index=pd.bdate_range("2026-01-01", periods=4))
        result = ah_spread(zscore)

        expected = pd.Series([0.0, 0.0, 0.5, 1.5],
                            index=pd.bdate_range("2026-01-01", periods=4))
        pd.testing.assert_series_equal(result, expected)

    def test_handles_nan_values(self):
        """Rule preserves NaN values."""
        zscore = pd.Series([1.5, np.nan, -0.5, np.nan],
                          index=pd.bdate_range("2026-01-01", periods=4))
        result = ah_spread(zscore)

        assert result.iloc[0] == 0.0  # clipped from -1.5
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 0.5
        assert np.isnan(result.iloc[3])

    def test_handles_empty_series(self):
        """Rule handles empty series."""
        zscore = pd.Series(dtype=float)
        result = ah_spread(zscore)

        assert result.empty

    def test_handles_zero_zscore(self):
        """Rule returns zero for zero zscore."""
        zscore = pd.Series([0.0, 0.0, 0.0],
                          index=pd.bdate_range("2026-01-01", periods=3))
        result = ah_spread(zscore)

        assert (result == 0.0).all()

    def test_signal_direction_positive_zscore(self):
        """Positive zscore (A outperforming) gives zero signal (long-only)."""
        zscore = pd.Series([2.0, 1.5, 1.0],
                          index=pd.bdate_range("2026-01-01", periods=3))
        result = ah_spread(zscore)

        assert (result == 0.0).all()

    def test_signal_direction_underperformance(self):
        """Negative zscore (A underperforming) gives positive signal."""
        zscore = pd.Series([-2.0, -1.5, -1.0],
                          index=pd.bdate_range("2026-01-01", periods=3))
        result = ah_spread(zscore)

        assert (result > 0).all()

    def test_long_only_constraint(self):
        """All forecast values are >= 0."""
        zscore = pd.Series([3.0, 1.0, 0.0, -1.0, -3.0],
                          index=pd.bdate_range("2026-01-01", periods=5))
        result = ah_spread(zscore)

        assert (result >= 0).all()

    def test_a_and_h_forecasts_long_only(self):
        """
        With long-only constraint, A and H shares have complementary signals.

        When A-share z-score is positive: A gets 0, H gets positive → long H only
        When A-share z-score is negative: A gets positive, H gets 0 → long A only
        """
        zscore_a = pd.Series([2.0, 1.5, -1.0, -0.5],
                            index=pd.bdate_range("2026-01-01", periods=4))
        zscore_h = -zscore_a

        forecast_a = ah_spread(zscore_a)
        forecast_h = ah_spread(zscore_h)

        # Both should be >= 0
        assert (forecast_a >= 0).all()
        assert (forecast_h >= 0).all()

        # When A has signal, H is zero and vice versa
        assert ((forecast_a > 0) & (forecast_h == 0)).any()
        assert ((forecast_h > 0) & (forecast_a == 0)).any()
