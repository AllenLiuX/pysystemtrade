"""
Tests for ah_spread trading rule.

Tests cover:
- Rule returns negative of zscore
- Rule handles NaN values correctly
- Rule handles empty series
"""

import pytest
import pandas as pd
import numpy as np

from systems.provided.rules.ah_spread import ah_spread


class TestAhSpreadRule:
    """Tests for ah_spread rule function."""

    def test_returns_negative_of_zscore(self):
        """Rule returns the negative of the input zscore."""
        zscore = pd.Series([1.5, 0.5, -0.5, -1.5],
                          index=pd.bdate_range("2026-01-01", periods=4))
        result = ah_spread(zscore)

        expected = pd.Series([-1.5, -0.5, 0.5, 1.5],
                            index=pd.bdate_range("2026-01-01", periods=4))
        pd.testing.assert_series_equal(result, expected)

    def test_handles_nan_values(self):
        """Rule preserves NaN values."""
        zscore = pd.Series([1.5, np.nan, -0.5, np.nan],
                          index=pd.bdate_range("2026-01-01", periods=4))
        result = ah_spread(zscore)

        assert result.iloc[0] == -1.5
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

    def test_signal_direction(self):
        """Positive zscore (A outperforming) gives negative signal."""
        zscore = pd.Series([2.0, 1.5, 1.0],
                          index=pd.bdate_range("2026-01-01", periods=3))
        result = ah_spread(zscore)

        assert (result < 0).all()

    def test_signal_direction_underperformance(self):
        """Negative zscore (A underperforming) gives positive signal."""
        zscore = pd.Series([-2.0, -1.5, -1.0],
                          index=pd.bdate_range("2026-01-01", periods=3))
        result = ah_spread(zscore)

        assert (result > 0).all()
