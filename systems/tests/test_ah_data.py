"""
Tests for AHData stage.

Tests cover:
- Pair code lookup (bidirectional)
- Log return spread calculation
- Z-score calculation
- Edge cases (non-AH instruments, missing data)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from systems.ah_data import AHData

# Mock AH pairs for tests that need the mapping
MOCK_AH_PAIRS = [
    ("601318.SH", "02318"),
    ("600036.SH", "03968"),
    ("601398.SH", "01398"),
    ("601288.SH", "01288"),
]


class MockCache:
    """Mock cache that simply calls the function without caching."""

    def calc_or_cache(
        self,
        func,
        this_stage,
        *args,
        protected=False,
        not_pickable=False,
        instrument_classify=True,
        **kwargs,
    ):
        return func(this_stage, *args, **kwargs)


class MockRawData:
    """Mock RawData stage for testing."""

    def __init__(self, price_data: dict):
        self._price_data = price_data

    def get_daily_prices(self, instrument_code: str) -> pd.Series:
        if instrument_code in self._price_data:
            return self._price_data[instrument_code]
        raise Exception(f"No data for {instrument_code}")


class MockSystem:
    """Mock system for testing AHData."""

    def __init__(self, price_data: dict):
        self.rawdata = MockRawData(price_data)
        self.cache = MockCache()
        self._ah_data = None

    @property
    def ah_data(self):
        if self._ah_data is None:
            self._ah_data = AHData()
            self._ah_data._parent = self
        return self._ah_data


def make_price_series(values: list, start_date: str = "2026-01-01") -> pd.Series:
    """Create a price series with business day frequency."""
    dates = pd.bdate_range(start=start_date, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


class TestGetAhPairCode:
    """Tests for get_ah_pair_code method."""

    @patch("sysdata.astock.akshare_client.AkshareClient.get_ah_pairs", return_value=MOCK_AH_PAIRS)
    def test_a_to_h_lookup(self, mock_pairs):
        """A-share code returns H-share code."""
        ah_data = AHData()
        result = ah_data.get_ah_pair_code("601318.SH")
        assert result == "02318.HK"

    @patch("sysdata.astock.akshare_client.AkshareClient.get_ah_pairs", return_value=MOCK_AH_PAIRS)
    def test_h_to_a_lookup(self, mock_pairs):
        """H-share code returns A-share code."""
        ah_data = AHData()
        result = ah_data.get_ah_pair_code("02318.HK")
        assert result == "601318.SH"

    @patch("sysdata.astock.akshare_client.AkshareClient.get_ah_pairs", return_value=MOCK_AH_PAIRS)
    def test_non_ah_instrument_returns_none(self, mock_pairs):
        """Non-AH instrument returns None."""
        ah_data = AHData()
        result = ah_data.get_ah_pair_code("000001.SZ")
        assert result is None

    @patch("sysdata.astock.akshare_client.AkshareClient.get_ah_pairs", return_value=MOCK_AH_PAIRS)
    def test_multiple_pairs(self, mock_pairs):
        """Multiple known pairs resolve correctly."""
        ah_data = AHData()
        pairs = [
            ("600036.SH", "03968.HK"),
            ("601398.SH", "01398.HK"),
            ("601288.SH", "01288.HK"),
        ]
        for a_code, h_code in pairs:
            assert ah_data.get_ah_pair_code(a_code) == h_code
            assert ah_data.get_ah_pair_code(h_code) == a_code


class TestGetListOfAhInstruments:
    """Tests for get_list_of_ah_instruments method."""

    @patch("sysdata.astock.akshare_client.AkshareClient.get_ah_pairs", return_value=MOCK_AH_PAIRS)
    def test_returns_sorted_list(self, mock_pairs):
        """Returns a sorted list of instrument codes."""
        ah_data = AHData()
        instruments = ah_data.get_list_of_ah_instruments()
        assert isinstance(instruments, list)
        assert len(instruments) > 0
        assert instruments == sorted(instruments)

    @patch("sysdata.astock.akshare_client.AkshareClient.get_ah_pairs", return_value=MOCK_AH_PAIRS)
    def test_contains_both_a_and_h(self, mock_pairs):
        """List contains both A-share and H-share codes."""
        ah_data = AHData()
        instruments = ah_data.get_list_of_ah_instruments()
        has_a = any(c.endswith(".SH") or c.endswith(".SZ") for c in instruments)
        has_h = any(c.endswith(".HK") for c in instruments)
        assert has_a
        assert has_h


class TestGetAhLogReturnSpread:
    """Tests for get_ah_log_return_spread method."""

    def _make_mock_system(self, a_prices, h_prices, a_code="601318.SH", h_code="02318.HK"):
        """Create a mock system with controlled price data."""
        price_data = {
            a_code: make_price_series(a_prices),
            h_code: make_price_series(h_prices),
        }
        system = MockSystem(price_data)
        return system

    def test_spread_for_known_pair(self):
        """Spread is computed for a known A+H pair."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100, 100, 100, 100, 100, 100]

        system = self._make_mock_system(a_prices, h_prices)
        spread = system.ah_data.get_ah_log_return_spread("601318.SH")

        assert not spread.empty
        assert len(spread) == len(a_prices) - 1
        assert spread.iloc[-1] > 0

    def test_spread_symmetric(self):
        """Spread is the same regardless of which leg is queried."""
        a_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                    111, 110, 112, 114, 113, 115, 117, 116, 118, 120,
                    119, 121, 123, 122, 124, 126]
        h_prices = [100, 101, 102, 101, 103, 102, 104, 103, 105, 104,
                    106, 105, 107, 106, 108, 107, 109, 108, 110, 109,
                    111, 110, 112, 111, 113, 112]

        system = self._make_mock_system(a_prices, h_prices)
        spread_a = system.ah_data.get_ah_log_return_spread("601318.SH")
        spread_h = system.ah_data.get_ah_log_return_spread("02318.HK")

        pd.testing.assert_series_equal(spread_a, spread_h)

    def test_non_ah_instrument_returns_empty(self):
        """Non-AH instrument returns empty series."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        system = self._make_mock_system(a_prices, h_prices)
        spread = system.ah_data.get_ah_log_return_spread("000001.SZ")

        assert spread.empty

    def test_spread_with_common_dates_only(self):
        """Spread is computed only on common trading days."""
        a_dates = pd.bdate_range("2026-01-01", periods=10)
        a_prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                            index=a_dates)

        h_dates = pd.bdate_range("2026-01-05", periods=10)
        h_prices = pd.Series([100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                            index=h_dates)

        price_data = {
            "601318.SH": a_prices,
            "02318.HK": h_prices,
        }
        system = MockSystem(price_data)
        spread = system.ah_data.get_ah_log_return_spread("601318.SH")

        common = a_dates.intersection(h_dates)
        assert len(spread) == len(common) - 1


class TestGetAhSpreadZscore:
    """Tests for get_ah_spread_zscore method."""

    def _make_mock_system(self, a_prices, h_prices, a_code="601318.SH", h_code="02318.HK"):
        """Create a mock system with controlled price data."""
        price_data = {
            a_code: make_price_series(a_prices),
            h_code: make_price_series(h_prices),
        }
        system = MockSystem(price_data)
        return system

    def test_zscore_values(self):
        """Z-score is computed correctly."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                    130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                    140, 141, 142, 143, 144, 145]
        h_prices = [100] * len(a_prices)

        system = self._make_mock_system(a_prices, h_prices)
        zscore = system.ah_data.get_ah_spread_zscore("601318.SH", lookback=20)

        assert not zscore.empty
        assert zscore.iloc[:19].isna().all()
        assert zscore.iloc[19:].notna().any()

    def test_zscore_with_custom_lookback(self):
        """Z-score respects custom lookback parameter."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        system = self._make_mock_system(a_prices, h_prices)

        zscore_5 = system.ah_data.get_ah_spread_zscore("601318.SH", lookback=5)
        zscore_10 = system.ah_data.get_ah_spread_zscore("601318.SH", lookback=10)

        assert not zscore_5.equals(zscore_10)
        assert zscore_5.iloc[:4].isna().all()
        assert zscore_10.iloc[:9].isna().all()

    def test_zscore_for_non_ah_returns_empty(self):
        """Z-score for non-AH instrument returns empty series."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        system = self._make_mock_system(a_prices, h_prices)
        zscore = system.ah_data.get_ah_spread_zscore("000001.SZ")

        assert zscore.empty

    def test_zscore_mean_reversion_signal(self):
        """Z-score correctly identifies mean-reversion opportunity."""
        a_prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                    120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
                    140, 142, 144, 146, 148, 150]
        h_prices = [100] * len(a_prices)

        system = self._make_mock_system(a_prices, h_prices)
        zscore = system.ah_data.get_ah_spread_zscore("601318.SH", lookback=10)

        late_zscore = zscore.dropna().iloc[-1]
        assert late_zscore > 0

    def test_h_share_zscore_is_negated(self):
        """H-share z-score is the negative of A-share z-score."""
        a_prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                    120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
                    140, 142, 144, 146, 148, 150]
        h_prices = [100] * len(a_prices)

        system = self._make_mock_system(a_prices, h_prices)
        zscore_a = system.ah_data.get_ah_spread_zscore("601318.SH", lookback=10)
        zscore_h = system.ah_data.get_ah_spread_zscore("02318.HK", lookback=10)

        # Both should have data
        assert not zscore_a.dropna().empty
        assert not zscore_h.dropna().empty

        # H-share z-score should be the negative of A-share z-score
        common = zscore_a.dropna().index.intersection(zscore_h.dropna().index)
        pd.testing.assert_series_equal(
            zscore_a.loc[common],
            -zscore_h.loc[common]
        )


class TestAHPairNormalizedPosition:
    """Tests for get_ah_pair_normalized_position method."""

    def _make_mock_system_with_positions(self, a_prices, h_prices, a_positions, h_positions):
        """Create a mock system with controlled price data and raw positions."""
        price_data = {
            "601318.SH": make_price_series(a_prices),
            "02318.HK": make_price_series(h_prices),
        }
        system = MockSystem(price_data)

        class MockPositionSize:
            def __init__(self, pos_a, pos_h):
                self._pos_a = pos_a
                self._pos_h = pos_h

            def get_subsystem_position(self, instrument_code):
                if instrument_code == "601318.SH":
                    return self._pos_a
                return self._pos_h

        system.positionSize = MockPositionSize(
            pd.Series(a_positions, index=pd.bdate_range("2026-01-01", periods=len(a_positions))),
            pd.Series(h_positions, index=pd.bdate_range("2026-01-01", periods=len(h_positions))),
        )
        return system

    def test_normalized_positions_are_dollar_neutral(self):
        """Normalized positions satisfy pos_A + pos_H = 0."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [1.0, 1.2, 1.5, 1.3, 1.1, 1.4, 1.6, 1.2, 1.0, 0.8,
                 1.1, 1.3, 1.5, 1.7, 1.4, 1.2, 1.0, 0.9, 1.1, 1.3,
                 1.5, 1.2, 1.0, 0.8, 1.1, 1.4]
        h_pos = [-0.8, -1.0, -1.2, -1.1, -0.9, -1.1, -1.3, -1.0, -0.8, -0.6,
                 -0.9, -1.1, -1.2, -1.4, -1.1, -1.0, -0.8, -0.7, -0.9, -1.1,
                 -1.2, -1.0, -0.8, -0.6, -0.9, -1.1]

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        norm_a = system.ah_data.get_ah_pair_normalized_position("601318.SH")
        norm_h = system.ah_data.get_ah_pair_normalized_position("02318.HK")

        assert (norm_a + norm_h).abs().max() < 1e-10

    def test_normalized_positions_have_equal_magnitude(self):
        """|pos_A| == |pos_H| after normalization."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [1.0, 1.5, 2.0, 1.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5,
                 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0, 0.5, 1.0, 1.5,
                 2.0, 1.5, 1.0, 0.5, 1.0, 1.5]
        h_pos = [-0.5, -0.75, -1.0, -0.75, -0.5, -0.75, -1.0, -0.75, -0.5, -0.25,
                 -0.5, -0.75, -1.0, -1.25, -1.0, -0.75, -0.5, -0.25, -0.5, -0.75,
                 -1.0, -0.75, -0.5, -0.25, -0.5, -0.75]

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        norm_a = system.ah_data.get_ah_pair_normalized_position("601318.SH")
        norm_h = system.ah_data.get_ah_pair_normalized_position("02318.HK")

        pd.testing.assert_series_equal(norm_a.abs(), norm_h.abs())

    def test_normalized_position_uses_average_magnitude(self):
        """Normalized position magnitude is average of |pos_A| and |pos_H|."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [2.0] * len(a_prices)
        h_pos = [-1.0] * len(a_prices)

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        norm_a = system.ah_data.get_ah_pair_normalized_position("601318.SH")

        expected = pd.Series([1.5] * len(a_prices), index=pd.bdate_range("2026-01-01", periods=len(a_prices)))
        pd.testing.assert_series_equal(norm_a, expected)

    def test_normalized_position_preserves_a_share_direction(self):
        """Normalized position direction follows A-share's raw position sign."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [-1.5] * len(a_prices)
        h_pos = [-1.0] * len(a_prices)

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        norm_a = system.ah_data.get_ah_pair_normalized_position("601318.SH")
        norm_h = system.ah_data.get_ah_pair_normalized_position("02318.HK")

        assert (norm_a < 0).all()
        assert (norm_h > 0).all()

    def test_normalized_position_handles_zero_positions(self):
        """Zero positions on both legs result in zero normalized position."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [0.0] * len(a_prices)
        h_pos = [0.0] * len(a_prices)

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        norm_a = system.ah_data.get_ah_pair_normalized_position("601318.SH")
        assert (norm_a == 0.0).all()

    def test_normalized_position_for_non_ah_returns_empty(self):
        """Non-AH instrument returns empty series."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [1.0] * len(a_prices)
        h_pos = [-1.0] * len(a_prices)

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        result = system.ah_data.get_ah_pair_normalized_position("000001.SZ")
        assert result.empty

    def test_normalized_position_h_share_query_returns_h_leg(self):
        """Querying H-share returns the H leg's normalized position directly."""
        a_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122, 123, 124, 125]
        h_prices = [100] * len(a_prices)

        a_pos = [2.0] * len(a_prices)
        h_pos = [-1.0] * len(a_prices)

        system = self._make_mock_system_with_positions(a_prices, h_prices, a_pos, h_pos)

        norm_h = system.ah_data.get_ah_pair_normalized_position("02318.HK")

        # H-share should get -1.5 (negative of avg magnitude, since A is positive)
        expected = pd.Series([-1.5] * len(a_prices), index=pd.bdate_range("2026-01-01", periods=len(a_prices)))
        pd.testing.assert_series_equal(norm_h, expected)
