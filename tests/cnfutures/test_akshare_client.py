import pytest
from sysdata.cnfutures.akshare_client import CnFuturesClient


class TestGetContractList:
    """Test futures_display_main_sina wrapper."""

    def test_returns_dataframe_with_expected_columns(self):
        client = CnFuturesClient()
        df = client.get_contract_list()
        assert not df.empty
        assert list(df.columns) == ["symbol", "exchange", "name"]

    def test_contains_known_exchanges(self):
        client = CnFuturesClient()
        df = client.get_contract_list()
        known = {"dce", "shfe", "czce", "cffex", "gfex", "ine"}
        actual = set(df["exchange"].unique())
        assert actual & known


class TestGetContractDetail:
    """Test futures_contract_detail wrapper."""

    def test_returns_key_value_dataframe(self):
        client = CnFuturesClient()
        df = client.get_contract_detail("RB2410")
        assert not df.empty
        assert list(df.columns) == ["item", "value"]

    def test_contains_expected_fields(self):
        client = CnFuturesClient()
        df = client.get_contract_detail("RB2410")
        items = set(df["item"].tolist())
        expected = {"交易品种", "交易代码", "上市交易所", "交易单位", "最小变动价位"}
        assert expected <= items


class TestGetDailyData:
    """Test futures_zh_daily_sina wrapper."""

    def test_returns_dataframe_with_expected_columns(self):
        client = CnFuturesClient()
        df = client.get_daily_data("RB0")
        assert not df.empty
        expected_cols = {"date", "open", "high", "low", "close", "volume", "hold", "settle"}
        assert expected_cols <= set(df.columns)

    def test_specific_contract_returns_data(self):
        client = CnFuturesClient()
        df = client.get_daily_data("RB2410")
        assert not df.empty
        assert len(df) > 0
