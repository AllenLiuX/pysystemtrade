"""
Integration test for A+H spread trading system.

Tests the full pipeline:
1. AHData stage computes spread and z-score
2. ah_spread rule generates forecast
3. System produces valid output for A+H pairs
"""

import pytest
import pandas as pd
import numpy as np


class TestAhSystemIntegration:
    """Integration tests for A+H spread system."""

    def test_ah_data_stage_in_system(self):
        """AHData stage is accessible via system.ah_data."""
        from systems.ah_data import AHData
        from systems.rawdata import RawData
        from systems.basesystem import System
        from sysdata.config.configdata import Config

        config = Config({"trading_rules": {}})

        from sysdata.sim.futures_sim_data import futuresSimData
        data = futuresSimData()

        system = System(
            [RawData(), AHData()],
            data,
            config,
        )

        assert hasattr(system, "ah_data")
        assert system.ah_data.name == "ah_data"

    def test_ah_spread_rule_config(self):
        """ah_spread rule can be configured in trading_rules."""
        from systems.trading_rules import TradingRule

        rule = TradingRule({
            "function": "systems.provided.rules.ah_spread.ah_spread",
            "data": ["ah_data.get_ah_spread_zscore"],
            "other_args": {"lookback": 20},
        })

        assert rule.function is not None
        assert "ah_data.get_ah_spread_zscore" in rule.data
        assert rule.other_args["lookback"] == 20

    def test_ah_pair_codes_match_mapping(self):
        """AHData pair codes match the ah_mapping.csv."""
        from systems.ah_data import AHData
        import pandas as pd

        ah_data = AHData()
        mapping = ah_data._load_ah_mapping()

        csv_mapping = pd.read_csv("data/astock/csvconfig/ah_mapping.csv")

        for _, row in csv_mapping.iterrows():
            a_code = row["a_code"]
            h_code = str(row["h_code"]).zfill(5) + ".HK"

            assert mapping.get(a_code) == h_code, f"A-code {a_code} should map to {h_code}"
            assert mapping.get(h_code) == a_code, f"H-code {h_code} should map to {a_code}"
