"""
A+H Spread Trading System Builder.

Creates a pysystemtrade system with AHData stage and ah_spread rule
integrated into the standard system stack.

Usage:
    from systems.provided.ah_system import build_ah_system
    system = build_ah_system(data, config)
"""

from systems.basesystem import System
from systems.rawdata import RawData
from systems.ah_data import AHData
from systems.forecasting import Rules
from systems.forecast_scale_cap import ForecastScaleCap
from systems.forecast_combine import ForecastCombine
from systems.portfolios import Portfolios
from systems.position_sizing import PositionSizing
from systems.account import Account
from sysdata.sim.futures_sim_data import futuresSimData
from sysdata.config.configdata import Config


DEFAULT_AH_RULES = {
    "ah_spread": {
        "function": "systems.provided.rules.ah_spread.ah_spread",
        "data": ["ah_data.get_ah_spread_zscore"],
        "other_args": {"lookback": 20},
    },
}


def build_ah_system(
    data: futuresSimData,
    config: Config = None,
    trading_rules: dict = None,
) -> System:
    """
    Build an A+H spread trading system.

    :param data: simData instance with A-share and H-share price data
    :param config: System configuration (optional)
    :param trading_rules: Custom trading rules dict (optional, uses default ah_spread rule)
    :returns: System with AHData stage integrated
    """
    if trading_rules is None:
        trading_rules = DEFAULT_AH_RULES

    if config is None:
        config = Config({"trading_rules": trading_rules})
    else:
        existing_rules = config.trading_rules
        if "ah_spread" not in existing_rules:
            existing_rules.update(trading_rules)

    rules = Rules(trading_rules)

    system = System(
        [
            Account(),
            Portfolios(),
            PositionSizing(),
            RawData(),
            AHData(),
            ForecastCombine(),
            ForecastScaleCap(),
            rules,
        ],
        data,
        config,
    )

    return system
