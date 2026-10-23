"""Cointegration-based pairs trading."""

from .cointegration import (
    PairFit,
    engle_granger_test,
    estimate_long_run_short_run_relationships,
    screen_pairs,
)
from .exchange import OrderBook, PriceLevel, SimulatedExchange
from .simulation import simulate_cointegrated_pair
from .strategy import PairsTrader, StrategyConfig

__all__ = [
    "PairFit",
    "engle_granger_test",
    "estimate_long_run_short_run_relationships",
    "screen_pairs",
    "OrderBook",
    "PriceLevel",
    "SimulatedExchange",
    "simulate_cointegrated_pair",
    "PairsTrader",
    "StrategyConfig",
]
