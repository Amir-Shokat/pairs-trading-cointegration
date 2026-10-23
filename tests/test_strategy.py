import numpy as np

from pairs_trading import PairsTrader, SimulatedExchange, StrategyConfig, simulate_cointegrated_pair
from pairs_trading.cointegration import fit_pair


def _run(config=None, seed=11):
    prices = simulate_cointegrated_pair(n=3000, seed=seed)
    fit = fit_pair(np.log(prices.iloc[:1500]), "Y", "X")
    exchange = SimulatedExchange(prices.iloc[1500:], seed=seed)
    trader = PairsTrader(exchange, [fit], config or StrategyConfig())
    return exchange, trader.run()


def test_positions_never_exceed_limit():
    exchange, log = _run()
    assert log["pos_Y"].abs().max() <= 100
    assert log["pos_X"].abs().max() <= 100


def test_strategy_trades_and_is_profitable_on_mean_reverting_spread():
    exchange, log = _run()
    assert len(exchange.trades) > 10
    assert exchange.get_pnl() > 0


def test_legs_stay_hedged():
    exchange, log = _run()
    held = log[log["pos_Y"] != 0]
    assert (np.sign(held["pos_Y"]) == -np.sign(held["pos_X"])).mean() > 0.95


def test_exit_at_mean_trades_more_often():
    ex_flip, _ = _run(StrategyConfig(exit_at_mean=False))
    ex_mean, _ = _run(StrategyConfig(exit_at_mean=True))
    assert len(ex_mean.trades) >= len(ex_flip.trades)
