import numpy as np
import pandas as pd
import pytest

from pairs_trading import engle_granger_test, estimate_long_run_short_run_relationships, screen_pairs
from pairs_trading.simulation import simulate_cointegrated_pair, simulate_random_walks


def test_recovers_true_parameters():
    prices = simulate_cointegrated_pair(n=5000, c=1.7, gamma=0.5, kappa=0.05, seed=1)
    logp = np.log(prices)
    c, gamma, alpha, z = estimate_long_run_short_run_relationships(logp["Y"], logp["X"])
    assert gamma == pytest.approx(0.5, abs=0.03)
    assert c == pytest.approx(1.7, abs=0.15)
    assert -0.1 < alpha < 0
    assert len(z) == len(prices)


def test_cointegrated_pair_is_detected():
    logp = np.log(simulate_cointegrated_pair(n=3000, seed=2))
    _, p = engle_granger_test(logp["Y"], logp["X"])
    assert p < 0.01


def test_independent_walks_are_not_cointegrated():
    logp = np.log(simulate_random_walks(n=3000, seed=3))
    _, p = engle_granger_test(logp["A"], logp["B"])
    assert p > 0.05


def test_mackinnon_is_more_conservative_than_plain_adf():
    logp = np.log(simulate_cointegrated_pair(n=400, kappa=0.03, seed=4))
    _, p_mk = engle_granger_test(logp["Y"], logp["X"], method="mackinnon")
    _, p_adf = engle_granger_test(logp["Y"], logp["X"], method="adf")
    assert p_mk >= p_adf


def test_screen_pairs_flags_only_the_cointegrated_pair():
    pair = simulate_cointegrated_pair(n=3000, names=("NVDA", "CSCO"), seed=5)
    noise = simulate_random_walks(n=3000, names=("ING", "XOM"), seed=6)
    result = screen_pairs(pd.concat([pair, noise], axis=1))
    assert len(result) == 6
    tradable = result[result["tradable"]]
    assert {frozenset((r.y, r.x)) for r in tradable.itertuples()} == {frozenset(("NVDA", "CSCO"))}


def test_input_validation():
    s = pd.Series([1.0, 2.0, np.nan])
    with pytest.raises(ValueError):
        estimate_long_run_short_run_relationships(s, pd.Series([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError):
        estimate_long_run_short_run_relationships([1, 2, 3], pd.Series([1.0, 2.0, 3.0]))
