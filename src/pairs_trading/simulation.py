"""Synthetic cointegrated price paths for testing and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_cointegrated_pair(
    n: int = 2000,
    c: float = 1.7,
    gamma: float = 0.5,
    kappa: float = 0.05,
    spread_vol: float = 0.004,
    x_vol: float = 0.01,
    x0: float = 100.0,
    names: tuple[str, str] = ("Y", "X"),
    seed: int | None = 0,
) -> pd.DataFrame:
    """Simulate prices where ln y = c + gamma ln x + z and z is an AR(1)/OU process.

    ``kappa`` is the mean-reversion speed of the spread per step, so the true
    error-correction coefficient is approximately ``-kappa``.
    """
    rng = np.random.default_rng(seed)
    log_x = np.log(x0) + np.cumsum(rng.normal(0, x_vol, n))
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = (1 - kappa) * z[t - 1] + rng.normal(0, spread_vol)
    log_y = c + gamma * log_x + z
    y_name, x_name = names
    return pd.DataFrame({y_name: np.exp(log_y), x_name: np.exp(log_x)})


def simulate_random_walks(n: int = 2000, names=("A", "B"), vol: float = 0.01, seed=0) -> pd.DataFrame:
    """Independent random walks: a pair that should *not* be cointegrated."""
    rng = np.random.default_rng(seed)
    data = {name: 100 * np.exp(np.cumsum(rng.normal(0, vol, n))) for name in names}
    return pd.DataFrame(data)
