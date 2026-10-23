"""Engle-Granger two-step cointegration analysis.

Long run:   ln y_t = c + gamma * ln x_t + z_t
Short run:  ln y_t - ln y_{t-1} = alpha * z_{t-1} + e_t

A pair is tradable when the residual z_t is stationary (it mean-reverts).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class PairFit:
    y: str
    x: str
    c: float
    gamma: float
    alpha: float
    adf_stat: float
    p_value: float
    resid_std: float

    @property
    def half_life(self) -> float:
        """Half-life of a shock to the spread, in observations (inf if alpha >= 0)."""
        if self.alpha >= 0:
            return float("inf")
        return float(np.log(0.5) / np.log(1 + self.alpha))


def _check_inputs(y: pd.Series, x: pd.Series) -> None:
    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        raise TypeError("y and x must be pandas Series")
    if y.isnull().any() or x.isnull().any():
        raise ValueError("input series contain NaN values")
    if not y.index.equals(x.index):
        raise ValueError("y and x must share the same index")


def estimate_long_run_short_run_relationships(y: pd.Series, x: pd.Series):
    """Return (c, gamma, alpha, z) for the two-step error-correction model."""
    _check_inputs(y, x)
    long_run = OLS(y, add_constant(x)).fit()
    c, gamma = long_run.params.iloc[0], long_run.params.iloc[1]
    z = long_run.resid

    short_run = OLS(y.diff().iloc[1:], z.shift().iloc[1:]).fit()
    alpha = short_run.params.iloc[0]
    return float(c), float(gamma), float(alpha), z


def engle_granger_test(y: pd.Series, x: pd.Series, method: str = "mackinnon"):
    """Engle-Granger cointegration test. Returns (test statistic, p-value).

    ``method="mackinnon"`` (default) uses residual-based critical values that account
    for z being estimated. ``method="adf"`` reproduces the plain ADF p-value used in
    the original course code, which is slightly too optimistic in small samples.
    """
    _check_inputs(y, x)
    if method == "mackinnon":
        stat, p_value, _ = coint(y, x, trend="c", maxlag=1, autolag=None)
        return float(stat), float(p_value)
    if method == "adf":
        _, _, _, z = estimate_long_run_short_run_relationships(y, x)
        with warnings.catch_warnings():   # newer statsmodels warns about its return type changing
            warnings.simplefilter("ignore", FutureWarning)
            result = adfuller(z, maxlag=1, autolag=None)
        stat, p_value = result[0], result[1]
        return float(stat), float(p_value)
    raise ValueError(f"unknown method {method!r}")


def fit_pair(log_prices: pd.DataFrame, y: str, x: str, method: str = "mackinnon") -> PairFit:
    c, gamma, alpha, z = estimate_long_run_short_run_relationships(log_prices[y], log_prices[x])
    stat, p_value = engle_granger_test(log_prices[y], log_prices[x], method=method)
    return PairFit(y, x, c, gamma, alpha, stat, p_value, float(z.std()))


def screen_pairs(
    prices: pd.DataFrame,
    significance: float = 0.01,
    log: bool = True,
    method: str = "mackinnon",
) -> pd.DataFrame:
    """Test every pair of columns in ``prices`` and return the results sorted by p-value.

    Each unordered pair is tested in both directions (y on x and x on y), because the
    Engle-Granger test is not symmetric; the direction with the lower p-value is kept.
    """
    data = np.log(prices) if log else prices
    data = data.dropna()
    rows = []
    for a, b in combinations(data.columns, 2):
        fits = [fit_pair(data, a, b, method), fit_pair(data, b, a, method)]
        best = min(fits, key=lambda f: f.p_value)
        rows.append({**best.__dict__, "half_life": best.half_life})
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["tradable"] = out["p_value"] < significance
    return out
