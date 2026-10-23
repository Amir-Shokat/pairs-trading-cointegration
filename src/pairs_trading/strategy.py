"""Bid/ask-aware spread trading on cointegrated pairs.

For a pair with long-run relation ln y = c + gamma * ln x + z, two executable spreads
are computed from the live order books:

    z1 = ln(best_bid_y) - c - gamma * ln(best_ask_x)   # value of selling Y, buying X
    z2 = ln(best_ask_y) - c - gamma * ln(best_bid_x)   # value of buying Y, selling X

If z1 is above the threshold, Y is rich relative to X: sell Y and buy X. If z2 is below
minus the threshold, Y is cheap: buy Y and sell X. Using the side of the book we would
actually trade at means a signal is only taken when it survives the bid/ask spread.

The hedge ratio in shares is A = gamma * P_y / P_x, the number of X shares that offset the
log-price exposure of one Y share.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cointegration import PairFit


@dataclass
class StrategyConfig:
    entry_k: float = 1.0           # trade when the spread is entry_k rolling std devs from 0
    window: int = 500              # rolling window for the spread std dev
    min_history: int = 50          # use the in-sample std dev until this many observations
    position_limit: int = 100
    hedge_tolerance: float = 0.2   # flatten the pair if the share ratio drifts >20% from A
    exit_at_mean: bool = False     # close positions when the spread crosses zero


@dataclass
class PairState:
    fit: PairFit
    history: deque = field(default_factory=lambda: deque(maxlen=500))


class PairsTrader:
    def __init__(self, exchange, fits: list[PairFit], config: StrategyConfig | None = None):
        self.exchange = exchange
        self.config = config or StrategyConfig()
        self.pairs = [PairState(f, deque(maxlen=self.config.window)) for f in fits]
        self.log: list[dict] = []

    # -- helpers ----------------------------------------------------------------------
    def threshold(self, state: PairState) -> float:
        cfg = self.config
        if len(state.history) >= cfg.min_history:
            std = float(np.std(state.history, ddof=1))
        else:
            std = state.fit.resid_std
        return cfg.entry_k * std

    def _capacity(self, instrument: str, side: str) -> int:
        pos = self.exchange.get_positions()[instrument]
        limit = self.config.position_limit
        return max(0, limit - pos) if side == "bid" else max(0, limit + pos)

    def _execute_pair(self, y: str, x: str, y_side: str, a: float) -> tuple[int, int]:
        """Trade Y on ``y_side`` and X on the opposite side in the ratio 1 : a."""
        x_side = "bid" if y_side == "ask" else "ask"
        book_y = self.exchange.get_last_price_book(y)
        book_x = self.exchange.get_last_price_book(x)
        y_level = book_y.bids[0] if y_side == "ask" else book_y.asks[0]
        x_level = book_x.asks[0] if x_side == "bid" else book_x.bids[0]

        n_y = int(min(
            y_level.volume,
            x_level.volume / a,
            self._capacity(y, y_side),
            self._capacity(x, x_side) / a,
        ))
        if n_y <= 0 or round(a * n_y) <= 0:
            return 0, 0
        filled_y = self.exchange.insert_order(y, y_level.price, n_y, y_side, "ioc")
        if filled_y == 0:
            return 0, 0
        # size the hedge leg on what actually filled, not on what we asked for
        filled_x = self.exchange.insert_order(x, x_level.price, int(round(a * filled_y)), x_side, "ioc")
        return filled_y, filled_x

    def _flatten(self, instruments) -> None:
        for name in instruments:
            pos = self.exchange.get_positions()[name]
            if pos == 0:
                continue
            book = self.exchange.get_last_price_book(name)
            if pos > 0:
                self.exchange.insert_order(name, book.bids[0].price, pos, "ask", "ioc")
            else:
                self.exchange.insert_order(name, book.asks[0].price, -pos, "bid", "ioc")

    # -- one decision step --------------------------------------------------------------
    def on_tick(self) -> None:
        for state in self.pairs:
            f = state.fit
            book_y = self.exchange.get_last_price_book(f.y)
            book_x = self.exchange.get_last_price_book(f.x)
            if not (book_y.bids and book_y.asks and book_x.bids and book_x.asks):
                continue

            bid_y, ask_y = book_y.bids[0].price, book_y.asks[0].price
            bid_x, ask_x = book_x.bids[0].price, book_x.asks[0].price
            z1 = np.log(bid_y) - f.c - f.gamma * np.log(ask_x)
            z2 = np.log(ask_y) - f.c - f.gamma * np.log(bid_x)
            z_mid = np.log(book_y.mid) - f.c - f.gamma * np.log(book_x.mid)
            threshold = self.threshold(state)
            a = f.gamma * book_y.mid / book_x.mid

            action = None
            pos_y = self.exchange.get_positions()[f.y]
            if self.config.exit_at_mean and pos_y != 0 and np.sign(z_mid) == np.sign(pos_y):
                # short Y was opened when z > 0; close once z has reverted through zero
                self._flatten([f.y, f.x])
                action = "exit"
            elif z1 > threshold:
                self._execute_pair(f.y, f.x, "ask", a)
                action = "sell_spread"
            elif z2 < -threshold:
                self._execute_pair(f.y, f.x, "bid", a)
                action = "buy_spread"

            positions = self.exchange.get_positions()
            if positions[f.y] != 0:
                ratio = abs(positions[f.x] / positions[f.y])
                tol = self.config.hedge_tolerance
                if not (1 - tol) * a <= ratio <= (1 + tol) * a:
                    self._flatten([f.y, f.x])
                    action = "rehedge_flatten"

            state.history.append(z_mid)
            self.log.append({
                "step": self.exchange.step_index, "pair": f"{f.y}/{f.x}", "z": z_mid,
                "threshold": threshold, "action": action,
                **{f"pos_{k}": v for k, v in self.exchange.get_positions().items()},
                "pnl": self.exchange.get_pnl(),
            })

    def run(self) -> pd.DataFrame:
        """Trade until the exchange's price path is exhausted; returns the decision log."""
        while True:
            self.on_tick()
            if not self.exchange.step():
                break
        return pd.DataFrame(self.log)
