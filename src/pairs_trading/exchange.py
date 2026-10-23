"""A minimal simulated exchange with the same interface the strategy expects.

The original coursework ran against Optiver's Optibook, a private training exchange.
This module reproduces the small part of its interface the strategy uses
(order books, positions, PnL, IOC orders) so the code can be run and tested anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceLevel:
    price: float
    volume: int


@dataclass(frozen=True)
class OrderBook:
    instrument_id: str
    bids: list[PriceLevel] = field(default_factory=list)
    asks: list[PriceLevel] = field(default_factory=list)

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2


@dataclass
class Trade:
    step: int
    instrument_id: str
    side: str
    price: float
    volume: int


class SimulatedExchange:
    """Replays a path of mid prices and quotes a top-of-book around each mid.

    Parameters
    ----------
    mids:
        DataFrame indexed by step with one column of mid prices per instrument.
    half_spread:
        Half bid/ask spread, as a fraction of the mid price.
    depth:
        (low, high) range of the random top-of-book volume.
    position_limit:
        Orders that would take an absolute position above this are rejected.
    tick:
        Prices are rounded to this tick size.
    """

    def __init__(
        self,
        mids: pd.DataFrame,
        half_spread: float = 0.0005,
        depth: tuple[int, int] = (20, 200),
        position_limit: int = 100,
        tick: float = 0.01,
        seed: int | None = 0,
    ) -> None:
        if mids.isnull().any().any():
            raise ValueError("mid prices contain NaN values")
        self.mids = mids.reset_index(drop=True)
        self.half_spread = half_spread
        self.depth = depth
        self.position_limit = position_limit
        self.tick = tick
        self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.positions = {name: 0 for name in mids.columns}
        self.cash = 0.0
        self.trades: list[Trade] = []
        self._books: dict[str, OrderBook] = {}
        self._build_books()

    # -- market state -----------------------------------------------------------------
    @property
    def instruments(self) -> list[str]:
        return list(self.mids.columns)

    @property
    def finished(self) -> bool:
        return self.step_index >= len(self.mids) - 1

    def _round(self, price: float) -> float:
        return round(round(price / self.tick) * self.tick, 10)

    def _build_books(self) -> None:
        row = self.mids.iloc[self.step_index]
        for name, mid in row.items():
            half = max(self.tick, mid * self.half_spread)
            bid, ask = self._round(mid - half), self._round(mid + half)
            if ask <= bid:
                ask = bid + self.tick
            low, high = self.depth
            self._books[name] = OrderBook(
                name,
                bids=[PriceLevel(bid, int(self.rng.integers(low, high + 1)))],
                asks=[PriceLevel(ask, int(self.rng.integers(low, high + 1)))],
            )

    def step(self) -> bool:
        """Advance one time step. Returns False once the price path is exhausted."""
        if self.finished:
            return False
        self.step_index += 1
        self._build_books()
        return True

    # -- Optibook-style API --------------------------------------------------------------
    def get_last_price_book(self, instrument_id: str) -> OrderBook:
        return self._books[instrument_id]

    def get_positions(self) -> dict[str, int]:
        return dict(self.positions)

    def get_pnl(self) -> float:
        value = sum(pos * self._books[name].mid for name, pos in self.positions.items())
        return self.cash + value

    def insert_order(self, instrument_id: str, price: float, volume: int, side: str,
                     order_type: str = "ioc") -> int:
        """Insert an immediate-or-cancel order. Returns the filled volume."""
        if order_type != "ioc":
            raise NotImplementedError("only IOC orders are simulated")
        if side not in ("bid", "ask"):
            raise ValueError(f"invalid side {side!r}")
        volume = int(volume)
        if volume <= 0:
            return 0
        signed = volume if side == "bid" else -volume
        if abs(self.positions[instrument_id] + signed) > self.position_limit:
            return 0

        book = self._books[instrument_id]
        if side == "bid":
            level = book.asks[0]
            if price < level.price:
                return 0
        else:
            level = book.bids[0]
            if price > level.price:
                return 0
        filled = min(volume, level.volume)
        remaining = PriceLevel(level.price, level.volume - filled)
        if side == "bid":
            self._books[instrument_id] = OrderBook(instrument_id, book.bids, [remaining])
            self.cash -= filled * level.price
            self.positions[instrument_id] += filled
        else:
            self._books[instrument_id] = OrderBook(instrument_id, [remaining], book.asks)
            self.cash += filled * level.price
            self.positions[instrument_id] -= filled
        self.trades.append(Trade(self.step_index, instrument_id, side, level.price, filled))
        return filled
