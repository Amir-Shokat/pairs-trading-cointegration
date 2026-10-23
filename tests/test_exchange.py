import pandas as pd
import pytest

from pairs_trading import SimulatedExchange


@pytest.fixture
def exchange():
    mids = pd.DataFrame({"A": [100.0, 101.0, 102.0]})
    return SimulatedExchange(mids, half_spread=0.001, depth=(50, 50), position_limit=100, seed=0)


def test_book_brackets_mid(exchange):
    book = exchange.get_last_price_book("A")
    assert book.bids[0].price < 100 < book.asks[0].price


def test_ioc_buy_fills_at_ask_and_updates_cash(exchange):
    ask = exchange.get_last_price_book("A").asks[0].price
    filled = exchange.insert_order("A", ask, 10, "bid")
    assert filled == 10
    assert exchange.get_positions()["A"] == 10
    assert exchange.cash == pytest.approx(-10 * ask)


def test_ioc_does_not_fill_below_ask(exchange):
    bid = exchange.get_last_price_book("A").bids[0].price
    assert exchange.insert_order("A", bid, 10, "bid") == 0


def test_fill_is_capped_by_depth(exchange):
    ask = exchange.get_last_price_book("A").asks[0].price
    assert exchange.insert_order("A", ask, 80, "bid") == 50


def test_position_limit_rejects_order(exchange):
    exchange.positions["A"] = 95
    ask = exchange.get_last_price_book("A").asks[0].price
    assert exchange.insert_order("A", ask, 10, "bid") == 0


def test_pnl_marks_to_mid(exchange):
    ask = exchange.get_last_price_book("A").asks[0].price
    exchange.insert_order("A", ask, 10, "bid")
    exchange.step()
    assert exchange.get_pnl() == pytest.approx(10 * (101.0 - ask))


def test_step_stops_at_end(exchange):
    assert exchange.step() and exchange.step()
    assert not exchange.step()
