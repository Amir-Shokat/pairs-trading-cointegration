"""Fit a pair on a training window and trade it on a simulated exchange.

    python scripts/run_backtest.py                  # synthetic cointegrated pair
    python scripts/run_backtest.py --yahoo NVDA CSCO  # real daily data (pip install yfinance)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pairs_trading import PairsTrader, SimulatedExchange, StrategyConfig, simulate_cointegrated_pair
from pairs_trading.cointegration import fit_pair


def load_prices(args):
    if args.yahoo:
        import yfinance as yf

        data = yf.download(list(args.yahoo), start=args.start, auto_adjust=True, progress=False)["Close"]
        return data[list(args.yahoo)].dropna()
    return simulate_cointegrated_pair(n=args.n, seed=args.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yahoo", nargs=2, metavar=("Y", "X"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--entry-k", type=float, default=1.0)
    parser.add_argument("--exit-at-mean", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    prices = load_prices(args)
    y, x = prices.columns[:2]
    split = int(len(prices) * args.train_frac)
    train, test = prices.iloc[:split], prices.iloc[split:]

    fit = fit_pair(np.log(train), y, x)
    print(f"Fitted {y} on {x}: c={fit.c:.3f} gamma={fit.gamma:.3f} alpha={fit.alpha:.4f} "
          f"p={fit.p_value:.4f} half-life={fit.half_life:.1f}")
    if fit.p_value > 0.05:
        print("WARNING: the pair is not cointegrated on the training window (p > 0.05); "
              "spread trading results are not meaningful.")

    exchange = SimulatedExchange(test, seed=args.seed)
    config = StrategyConfig(entry_k=args.entry_k, exit_at_mean=args.exit_at_mean)
    log = PairsTrader(exchange, [fit], config).run()
    trades = len(exchange.trades)
    print(f"Out-of-sample steps: {len(test)}  trades: {trades}  final PnL: {exchange.get_pnl():.2f}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(log["step"], log["z"], lw=0.8, label="spread z")
    axes[0].plot(log["step"], log["threshold"], "r--", lw=0.8, label="+threshold")
    axes[0].plot(log["step"], -log["threshold"], "r--", lw=0.8)
    axes[0].legend(); axes[0].set_title(f"Spread {y}/{x}")
    axes[1].plot(log["step"], log[f"pos_{y}"], label=y)
    axes[1].plot(log["step"], log[f"pos_{x}"], label=x)
    axes[1].legend(); axes[1].set_title("Positions")
    axes[2].plot(log["step"], log["pnl"]); axes[2].set_title("PnL")
    fig.tight_layout()
    fig.savefig(out / "backtest.png", dpi=120)
    log.to_csv(out / "backtest_log.csv", index=False)
    print(f"Saved {out / 'backtest.png'}")


if __name__ == "__main__":
    main()
