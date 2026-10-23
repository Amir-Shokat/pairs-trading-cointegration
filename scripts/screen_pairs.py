"""Screen every pair in a CSV of prices (one column per stock) for cointegration.

    python scripts/screen_pairs.py prices.csv --significance 0.01
"""

import argparse

import pandas as pd

from pairs_trading import screen_pairs

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="CSV with a date/step index column and one price column per stock")
parser.add_argument("--significance", type=float, default=0.01)
args = parser.parse_args()

prices = pd.read_csv(args.csv, index_col=0)
result = screen_pairs(prices, significance=args.significance)
pd.set_option("display.width", 160)
print(result.round(4).to_string(index=False))
