"""
Re-validate the momentum_breakout config on REAL option premiums for the exact
instrument the live bot trades: the nearest-OTM strike whose one-lot premium fits
the CAPITAL_PER_TRADE cap, at the configured DTE.

Why this exists
---------------
backtester/engine.py reports a *delta-proxy* P&L on the NIFTY underlying
(ATM delta 0.45, zero theta). Live, the bot cannot afford an ATM lot under the
~Rs5k cap, so options_selector buys a far-OTM contract — low delta, high theta.
This script closes that gap: it takes the engine's trade list (entry/exit
timestamps + direction, now using the corrected intrabar exits) and re-prices
each trade with the ACTUAL OTM premium from Kite, via the existing
option_pricer (which already selects the nearest OTM strike under the cap).

Limitation: Kite only serves history for currently-listed contracts, so only
the most recent ~2-3 weeks of signals are priceable. Treat this as a small but
ground-truth recent-window check, not a full-history tune.

Run:  python research/revalidate_otm.py
"""
import sys
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")

import json
import pandas as pd

from indicators import compute_all, resample_ohlc
from backtester.engine import run_backtest
from backtester.option_pricer import enrich_with_real_option_prices
from config import (
    BROKERAGE_PER_ORDER, CAPITAL_PER_TRADE, LOT_SIZE,
    MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY, OPTIMIZED_PARAMS_PATH,
)

PRICEABLE_WINDOW_DAYS = 35   # only attempt to price recent trades

# ── 1. Signals + corrected (intrabar) exits on 15-min bars ───────────────────
params = json.load(open(OPTIMIZED_PARAMS_PATH))
raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)
df15 = compute_all(resample_ohlc(raw, 15))
trades, _daily, m = run_backtest(df15, params=params)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])

print("=== Delta-proxy backtest (underlying, no theta) — full history ===")
print(f"  trades={m['total_trades']}  win={m['win_rate']:.1%}  "
      f"pnl=Rs{m['total_pnl']:,.0f}  PF={m['profit_factor']:.2f}")

# ── 2. Restrict to the priceable recent window ───────────────────────────────
cutoff = trades["entry_time"].max() - pd.Timedelta(days=PRICEABLE_WINDOW_DAYS)
recent = trades[trades["entry_time"] >= cutoff].reset_index(drop=True)
print(f"\nPricing the last {PRICEABLE_WINDOW_DAYS}d: {len(recent)} trades "
      f"({cutoff.date()} .. {trades['entry_time'].max().date()})")

# ── 3. Real OTM-premium P&L (Kite) ───────────────────────────────────────────
from kiteconnect import KiteConnect
from config import KITE_API_KEY, KITE_ACCESS_TOKEN
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)
inst = pd.DataFrame(kite.instruments("NFO"))
inst = inst[inst["name"] == "NIFTY"].copy()
inst["expiry"] = pd.to_datetime(inst["expiry"]).dt.date

priced = enrich_with_real_option_prices(kite, recent, inst, drop_unpriced=True)

print(f"\n=== REAL OTM-option P&L (premium-capped under Rs{CAPITAL_PER_TRADE:.0f}, "
      f"{MIN_DAYS_TO_EXPIRY}-{MAX_DAYS_TO_EXPIRY} DTE) ===")
if priced.empty:
    print("  No trades could be priced from real Kite data in this window.")
    sys.exit(0)

n = len(priced)
wins = (priced["pnl"] > 0).sum()
gross_win = priced.loc[priced["pnl"] > 0, "pnl"].sum()
gross_loss = -priced.loc[priced["pnl"] < 0, "pnl"].sum()
pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
print(f"  priced trades : {n}")
print(f"  win rate      : {wins/n:.1%}")
print(f"  total P&L     : Rs{priced['pnl'].sum():,.0f}")
print(f"  avg P&L/trade : Rs{priced['pnl'].mean():,.0f}")
print(f"  profit factor : {pf:.2f}")
print(f"  avg entry prem: Rs{priced['entry_price'].mean():.1f}  "
      f"(avg DTE {priced['dte'].mean():.1f})")
print(f"  by exit_reason:")
for reason, g in priced.groupby("exit_reason"):
    print(f"    {reason:11s} n={len(g):2d}  win={ (g['pnl']>0).mean():.0%}  "
          f"avg=Rs{g['pnl'].mean():,.0f}")

print("\n  Per-trade detail:")
cols = ["entry_time", "direction", "atm_strike", "dte",
        "entry_price", "exit_price", "exit_reason", "pnl"]
cols = [c for c in cols if c in priced.columns]
with pd.option_context("display.width", 160, "display.max_columns", None):
    print(priced[cols].to_string(index=False))
